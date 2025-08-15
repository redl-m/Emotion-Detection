import os
import sys
import json
import base64
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import torch
import threading

# --- Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# --- Local Imports ---
from model.model import EmotionCNN
from server.analysis import FaceReIDTracker, LocalLLM, RemoteLLM, analyze_frame, generate_summary_payload

# --- Global State for the Worker Thread ---
frame_lock = threading.Lock()
latest_frame_data = None

# --- Global LLM instances (lazy loaded) ---
local_llm = None
remote_llm = None
llm_lock = threading.Lock()  # Lock to prevent multiple threads from loading models at once

# --- API Key Storage ---
# This will be stored in memory for the lifetime of the server.
# Set it here if you want a default key.
LLM_API_KEY = None


def create_app():
    """Creates and infigures the Flask application and SocketIO server."""
    app = Flask(__name__, template_folder=os.path.join(project_root, 'templates'),
                static_folder=os.path.join(project_root, 'static'))
    app.config['SECRET_KEY'] = 'secret-emotion-key!'
    socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

    # --- Server State ---
    # Load the ML model
    emotion_model = EmotionCNN()
    emotion_model_path = os.path.join(project_root, 'model.pth')
    emotion_model.load_state_dict(torch.load(emotion_model_path, map_location=torch.device('cpu')))
    emotion_model.eval()

    # Initialize Tracker and session data
    tracker = FaceReIDTracker(tolerance=0.55)
    tracking_data = {}

    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    # --- Worker Thread for Frame Processing ---
    def analysis_worker():
        """
        Runs in a background thread, continuously processing the latest available frame.
        """
        global latest_frame_data
        print("INFO: Analysis worker thread started.")
        while True:
            frame_to_process = None
            head_pose_to_process = None

            with frame_lock:
                if latest_frame_data is not None:
                    frame_to_process, head_pose_to_process = latest_frame_data
                    latest_frame_data = None

            if frame_to_process is not None:
                try:
                    results = analyze_frame(
                        frame=frame_to_process,
                        emotion_model=emotion_model,
                        tracker=tracker,
                        tracking_data=tracking_data,
                        head_pose_data=head_pose_to_process
                    )
                    if results:
                        socketio.emit('frame_data', json.dumps(results))
                except Exception as e:
                    print(f"Error in analysis worker: {e}", file=sys.stderr)

            socketio.sleep(0.1)

    # --- SocketIO Event Handlers ---
    @app.route('/')
    def index():
        return render_template('index.html')

    @socketio.on('connect')
    def on_connect():
        if not hasattr(app, 'analysis_thread_started'):
            socketio.start_background_task(target=analysis_worker)
            app.analysis_thread_started = True
            print("INFO: Client connected, background worker started.")

    # --- API KEY MANAGEMENT HANDLERS ---
    @socketio.on('get_api_key_status')
    def on_get_api_key_status():
        """Client is asking if the API key is set."""
        global LLM_API_KEY
        print("DEBUG: status: " + str(LLM_API_KEY is not None and LLM_API_KEY != ""))
        emit('api_key_status_update', {'is_set': LLM_API_KEY is not None and LLM_API_KEY != ""})

    @socketio.on('set_api_key')
    def on_set_api_key(data):
        """Client is setting a new API key."""
        global LLM_API_KEY, remote_llm
        new_key = data.get('key')

        if new_key and new_key.strip():
            LLM_API_KEY = new_key
            print("INFO: API Key has been set by the user.")
            # Invalidate the old remote_llm instance so it gets recreated with the new key
            with llm_lock:
                remote_llm = None
        else:
            LLM_API_KEY = None
            print("INFO: API Key has been cleared.")

        emit('api_key_status_update', {'is_set': LLM_API_KEY is not None})

    @socketio.on('client_ready')
    def on_client_ready():
        emit('known_faces_update', tracker.known_face_metadata)

    @socketio.on('start_tracking')
    def on_start_tracking():
        nonlocal tracking_data
        tracking_data.clear()
        print("INFO: New clip started. Emotion tracking data cleared.")

    @socketio.on('rename_person')
    def on_rename_person(data):
        person_id, new_name = data.get('id'), data.get('name')
        if person_id is not None and new_name and tracker.rename_person(int(person_id), new_name):
            emit('known_faces_update', tracker.known_face_metadata, broadcast=True)

    @socketio.on('merge_persons')
    def on_merge_persons(data):
        source_id, target_id = int(data['source_id']), int(data['target_id'])
        if tracker.merge_persons(source_id, target_id, tracking_data):
            emit('merge_notification', {'source_id': source_id, 'target_id': target_id}, broadcast=True)
            emit('known_faces_update', tracker.known_face_metadata, broadcast=True)

    @socketio.on('frame')
    def on_frame(payload):
        global latest_frame_data
        try:
            image_data = base64.b64decode(payload['data_url'].split(',', 1)[1])
            frame = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

            with frame_lock:
                latest_frame_data = (frame, payload.get('head_pose', {}))
        except Exception as e:
            print(f"Error decoding frame in on_frame: {e}", file=sys.stderr)

    @socketio.on('get_summary')
    def on_get_summary(data):
        global local_llm, remote_llm, LLM_API_KEY

        use_llm_mode = int(data.get("use_llm", 0))  # Default to 0 (heuristic)
        active_llm = None  # This will hold the llm instance to be used

        print(f"INFO: Summary requested with mode: {use_llm_mode}")

        # --- LLM INSTANCE SELECTION AND LAZY LOADING ---
        with llm_lock:
            if use_llm_mode == 1:  # Use Local LLM
                if local_llm is None:
                    print("INFO: Initializing and loading the local LLM...")
                    emit('summary_status', {'status': 'loading_model', 'message': 'Local AI model is loading...'})
                    try:
                        local_llm = LocalLLM()
                        print("INFO: Local LLM loaded successfully.")
                    except Exception as e:
                        print(f"FATAL: Failed to load local LLM: {e}", file=sys.stderr)
                        emit('summary_status', {'status': 'error', 'message': f'Failed to load Local AI model: {e}'})
                        return
                active_llm = local_llm

            elif use_llm_mode == 2:  # Use Remote LLM via API
                if not LLM_API_KEY:
                    print("WARN: API key requested, but none is set. Prompting user.")
                    emit('request_api_key') # Requesting API key from frontend
                    emit('summary_status',
                         {'status': 'error', 'message': 'API Key is not set. Please set the key and try again.'})
                    return

                if remote_llm is None:
                    print("INFO: Initializing remote LLM client with API key...")
                    try:
                        # You can customize the URL or model name here if needed
                        remote_llm = RemoteLLM(api_key=LLM_API_KEY)
                        print("INFO: Remote LLM client is ready.")
                    except Exception as e:
                        print(f"ERROR: Failed to initialize Remote LLM client: {e}", file=sys.stderr)
                        emit('summary_status', {'status': 'error', 'message': f'Failed to setup API client: {e}'})
                        return
                active_llm = remote_llm

        # --- SUMMARY GENERATION ---
        if use_llm_mode > 0:
            emit('summary_status', {'status': 'generating', 'message': 'Model loaded. Generating summary...'})
        else:
            emit('summary_status', {'status': 'generating', 'message': 'Generating heuristic summary...'})

        print("INFO: Generating summary payload...")
        summary_payload = generate_summary_payload(
            tracking_data,
            tracker,
            emotion_labels,
            llm=active_llm
        )
        emit('tracking_summary', json.dumps(summary_payload))
        print("INFO: Summary generated and sent.")

    return socketio, app


if __name__ == '__main__':
    socketio, app = create_app()
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)