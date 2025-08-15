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
from server.analysis import FaceReIDTracker, LocalLLM, analyze_frame, generate_summary_payload

# --- Global State for the Worker Thread ---
frame_lock = threading.Lock()
latest_frame_data = None

# --- Global LLM instance (lazy loaded) ---
local_llm = None
llm_lock = threading.Lock()  # Lock to prevent multiple threads from loading the model at once


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
        global local_llm

        use_llm = data.get("use_llm")

        print("INFO: LLM used: " + str(use_llm))

        local_llm = None # default

        # --- LAZY LOADING THE LLM ---
        with llm_lock: # Ensure the model is only loaded once if multiple requests arrive simultaneously
            if local_llm is None and use_llm == 1:
                print("INFO: First summary request. Initializing and loading the local LLM...")

                emit('summary_status', {'status': 'loading_model',
                                        'message': 'AI model is loading for the first time. This may take a minute...'})
                try:
                    local_llm = LocalLLM(
                        model_name="tiiuae/falcon-7b-instruct",
                        quantize_4bit=True
                    )
                    print("INFO: Local LLM has been successfully loaded into memory.")
                except Exception as e:
                    print(f"FATAL: Failed to load the local LLM: {e}", file=sys.stderr)
                    emit('summary_status', {'status': 'error', 'message': f'Failed to load AI model: {e}'})
                    return  # Abort if model fails to load

        print("INFO: Generating summary payload...")
        emit('summary_status', {'status': 'generating', 'message': 'Model loaded. Generating summary...'})

        summary_payload = generate_summary_payload(
            tracking_data,
            tracker,
            emotion_labels,
            llm=local_llm
        )
        emit('tracking_summary', json.dumps(summary_payload))
        print("INFO: Summary generated and sent via local LLM.")

    return socketio, app


if __name__ == '__main__':
    socketio, app = create_app()
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)