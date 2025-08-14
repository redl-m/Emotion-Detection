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
from server.analysis import FaceReIDTracker, analyze_frame, generate_summary_payload

# --- Global State for the Worker Thread ---
# Ensures thread-safe access to the latest_frame.
frame_lock = threading.Lock()
# Holds the most recent frame received from the client.
latest_frame_data = None


def create_app():
    """Creates and configures the Flask application and SocketIO server."""
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
        This prevents the SocketIO event handlers from being blocked by long-running analysis tasks.
        """
        global latest_frame_data
        print("INFO: Analysis worker thread started.")
        while True:
            frame_to_process = None
            head_pose_to_process = None

            # Thread-safely get the latest frame and clear the global variable
            with frame_lock:
                if latest_frame_data is not None:
                    frame_to_process, head_pose_to_process = latest_frame_data
                    latest_frame_data = None  # Consume the frame

            if frame_to_process is not None:
                try:
                    # Heavy computation in the background
                    results = analyze_frame(
                        frame=frame_to_process,
                        emotion_model=emotion_model,
                        tracker=tracker,
                        tracking_data=tracking_data,
                        head_pose_data=head_pose_to_process
                    )
                    # Emit results directly from the worker
                    if results:
                        socketio.emit('frame_data', json.dumps(results))

                except Exception as e:
                    print(f"Error in analysis worker: {e}", file=sys.stderr)

            # Control the maximum processing rate. This yields control and
            # prevents the CPU from being pegged at 100% if processing is fast.
            # A value of 0.1 aims for ~10 FPS analysis rate.
            socketio.sleep(0.1)

    # --- SocketIO Event Handlers ---
    @app.route('/')
    def index():
        return render_template('index.html')

    @socketio.on('connect')
    def on_connect():
        """Start the background worker when a client connects."""
        # Use a global flag to ensure the thread is only started once.
        if not hasattr(app, 'analysis_thread_started'):
            socketio.start_background_task(target=analysis_worker)
            app.analysis_thread_started = True

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
        """
        This handler only decodes the frame and places it in a shared variable
        for the worker thread to pick up.
        It overwrites the previous frame if the worker hasn't processed it yet,
        which is the desired behavior to prevent a backlog.
        """
        global latest_frame_data
        try:
            image_data = base64.b64decode(payload['data_url'].split(',', 1)[1])
            frame = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

            # Safely update the global frame variable
            with frame_lock:
                latest_frame_data = (frame, payload.get('head_pose', {}))

        except Exception as e:
            # This might happen if the payload is malformed, just log and continue
            print(f"Error decoding frame in on_frame: {e}", file=sys.stderr)

    @socketio.on('get_summary')
    def on_get_summary():
        summary = generate_summary_payload(tracking_data, tracker, emotion_labels)
        emit('tracking_summary', json.dumps(summary))
        print("INFO: Summary generated and sent.")

    return socketio, app


if __name__ == '__main__':
    socketio, app = create_app()
    # allow_unsafe_werkzeug is only for development with the Flask dev server.
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)