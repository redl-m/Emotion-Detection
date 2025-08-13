# server/app.py

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
import time

# --- Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# --- Local Imports ---
from model.model import EmotionCNN
from server.analysis import FaceReIDTracker, analyze_frame, generate_summary_payload


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

    # --- SocketIO Event Handlers ---
    @app.route('/')
    def index():
        return render_template('index.html')

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

    latest_results = None

    @socketio.on('frame')
    def on_frame(payload):
        nonlocal latest_results
        try:
            image_data = base64.b64decode(payload['data_url'].split(',', 1)[1])
            frame = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

            latest_results = analyze_frame(
                frame=frame,
                emotion_model=emotion_model,
                tracker=tracker,
                tracking_data=tracking_data,
                head_pose_data=payload.get('head_pose', {})
            )
        except Exception as e:
            print(f"Error processing frame: {e}", file=sys.stderr)

    # Background thread to send updates regularly
    def push_updates():
        while True:
            if latest_results is not None:
                socketio.emit('frame_data', json.dumps(latest_results))
            time.sleep(0.1)  # every 100ms → ~10fps

    threading.Thread(target=push_updates, daemon=True).start()

    @socketio.on('get_summary')
    def on_get_summary():
        summary = generate_summary_payload(tracking_data, tracker, emotion_labels)
        emit('tracking_summary', json.dumps(summary))
        print("INFO: Summary generated and sent.")

    return socketio, app


if __name__ == '__main__':
    socketio, app = create_app()
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)