import os
import sys
import json
import base64
import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import torch

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from model.model import EmotionCNN


# --- Simple Centroid Tracker ---
class SimpleCentroidTracker:
    # Increased max_disappeared from 30 to 50 frames.
    # At 5 FPS, this means the tracker will wait up to 10 seconds before
    # deregistering a person who has disappeared from view. This makes
    # tracking more robust to brief obstructions or fast head turns.
    def __init__(self, max_disappeared=50):
        # -----------------------
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return OrderedDict()

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x, y, w, h)) in enumerate(rects):
            input_centroids[i] = (x + int(w / 2), y + int(h / 2))

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = dist.cdist(np.array(object_centroids), input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])
        return self.objects


# --- Global Variables ---
tracker = SimpleCentroidTracker()
tracking_data = {}  # To store emotion history for summaries


def create_app():
    """Creates and configures the Flask application and SocketIO server."""
    # Correctly point to the templates and static folders from the 'server' subdirectory
    app = Flask(
        __name__,
        template_folder=os.path.join(project_root, 'templates'),
        static_folder=os.path.join(project_root, 'static')
    )
    app.config['SECRET_KEY'] = 'secret-emotion-key!'
    socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

    # --- Model and Classifier Loading ---
    model = EmotionCNN()
    model_path = os.path.join(project_root, 'model.pth')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def process_and_classify_frame(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Update tracker with face locations
        tracked_objects = tracker.update(faces_rects)

        results = []
        # Find the corresponding rect for each tracked object
        for (object_id, centroid) in tracked_objects.items():
            # This is a simplification: assumes the closest detected face rect belongs to the tracked centroid
            face_rect = min(faces_rects,
                            key=lambda rect: dist.euclidean(centroid, (rect[0] + rect[2] // 2, rect[1] + rect[3] // 2)))
            (x, y, w, h) = face_rect

            roi_gray = gray[y:y + h, x:x + w]
            roi_resized = cv2.resize(roi_gray, (48, 48))

            tensor = torch.from_numpy(roi_resized).to(torch.float32)
            tensor = tensor / 255.0
            tensor = (tensor - 0.5) * 2
            tensor = tensor.unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                logits = model(tensor)
                probs_tensor = torch.softmax(logits, dim=1).squeeze()
                confidence, predicted_class = torch.max(probs_tensor, 0)

                emotion_idx = predicted_class.item()

                # Store data for summary
                if object_id not in tracking_data:
                    tracking_data[object_id] = {'emotions': []}
                tracking_data[object_id]['emotions'].append(emotion_idx)

                results.append({
                    'id': object_id,
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'emotion': emotion_idx,
                    'confidence': float(confidence.item()),
                    'probs': probs_tensor.tolist()
                })
        return results

    @app.route('/')
    def index():
        return render_template('index.html')

    @socketio.on('start_tracking')
    def on_start_tracking():
        global tracker, tracking_data
        tracker = SimpleCentroidTracker()
        tracking_data.clear()
        print("INFO: Tracking session started and data cleared.")

    @socketio.on('frame')
    def on_frame(data_url):
        try:
            header, encoded = data_url.split(',', 1)
            image_data = base64.b64decode(encoded)
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            results = process_and_classify_frame(frame)
            emit('frame_data', json.dumps(results))
        except Exception as e:
            print(f"Error processing frame: {e}")

    @socketio.on('get_summary')
    def on_get_summary():
        summary_payload = {}
        for p_id, data in tracking_data.items():
            if not data['emotions']: continue

            emotion_counts = [data['emotions'].count(i) for i in range(len(emotion_labels))]
            total_detections = len(data['emotions'])
            distribution = [count / total_detections for count in emotion_counts]
            dominant_emotion_idx = np.argmax(emotion_counts)

            summary_payload[p_id] = {
                'id': p_id,
                'dominant_emotion': emotion_labels[dominant_emotion_idx],
                'distribution': distribution,
                'total_detections': total_detections
            }

        emit('tracking_summary', json.dumps(summary_payload))
        print("INFO: Summary generated and sent.")

    return socketio, app


if __name__ == '__main__':
    socketio, app = create_app()
    socketio.run(app, host='0.0.0.0', port=5000)