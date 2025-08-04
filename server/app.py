import os
import sys
import json
import base64
import numpy as np
from collections import OrderedDict, Counter
from scipy.spatial import distance as dist
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import torch
import torch.nn as nn

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from model.model import EmotionCNN


# --- Face Embedding Model ---
class FaceEmbeddingModel(nn.Module):
    """
    A simplified CNN to generate face embeddings.
    This model takes a 96x96 grayscale face image and outputs a 128-dimensional embedding vector.
    For effective re-identification, this model should be trained on a large dataset of faces
    (e.g., using a triplet loss function). For this implementation, we assume that a
    'face_embedding_model.pth' file with pre-trained weights exists in the root directory.
    """

    def __init__(self):
        super(FaceEmbeddingModel, self).__init__()
        # Input shape: (N, 1, 96, 96)
        self.convnet = nn.Sequential(
            nn.Conv2d(1, 32, 5), nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 5), nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 5), nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        # The flattened size will be 128 * 8 * 8 = 8192
        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256), nn.PReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        """Generates a 128-d embedding for a face."""
        output = self.convnet(x)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        # L2-normalize the embedding
        output = nn.functional.normalize(output, p=2, dim=1)
        return output


# --- Face Re-Identification Tracker ---
class FaceReIDTracker:
    """
    A tracker that assigns a persistent ID to each person using face embeddings.
    It "remembers" faces it has seen before, even if they leave and re-enter the frame.
    """

    def __init__(self, embedding_model, max_disappeared=50, reid_threshold=0.75):
        """
        Initializes the tracker.
        Args:
            embedding_model: The pre-trained model for generating face embeddings.
            max_disappeared (int): The number of consecutive frames a person can be
                                   missing before they are deregistered from active tracking.
            reid_threshold (float): The cosine similarity threshold for re-identification.
                                    A lower value means a stricter match is required.
        """
        self.embedding_model = embedding_model
        self.embedding_model.eval()

        self.next_person_id = 0
        self.known_faces = OrderedDict()  # Stores {person_id: average_embedding}
        self.active_tracks = OrderedDict()  # Stores {person_id: {centroid, disappeared}}

        self.max_disappeared = max_disappeared
        self.reid_threshold = reid_threshold

    def _get_embedding(self, frame, rect):
        """Extracts a face ROI, preprocesses it, and returns its embedding."""
        (x, y, w, h) = rect
        roi = frame[y:y + h, x:x + w]
        roi_resized = cv2.resize(roi, (96, 96))

        tensor = torch.from_numpy(roi_resized).to(torch.float32)
        tensor = tensor / 255.0
        tensor = (tensor - 0.5) * 2  # Normalize to [-1, 1]
        tensor = tensor.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            embedding = self.embedding_model(tensor).squeeze()
        return embedding

    def _update_known_face(self, person_id, new_embedding):
        """Updates the stored embedding for a known person using a moving average."""
        alpha = 0.25  # Learning rate for embedding update
        if person_id in self.known_faces:
            self.known_faces[person_id] = (1 - alpha) * self.known_faces[person_id] + alpha * new_embedding
        else:
            self.known_faces[person_id] = new_embedding

    def update(self, gray_frame, rects):
        """
        Updates the tracker with new face detections from a frame.
        Args:
            gray_frame: The grayscale video frame.
            rects: A list of bounding boxes for detected faces.
        Returns:
            An OrderedDict of {person_id: bbox} for currently tracked persons.
        """
        if len(rects) == 0:
            for person_id in list(self.active_tracks.keys()):
                self.active_tracks[person_id]['disappeared'] += 1
                if self.active_tracks[person_id]['disappeared'] > self.max_disappeared:
                    del self.active_tracks[person_id]
            return OrderedDict()

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x, y, w, h)) in enumerate(rects):
            input_centroids[i] = (x + w // 2, y + h // 2)

        # If no one is being tracked, try to identify or register all new faces
        if len(self.active_tracks) == 0:
            for i, rect in enumerate(rects):
                embedding = self._get_embedding(gray_frame, rect)
                best_match_id = -1
                highest_sim = -1

                # Compare with all known faces
                for p_id, known_embedding in self.known_faces.items():
                    similarity = torch.dot(embedding, known_embedding).item()
                    if similarity > self.reid_threshold and similarity > highest_sim:
                        highest_sim = similarity
                        best_match_id = p_id

                if best_match_id != -1:  # Re-identified a known person
                    person_id = best_match_id
                else:  # Register as a new person
                    person_id = self.next_person_id
                    self.next_person_id += 1

                self.active_tracks[person_id] = {'centroid': input_centroids[i], 'disappeared': 0}
                self._update_known_face(person_id, embedding)
        else:
            # Match existing tracks with new detections based on centroid distance
            active_ids = list(self.active_tracks.keys())
            active_centroids = [d['centroid'] for d in self.active_tracks.values()]

            D = dist.cdist(np.array(active_centroids), input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                person_id = active_ids[row]
                self.active_tracks[person_id]['centroid'] = input_centroids[col]
                self.active_tracks[person_id]['disappeared'] = 0
                used_rows.add(row)
                used_cols.add(col)

            # Handle disappeared and newly appeared faces
            unused_rows = set(range(len(active_centroids))).difference(used_rows)
            unused_cols = set(range(len(input_centroids))).difference(used_cols)

            for row in unused_rows:
                person_id = active_ids[row]
                self.active_tracks[person_id]['disappeared'] += 1
                if self.active_tracks[person_id]['disappeared'] > self.max_disappeared:
                    del self.active_tracks[person_id]

            for col in unused_cols:
                # This is a new face in the frame, identify or register it
                rect = rects[col]
                embedding = self._get_embedding(gray_frame, rect)
                best_match_id = -1
                highest_sim = -1

                for p_id, known_embedding in self.known_faces.items():
                    # Do not match with people already actively tracked
                    if p_id in self.active_tracks and self.active_tracks[p_id]['disappeared'] == 0:
                        continue
                    similarity = torch.dot(embedding, known_embedding).item()
                    if similarity > self.reid_threshold and similarity > highest_sim:
                        highest_sim = similarity
                        best_match_id = p_id

                if best_match_id != -1:
                    person_id = best_match_id
                else:
                    person_id = self.next_person_id
                    self.next_person_id += 1

                self.active_tracks[person_id] = {'centroid': input_centroids[col], 'disappeared': 0}
                self._update_known_face(person_id, embedding)

        # Build the result dictionary {person_id: bbox}
        final_objects = OrderedDict()
        for person_id, data in self.active_tracks.items():
            # Find the rect that is closest to the tracked centroid
            final_rect = min(rects,
                             key=lambda r: dist.euclidean(data['centroid'], (r[0] + r[2] // 2, r[1] + r[3] // 2)))
            final_objects[person_id] = final_rect
        return final_objects


# --- Global Variables ---
tracker = None  # Will be initialized in create_app
tracking_data = {}  # To store emotion history for summaries


def generate_narrative_summary(emotions_sequence, emotion_labels):
    """
    Generates a brief, human-like summary of emotional changes over time.

    Args:
        emotions_sequence (list): A list of detected emotion indices.
        emotion_labels (list): A list of emotion names corresponding to the indices.

    Returns:
        str: A narrative summary of the emotional evolution.
    """
    if not emotions_sequence or len(emotions_sequence) < 10:
        return "Not enough data for a meaningful summary."

    n = len(emotions_sequence)
    # Split the data into beginning, middle, and end sections
    start_idx = n // 4
    end_idx = n - (n // 4)

    start_emotions = emotions_sequence[:start_idx]
    middle_emotions = emotions_sequence[start_idx:end_idx]
    end_emotions = emotions_sequence[end_idx:]

    # Find the most common emotion in each part
    try:
        start_mood = emotion_labels[Counter(start_emotions).most_common(1)[0][0]] if start_emotions else None
        middle_mood = emotion_labels[Counter(middle_emotions).most_common(1)[0][0]] if middle_emotions else None
        end_mood = emotion_labels[Counter(end_emotions).most_common(1)[0][0]] if end_emotions else None
        overall_mood = emotion_labels[Counter(emotions_sequence).most_common(1)[0][0]]
    except IndexError:
        return "Could not determine a consistent emotional pattern."

    # Build the narrative
    if start_mood == end_mood and start_mood == overall_mood:
        return f"The person consistently appeared to be {overall_mood.lower()} throughout the session."

    narrative = f"Initially, the person seemed to be feeling {start_mood.lower()}." if start_mood else ""

    if end_mood and end_mood != start_mood:
        narrative += f" Over time, their mood shifted towards {end_mood.lower()}."
    elif middle_mood and middle_mood != start_mood:
        narrative += f" Their mood then transitioned to feeling {middle_mood.lower()}."

    narrative += f" Overall, their dominant emotion was {overall_mood.lower()}."
    return narrative.strip()


def create_app():
    """Creates and configures the Flask application and SocketIO server."""
    app = Flask(
        __name__,
        template_folder=os.path.join(project_root, 'templates'),
        static_folder=os.path.join(project_root, 'static')
    )
    app.config['SECRET_KEY'] = 'secret-emotion-key!'
    socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

    # --- Model and Classifier Loading ---
    emotion_model = EmotionCNN()
    emotion_model_path = os.path.join(project_root, 'model.pth')
    emotion_model.load_state_dict(torch.load(emotion_model_path, map_location=torch.device('cpu')))
    emotion_model.eval()

    # Load the Face Embedding Model
    embedding_model = FaceEmbeddingModel()
    embedding_model_path = os.path.join(project_root, 'face_embedding_model.pth')
    try:
        embedding_model.load_state_dict(torch.load(embedding_model_path, map_location=torch.device('cpu')))
        print("INFO: Face embedding model loaded successfully.")
    except FileNotFoundError:
        print("WARNING: 'face_embedding_model.pth' not found. Face re-identification will not be reliable.")
        # In a real scenario, you might exit or use a fallback. Here, we'll continue with random weights.
    embedding_model.eval()

    # Initialize the global tracker here after models are loaded
    global tracker
    tracker = FaceReIDTracker(embedding_model)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def process_and_classify_frame(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Update tracker with face locations and get persistent IDs
        tracked_persons = tracker.update(gray, faces_rects)

        results = []
        for (person_id, (x, y, w, h)) in tracked_persons.items():
            roi_gray = gray[y:y + h, x:x + w]
            roi_resized = cv2.resize(roi_gray, (48, 48))

            tensor = torch.from_numpy(roi_resized).to(torch.float32)
            tensor = tensor / 255.0
            tensor = (tensor - 0.5) * 2
            tensor = tensor.unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                logits = emotion_model(tensor)
                probs_tensor = torch.softmax(logits, dim=1).squeeze()
                confidence, predicted_class = torch.max(probs_tensor, 0)
                emotion_idx = predicted_class.item()

                if person_id not in tracking_data:
                    tracking_data[person_id] = {'emotions': []}
                tracking_data[person_id]['emotions'].append(emotion_idx)

                results.append({
                    'id': person_id,
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
        # Re-initialize tracker and data stores for a new session
        tracker.__init__(embedding_model)  # Reset tracker state but keep the model
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

            # Generate the new narrative summary
            narrative = generate_narrative_summary(data['emotions'], emotion_labels)

            summary_payload[p_id] = {
                'id': p_id,
                'narrative_summary': narrative,
                'distribution': distribution,
                'total_detections': total_detections
            }

        emit('tracking_summary', json.dumps(summary_payload))
        print("INFO: Summary generated and sent.")

    return socketio, app


if __name__ == '__main__':
    socketio, app = create_app()
    socketio.run(app, host='0.0.0.0', port=5000)