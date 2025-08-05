import os
import sys
import json
import base64
import numpy as np
from collections import OrderedDict, Counter
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import torch
import torch.nn as nn
import face_recognition # The modern library for face recognition

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from model.model import EmotionCNN

# --- Global Variables ---
# These are defined globally so they can be accessed by the merge function
tracker = None
tracking_data = {}

# --- Face Re-Identification Tracker (Fully Rewritten) ---
class FaceReIDTracker:
    """
    Tracks, re-identifies, and manages faces using the 'face_recognition' library.
    This new implementation provides persistent identity across video sessions.
    """
    def __init__(self, tolerance=0.55):
        """Initializes the tracker for persistent face recognition."""
        self.known_face_encodings = []
        self.known_face_metadata = [] # Stores {'id': int, 'name': str}
        self.next_person_id = 0
        self.tolerance = tolerance

    def rename_person(self, person_id, new_name):
        """Renames a person identified by their unique ID."""
        for metadata in self.known_face_metadata:
            if metadata['id'] == person_id:
                metadata['name'] = new_name
                return True
        return False

    def merge_persons(self, source_id, target_id):
        """Merges a source person into a target person."""
        target_meta = next((m for m in self.known_face_metadata if m['id'] == target_id), None)
        if not target_meta:
            return False

        # Re-assign all metadata entries for the source ID to the target ID
        found_source = False
        for meta in self.known_face_metadata:
            if meta['id'] == source_id:
                meta['id'] = target_id
                meta['name'] = target_meta['name']
                found_source = True

        # Merge the historical emotion data
        if source_id in tracking_data and target_id in tracking_data:
            tracking_data[target_id]['emotions'].extend(tracking_data[source_id]['emotions'])
            del tracking_data[source_id]
        return found_source

    def update(self, rgb_frame):
        """
        Detects and identifies faces in a frame using their encodings.
        This method replaces the old centroid-based tracking.
        """
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        tracked_persons = OrderedDict()

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            metadata = None
            if self.known_face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, self.tolerance)
                # Find the best match if multiple encodings match the same person
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    metadata = self.known_face_metadata[best_match_index]

            if metadata is None: # It's a new person
                person_id = self.next_person_id
                self.next_person_id += 1
                new_metadata = {'id': person_id, 'name': f"Person {person_id}"}
                self.known_face_encodings.append(face_encoding)
                self.known_face_metadata.append(new_metadata)
                metadata = new_metadata

            bbox = (left, top, right - left, bottom - top)
            tracked_persons[metadata['id']] = {'bbox': bbox, 'name': metadata['name']}
        return tracked_persons

# --- AI Narrative Summary Generator ---
def generate_ai_narrative_summary(person_name, emotions_sequence, emotion_labels):
    """Generates a more insightful, LLM-like narrative of emotional evolution."""
    if not emotions_sequence or len(emotions_sequence) < 10:
        return f"Not enough emotional data was collected for {person_name} to generate a meaningful summary."

    counts = Counter(emotions_sequence)
    dominant_mood = emotion_labels[counts.most_common(1)[0][0]]
    start_mood = emotion_labels[emotions_sequence[0]]
    end_mood = emotion_labels[emotions_sequence[-1]]
    unique_emotions = [emotion_labels[e] for e in dict.fromkeys(emotions_sequence)]
    num_shifts = len(unique_emotions) - 1

    narrative = f"{person_name} began the session in a state of **{start_mood.lower()}**."
    if num_shifts == 0:
        narrative += f" They appeared to maintain this feeling consistently throughout."
    else:
        if start_mood != end_mood:
            narrative += f" Their emotional journey was dynamic, concluding with a feeling of **{end_mood.lower()}**."
        else:
            narrative += f" Despite experiencing several emotional shifts, they eventually returned to a **{end_mood.lower()}** state."
        narrative += f" The most prevalent emotion observed was **{dominant_mood.lower()}**."
        other_moods = [m for m in unique_emotions if m not in [start_mood, end_mood, dominant_mood]]
        if num_shifts > 2 and other_moods:
            narrative += f" Moments of **{other_moods[0].lower()}** were also noted during the interaction."
    return narrative.strip()


# --- Flask Application Factory ---
def create_app():
    """Creates and configures the Flask application and SocketIO server."""
    app = Flask(__name__, template_folder=os.path.join(project_root, 'templates'),
                static_folder=os.path.join(project_root, 'static'))
    app.config['SECRET_KEY'] = 'secret-emotion-key!'
    socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

    # --- Model Loading ---
    emotion_model = EmotionCNN()
    emotion_model_path = os.path.join(project_root, 'model.pth')
    emotion_model.load_state_dict(torch.load(emotion_model_path, map_location=torch.device('cpu')))
    emotion_model.eval()

    # --- Initialize Global Tracker ---
    global tracker
    tracker = FaceReIDTracker(tolerance=0.55)

    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def process_and_classify_frame(frame):
        # 1. Convert BGR (from OpenCV) to RGB (for face_recognition library)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 2. Get recognized persons from the tracker
        tracked_persons = tracker.update(rgb_frame)

        # 3. Prepare grayscale frame for emotion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        results = []
        for person_id, data in tracked_persons.items():
            (x, y, w, h) = data['bbox']
            roi_gray = gray[y:y + h, x:x + w]
            if roi_gray.size == 0: continue

            roi_resized = cv2.resize(roi_gray, (48, 48))
            tensor = torch.from_numpy(roi_resized).to(torch.float32)
            tensor = (tensor / 255.0 - 0.5) * 2
            tensor = tensor.unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                logits = emotion_model(tensor)
                probs = torch.softmax(logits, dim=1).squeeze()
                confidence, predicted_class = torch.max(probs, 0)

                if person_id not in tracking_data:
                    tracking_data[person_id] = {'emotions': []}
                tracking_data[person_id]['emotions'].append(predicted_class.item())

                results.append({
                    'id': person_id,
                    'name': data['name'], # Include the person's name
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'emotion': predicted_class.item(),
                    'confidence': float(confidence.item()),
                    'probs': probs.tolist()
                })
        return results

    # --- SocketIO Event Handlers ---
    @app.route('/')
    def index():
        return render_template('index.html')

    @socketio.on('client_ready')
    def on_client_ready():
        """When a client connects, send them the current list of known people."""
        if tracker:
            emit('known_faces_update', tracker.known_face_metadata)

    @socketio.on('start_tracking')
    def on_start_tracking():
        """Clears per-clip data but preserves the tracker's known faces."""
        global tracking_data
        tracking_data.clear()
        print("INFO: New clip started. Emotion tracking data cleared, known faces preserved.")

    @socketio.on('rename_person')
    def on_rename_person(data):
        person_id, new_name = data.get('id'), data.get('name')
        if person_id is not None and new_name and tracker.rename_person(int(person_id), new_name):
            emit('known_faces_update', tracker.known_face_metadata, broadcast=True)

    @socketio.on('merge_persons')
    def on_merge_persons(data):
        source_id, target_id = data.get('source_id'), data.get('target_id')
        if source_id is not None and target_id is not None and tracker.merge_persons(int(source_id), int(target_id)):
            emit('merge_notification', {'source_id': int(source_id), 'target_id': int(target_id)}, broadcast=True)
            emit('known_faces_update', tracker.known_face_metadata, broadcast=True)

    @socketio.on('frame')
    def on_frame(data_url):
        try:
            image_data = base64.b64decode(data_url.split(',', 1)[1])
            frame = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
            results = process_and_classify_frame(frame)
            emit('frame_data', json.dumps(results))
        except Exception as e:
            print(f"Error processing frame: {e}", file=sys.stderr)

    @socketio.on('get_summary')
    def on_get_summary():
        summary_payload = {}
        for p_id, data in tracking_data.items():
            if not data['emotions']: continue
            meta = next((m for m in tracker.known_face_metadata if m['id'] == p_id), None)
            person_name = meta['name'] if meta else f"Person {p_id}"

            total = len(data['emotions'])
            distribution = [data['emotions'].count(i) / total for i in range(len(emotion_labels))]
            narrative = generate_ai_narrative_summary(person_name, data['emotions'], emotion_labels)

            summary_payload[p_id] = {
                'id': p_id,
                'name': person_name,
                'narrative_summary': narrative,
                'distribution': distribution,
                'total_detections': total
            }
        emit('tracking_summary', json.dumps(summary_payload))
        print("INFO: Summary generated and sent.")

    return socketio, app

if __name__ == '__main__':
    socketio, app = create_app()
    # For newer versions of Flask/Werkzeug, you might need allow_unsafe_werkzeug=True
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)