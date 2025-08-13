# server/analysis.py

import numpy as np
import cv2
import torch
import face_recognition
from collections import OrderedDict, Counter


class FaceReIDTracker:
    """
    Tracks, re-identifies, and manages faces using the 'face_recognition' library.
    """

    def __init__(self, tolerance=0.55):
        self.known_face_encodings = []
        self.known_face_metadata = []
        self.next_person_id = 0
        self.tolerance = tolerance

    def rename_person(self, person_id, new_name):
        for metadata in self.known_face_metadata:
            if metadata['id'] == person_id:
                metadata['name'] = new_name
                return True
        return False

    def merge_persons(self, source_id, target_id, tracking_data):
        """ Merges persons, now also handling the external tracking_data dict. """
        target_meta = next((m for m in self.known_face_metadata if m['id'] == target_id), None)
        if not target_meta:
            return False

        found_source = False
        for meta in self.known_face_metadata:
            if meta['id'] == source_id:
                meta['id'] = target_id
                meta['name'] = target_meta['name']
                found_source = True

        if source_id in tracking_data and target_id in tracking_data:
            if 'emotions' in tracking_data[source_id]:
                tracking_data[target_id].setdefault('emotions', []).extend(tracking_data[source_id]['emotions'])
            if 'engagement' in tracking_data[source_id]:
                tracking_data[target_id].setdefault('engagement', []).extend(tracking_data[source_id]['engagement'])
            del tracking_data[source_id]

        return found_source

    def update(self, rgb_frame):
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        tracked_persons = OrderedDict()

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            metadata = None
            if self.known_face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, self.tolerance)
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    metadata = self.known_face_metadata[best_match_index]

            if metadata is None:
                person_id = self.next_person_id
                self.next_person_id += 1
                new_metadata = {'id': person_id, 'name': f"Person {person_id}"}
                self.known_face_encodings.append(face_encoding)
                self.known_face_metadata.append(new_metadata)
                metadata = new_metadata

            bbox = (left, top, right - left, bottom - top)
            tracked_persons[metadata['id']] = {'bbox': bbox, 'name': metadata['name']}
        return tracked_persons


def generate_ai_narrative_summary(person_name, emotions_sequence, emotion_labels):
    """
    Generates an LLM-like narrative of emotional evolution.
    (This is the exact same function from your app.py, moved here)
    """
    if not emotions_sequence or len(emotions_sequence) < 10:
        return f"Not enough emotional data for {person_name} to generate a meaningful summary."

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


def analyze_frame(frame, emotion_model, tracker, tracking_data, head_pose_data):
    """
    Processes a single frame: tracks faces, classifies emotions, and integrates engagement.
    This function is pure and does not rely on global state.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tracked_persons = tracker.update(rgb_frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # The head_pose_data keys are "0", "1", etc., from MediaPipe's detection order.
    # We'll align them with our tracker's detection order.
    pose_by_index = {int(k): v for k, v in head_pose_data.items()}

    results = []
    # We iterate through tracked_persons in the order they were detected.
    for i, (person_id, data) in enumerate(tracked_persons.items()):
        (x, y, w, h) = data['bbox']
        roi_gray = gray[y:y + h, x:x + w]
        if roi_gray.size == 0: continue

        roi_resized = cv2.resize(roi_gray, (48, 48))
        tensor = torch.from_numpy(roi_resized).to(torch.float32)
        tensor = (tensor / 255.0 - 0.5) * 2.0
        tensor = tensor.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            logits = emotion_model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze()
            confidence, predicted_class = torch.max(probs, 0)

            # Update session tracking data
            if person_id not in tracking_data:
                tracking_data[person_id] = {'emotions': [], 'engagement': []}
            tracking_data[person_id]['emotions'].append(predicted_class.item())

            # Get engagement data for this person based on detection order
            engagement = None
            if i in pose_by_index:
                engagement = pose_by_index[i].get('engagement')
                tracking_data[person_id]['engagement'].append(engagement)

            results.append({
                'id': person_id,
                'name': data['name'],
                'bbox': [int(x), int(y), int(w), int(h)],
                'emotion': predicted_class.item(),
                'confidence': float(confidence.item()),
                'probs': probs.tolist(),
                'engagement': engagement
            })
    return results


def generate_summary_payload(tracking_data, tracker, emotion_labels):
    """Creates the summary dictionary to be sent to the client."""
    summary_payload = {}
    for p_id, data in tracking_data.items():
        if not data.get('emotions'): continue

        meta = next((m for m in tracker.known_face_metadata if m['id'] == p_id), None)
        person_name = meta['name'] if meta else f"Person {p_id}"

        emotions = data['emotions']
        total = len(emotions)
        distribution = [emotions.count(i) / total for i in range(len(emotion_labels))]
        narrative = generate_ai_narrative_summary(person_name, emotions, emotion_labels)

        engagements = [e for e in data.get('engagement', []) if e is not None]
        avg_engagement = round(sum(engagements) / len(engagements), 2) if engagements else None

        summary_payload[p_id] = {
            'id': p_id,
            'name': person_name,
            'narrative_summary': narrative,
            'distribution': distribution,
            'total_detections': total,
            'average_engagement': avg_engagement
        }
    return summary_payload