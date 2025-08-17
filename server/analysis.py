import sys
import numpy as np
import cv2
import torch
import face_recognition
from collections import OrderedDict, Counter
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time
import requests
import json


class FaceReIDTracker:
    """
    Tracks, re-identifies, and manages faces using the 'face_recognition' library.
    """

    def __init__(self, tolerance=0.55):
        """

        :param tolerance: Maximum distance for a face encoding to be considered a match.
        Lower values make recognition stricter. Default is 0.55.
        """
        self.known_face_encodings = []
        self.known_face_metadata = []
        self.next_person_id = 0
        self.tolerance = tolerance

    def rename_person(self, person_id, new_name):
        """
        Rename a tracked person by updating their metadata.
        :param person_id: Unique identifier of the person to rename.
        :param new_name: New display name to assign.
        :return: bool: True if the person was found and renamed, otherwise False.
        """
        person_found = False
        for metadata in self.known_face_metadata:
            if metadata['id'] == person_id:
                metadata['name'] = new_name
                person_found = True
        return person_found

    def merge_persons(self, source_ids, target_id, new_name, tracking_data):
        """
        Merges multiple source persons into a target person and updates their name.
        Also handles the external tracking_data dictionary.
        :param source_ids: IDs of the persons to merge into the target.
        :param target_id: ID of the person to retain as the merged identity.
        :param new_name: New name to assign to the merged identity.
        :param tracking_data: External dictionary with additional person data.
        :return: True if merge was successful, False if the target ID does not exist.
        """
        # Ensure the target person exists in metadata
        target_meta = next((m for m in self.known_face_metadata if m['id'] == target_id), None)
        if not target_meta:
            print(f"Error: Target person with ID {target_id} not found.")
            return False

        # Process each source person
        for source_id in source_ids:
            if source_id == target_id:
                continue  # Cannot merge a person into themselves

            # Merge tracking data
            if source_id in tracking_data and target_id in tracking_data:
                if 'emotions' in tracking_data[source_id]:
                    tracking_data[target_id].setdefault('emotions', []).extend(tracking_data[source_id]['emotions'])
                if 'engagement' in tracking_data[source_id]:
                    tracking_data[target_id].setdefault('engagement', []).extend(tracking_data[source_id]['engagement'])
                del tracking_data[source_id]

            # Update metadata ID for all source entries
            for source_meta in self.known_face_metadata:
                if source_meta['id'] == source_id:
                    source_meta['id'] = target_id

        # After merging, update the name for all entries that now have the target_id
        for meta in self.known_face_metadata:
            if meta['id'] == target_id:
                meta['name'] = new_name

        return True

    def update(self, rgb_frame):
        """
        Detect and track faces in the given frame.

        - Compares detected face encodings against known identities.
        - Assigns existing IDs if a match is found.
        - Creates a new identity when an unknown face is detected.
        :param rgb_frame: Image frame in RGB format.
        :return: Mapping of tracked person IDs to dictionaries with bounding box
                 of the face and display name of the tracked person.
        """
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
                new_metadata = {'id': person_id, 'name': f"P{person_id}"}
                self.known_face_encodings.append(face_encoding)
                self.known_face_metadata.append(new_metadata)
                metadata = new_metadata

            bbox = (left, top, right - left, bottom - top)
            tracked_persons[metadata['id']] = {'bbox': bbox, 'name': metadata['name']}
        return tracked_persons


# ---------------- REMOTE LLM (API-BASED) SETUP ----------------
class RemoteLLM:
    """
    Wrapper for a remote LLM API endpoint.
    """

    def __init__(self, api_key, api_url="https://api.openai.com/v1/chat/completions", model="gpt-3.5-turbo"):
        if not api_key:
            raise ValueError("API key is required for RemoteLLM.")
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def generate_narrative(self, prompt, **gen_overrides):
        """
        Generates a short narrative from the local LLM based on a prompt.

        :param prompt: The input text prompt for the LLM.
        :param gen_overrides: Optional keyword arguments that override default generation parameters
                              (e.g., max_new_tokens, temperature, top_p).
        :return: A generated narrative string with special tokens removed.
        """
        print(f"INFO: Calling remote LLM API ({self.model})...")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant providing concise summaries."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": gen_overrides.get("max_new_tokens", 60),
            "temperature": gen_overrides.get("temperature", 0.5),
        }

        try:
            response = requests.post(self.api_url, headers=self.headers, data=json.dumps(payload), timeout=20)
            response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)

            # Parse the response to extract the generated text (this will vary greatly between different APIs)
            result_text = response.json()['choices'][0]['message']['content']

            print("INFO: Narrative generated successfully from API.")
            return result_text.strip()

        except requests.exceptions.RequestException as e:
            error_message = f"API request failed: {e}"
            print(f"ERROR: {error_message}", file=sys.stderr)
            return f"Error: Could not generate summary due to an API connection issue."
        except (KeyError, IndexError) as e:
            error_message = f"Failed to parse API response: {e}. Response: {response.text}"
            print(f"ERROR: {error_message}", file=sys.stderr)
            return f"Error: Could not understand the response from the API."


# ---------------- LOCAL LLM SETUP ----------------
class LocalLLM:
    """Local LLM wrapper with optional 4-bit quantization (bitsandbytes)."""

    def __init__(
            self,
            model_name="tiiuae/falcon-7b-instruct",
            device=None,
            quantize_4bit=True,
            trust_remote_code=False
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"INFO: Initializing LocalLLM on device: {self.device}")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quantization_config = None
        model_kwargs = {
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "trust_remote_code": trust_remote_code,
        }

        if self.device == "cuda":
            if quantize_4bit:
                print("INFO: Applying 4-bit quantization for CUDA device.")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                model_kwargs["quantization_config"] = quantization_config
            else:
                model_kwargs["torch_dtype"] = torch.float16  # float16 for better performance on GPU if not quantizing
        else:
            print(
                "WARNING: CPU device detected. Quantization is disabled. Model will be loaded in full precision (float32).")

        print(f"INFO: Loading model '{model_name}'... This may take a significant amount of time and memory.")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name
        )
        print("INFO: Model loaded successfully.")

        # Sensible short defaults for fast, concise summaries
        self.generation_defaults = dict(
            max_new_tokens=40,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )

    def generate_narrative(self, prompt, **gen_overrides):
        """

        :param prompt:
        :param gen_overrides:
        :return:
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        gen_cfg = {**self.generation_defaults, **gen_overrides}
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_cfg)

        gen_tokens = output_ids[0, inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        print("INFO: Narrative generated successfully.")
        return text.strip()


def generate_ai_narrative_summary(person_name, emotions_sequence, emotion_labels, llm=None, engagement_sequence=None):
    """
    Generates a narrative summary of a person’s emotional development (and optionally engagement)
    using either a provided LLM or a heuristic fallback method.

    :param person_name: The display name of the person.
    :param emotions_sequence: A list of predicted emotion indices over time.
    :param emotion_labels: A mapping of indices to emotion label strings.
    :param llm: Optional LocalLLM or RemoteLLM instance used to generate the summary.
    :param engagement_sequence: Optional list of engagement scores (floats between 0 and 1).
    :return: A human-readable summary string describing emotional and attentional trends.
    """
    if not emotions_sequence or len(emotions_sequence) < 5:
        return f"Not enough emotional data for {person_name} to generate a meaningful summary."

    avg_engagement = None
    if engagement_sequence:
        valid_engagements = [e for e in engagement_sequence if e is not None]
        if valid_engagements:
            avg_engagement = round(sum(valid_engagements) / len(valid_engagements), 2)

    # ---------- LLM version ----------
    if llm:
        print(f"INFO: Generating summary for {person_name} with LLM.")
        timeline_emotions = ", ".join([emotion_labels[e] for e in emotions_sequence])

        engagement_info = ""
        if avg_engagement is not None:
            engagement_info = (f"\nThe engagement sequence is: {engagement_sequence}"
                               f"with an average engagement score of {avg_engagement} on a scale from 0 to 1.")

        prompt = (
            f"Write a concise, human-readable summary of {person_name}'s emotional and attentional (engagement) development "
            f"over time based on the following emotional sequence: {timeline_emotions}.{engagement_info}\n"
            f"You must not quote the raw sequence itself, but summarize the overall trend.\n"
            f"Summary:"
        )
        print("DEBUG: Prompt for LLM:\n\t" + prompt)
        return llm.generate_narrative(prompt, max_new_tokens=100, temperature=0.2, top_p=0.9)

    # ---------- Heuristic fallback ----------
    counts = Counter(emotions_sequence)
    dominant_mood = emotion_labels[counts.most_common(1)[0][0]]
    start_mood = emotion_labels[emotions_sequence[0]]
    end_mood = emotion_labels[emotions_sequence[-1]]
    unique_emotions = [emotion_labels[e] for e in dict.fromkeys(emotions_sequence)]
    num_shifts = len(unique_emotions) - 1

    narrative = f"{person_name} began the session in a state of **{start_mood.lower()}**."
    if num_shifts == 0:
        narrative += " They appeared to maintain this feeling consistently throughout."
    else:
        if start_mood != end_mood:
            narrative += f" Their emotional journey was dynamic, concluding with a feeling of **{end_mood.lower()}**."
        else:
            narrative += f" Despite experiencing several emotional shifts, they eventually returned to a **{end_mood.lower()}** state."
        narrative += f" The most prevalent emotion observed was **{dominant_mood.lower()}**."
        other_moods = [m for m in unique_emotions if m not in [start_mood, end_mood, dominant_mood]]
        if num_shifts > 2 and other_moods:
            narrative += f" Moments of **{other_moods[0].lower()}** were also noted."

    if avg_engagement is not None:
        if avg_engagement > 0.7:
            narrative += f" Engagement was generally **high** ({avg_engagement})."
        elif avg_engagement > 0.4:
            narrative += f" Engagement was **moderate** ({avg_engagement})."
        else:
            narrative += f" Engagement appeared **low** ({avg_engagement})."

    return narrative.strip()


def analyze_frame(frame, emotion_model, tracker, tracking_data, head_pose_data):
    """
    Analyzes a single video frame to detect and classify emotions, update tracking data,
    and compute engagement if head pose data is available.

    :param frame: The video frame in OpenCV format.
    :param emotion_model: A PyTorch model for emotion recognition.
    :param tracker: A tracking object that assigns IDs and bounding boxes to detected persons.
    :param tracking_data: Dictionary storing tracked persons’ emotional and engagement histories.
    :param head_pose_data: Dictionary mapping tracked indices to head pose/engagement info.
    :return: A list of result dictionaries, each containing person ID, name, bounding box,
             predicted emotion, confidence score, probability distribution, and engagement (if any).
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tracked_persons = tracker.update(rgb_frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pose_by_index = {int(k): v for k, v in head_pose_data.items()}

    results = []
    for i, (person_id, data) in enumerate(tracked_persons.items()):
        (x, y, w, h) = data['bbox']
        roi_gray = gray[y:y + h, x:x + w]
        if roi_gray.size == 0:
            continue

        roi_resized = cv2.resize(roi_gray, (48, 48))
        tensor = torch.from_numpy(roi_resized).to(torch.float32)
        tensor = (tensor / 255.0 - 0.5) * 2.0
        tensor = tensor.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            logits = emotion_model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze()
            confidence, predicted_class = torch.max(probs, 0)

            if person_id not in tracking_data:
                tracking_data[person_id] = {'emotions': [], 'engagement': []}
            tracking_data[person_id]['emotions'].append(predicted_class.item())

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


def generate_summary_payload(tracking_data, tracker, emotion_labels, llm=None):
    """
    Generates a summary payload for all tracked persons, including narrative summaries,
    emotion distributions, and engagement statistics.

    :param tracking_data: Dictionary containing emotional and engagement data for each tracked person.
    :param tracker: Face recognition tracker with metadata about known persons.
    :param emotion_labels: List of emotion label strings (index-aligned with predictions).
    :param llm: Optional LocalLLM or RemoteLLM instance used for narrative generation.
    :return: A dictionary mapping person IDs to summary information, including narrative text,
             emotion distribution, detection count, and average engagement.
    """

    summary_payload = {}
    for p_id, data in tracking_data.items():
        if not data.get('emotions'):
            continue

        meta = next((m for m in tracker.known_face_metadata if m['id'] == p_id), None)
        person_name = meta['name'] if meta else f"Person {p_id}"

        emotions = data['emotions']
        total = len(emotions)
        distribution = [emotions.count(i) / total for i in range(len(emotion_labels))]

        engagements = [e for e in data.get('engagement', []) if e is not None]
        avg_engagement = round(sum(engagements) / len(engagements), 2) if engagements else None

        start_time = time.time()

        narrative = generate_ai_narrative_summary(
            person_name,
            emotions,
            emotion_labels,
            llm=llm,
            engagement_sequence=engagements
        )

        end_time = time.time()

        total_time = end_time - start_time

        if llm is not None:
            print(f"INFO: LLM's processing time: {total_time:.2f} seconds.")

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
