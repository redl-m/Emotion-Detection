import os
import sys
import threading
import traceback
import io
import re

import huggingface_hub
import numpy as np
import cv2
import torch
import face_recognition
from collections import OrderedDict, Counter

from huggingface_hub import hf_hub_download
from huggingface_hub.hf_api import RepoFile
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria
import time
import requests
import json
from huggingface_hub import HfApi

from server.extensions import LLM_API_KEY, DEFAULT_LLM_API_URL, DEFAULT_LOCAL_LLM_MODEL_NAME, APP_STATE


def llm_process_worker(task_queue, result_queue, status_queue, model_name):
    """
    This function runs in a separate process. It initializes a LocalLLM
    and then enters a loop, waiting for tasks and putting results back.
    """
    try:
        print(f"[Worker-{os.getpid()}] Starting to load model: {model_name}")
        local_llm = LocalLLM(model_name=model_name, status_queue=status_queue)
        print(f"[Worker-{os.getpid()}] Model loaded successfully.")

        while True:
            task_data = task_queue.get()
            print(f"DEBUG: [Worker-{os.getpid()}] Current task data: {task_data}")
            if task_data is None:
                break

            print(f"[Worker-{os.getpid()}] Received generation task.")

            # Pop 'person_id' from the task data: keep it to send back with the result
            person_id = task_data.pop('person_id', None)

            # Call the summary function with the remaining arguments.
            status_queue.put({
                'type': 'status',
                'payload': {'status': 'generating', 'message': f'Generating summary.'}
            })
            summary_text = generate_ai_narrative_summary(**task_data, llm=local_llm)

            # Put the result back in a dictionary, now including the person_id.
            result_payload = {
                "summary": summary_text,
                "person_id": person_id
            }
            result_queue.put(result_payload)

            # print("DEBUG: Entry added to queue: " + str(result_payload))  # The summary gets sent correctly

            print(f"[Worker-{os.getpid()}] Task complete, result sent.")

    except Exception as e:
        error_message = f"LLM Worker Process Error: {e}\n{traceback.format_exc()}"
        person_id_for_error = task_data.get('person_id') if 'task_data' in locals() else 'unknown'
        result_queue.put({"error": error_message, "person_id": person_id_for_error})
        print(error_message, file=sys.stderr)


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


class TqdmProgressCapturer(io.TextIOBase):
    """
    A file-like object that captures stdout, parses tqdm progress,
    and sends throttled updates to a queue.
    """
    def __init__(self, status_queue, file_info):
        self.status_queue = status_queue
        self.file_info = file_info
        self.original_stdout = sys.stdout
        self.last_percent = -1
        # Specific regex to capture the percentage from tqdm's output
        self.percent_regex = re.compile(r"(\d+)%\|")

    def write(self, s):
        # Write the output to the actual console first so the user can see it
        self.original_stdout.write(s)

        # Try to find a percentage in the string
        match = self.percent_regex.search(s)
        if match:
            percent_str = match.group(1)
            percent = int(percent_str)

            # Throttle updates: only send if the percentage has changed
            if percent > self.last_percent:
                self.last_percent = percent
                # TODO: both print statement and status queue entry never get written -> percentage does not get emitted
                self.original_stdout.write(f"INFO: Percentage has been updated to: {percent}%.\n")
                self.status_queue.put({
                    'type': 'status',
                    'payload': {
                        'status': 'file_downloading',
                        'message': f"Downloading {self.file_info.path}",
                        'percent_file': percent,
                        'filename': self.file_info.path
                    }
                })
        # The write method should return the number of characters written
        return len(s)

    # --- Added Methods for Robustness ---
    def flush(self):
        """Pass the flush command to the original stdout."""
        self.original_stdout.flush()

    def isatty(self):
        """Pretend to be an interactive terminal if the original stdout is."""
        return self.original_stdout.isatty()

    def fileno(self):
        """Return the file descriptor number of the original stdout."""
        return self.original_stdout.fileno()

    @property
    def encoding(self):
        """Return the encoding of the original stdout."""
        return self.original_stdout.encoding


# Context manager to safely redirect stdout
class ProgressRedirector:
    def __init__(self, status_queue, file_info):
        self.status_queue = status_queue
        self.file_info = file_info
        self.capturer = None

    def __enter__(self):
        # Create the capturer instance and redirect stdout
        self.capturer = TqdmProgressCapturer(self.status_queue, self.file_info)
        sys.stdout = self.capturer
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # IMPORTANT: Always restore the original stdout
        if self.capturer:
            sys.stdout = self.capturer.original_stdout


# ---------------- REMOTE LLM (API-BASED) SETUP ----------------
class RemoteLLM:
    """
    Wrapper for a remote LLM API endpoint.
    """

    def __init__(self, api_key, api_url=DEFAULT_LLM_API_URL,
                 model="gpt-3.5-turbo"):  # TODO: make model settable
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
            model_name=DEFAULT_LOCAL_LLM_MODEL_NAME,
            status_queue=None,
            device=None,
            quantize_4bit=True,
            trust_remote_code=False,  # TODO: might need to be enabled for certain models, make user interface button?
    ):

        APP_STATE["local_model_ready"] = False # default value of shared boolean
        print("INFO: Set local_model_ready state to False.")

        self.model_name = model_name
        self.status_queue = status_queue
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"INFO: Initializing LocalLLM '{model_name}' on device: {self.device}")

        try:

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True,
                trust_remote_code=trust_remote_code
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

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
                    model_kwargs["torch_dtype"] = torch.float16
            else:
                print("WARNING: CPU device detected. Quantization is disabled.")

            print(f"INFO: Calling from_pretrained for '{model_name}'... This is the slow step.")

            if self.is_model_cached():
                print(f"INFO: Loading '{model_name}' from cache.")
                self.status_queue.put({
                    'type': 'status',
                    'payload': {'status': 'model_loading_from_cache', 'message': f'Loading {self.model_name} from cache.'}
                })
            else:
                print(f"INFO: Downloading '{model_name}'. This might take a while.")
                self.status_queue.put({
                    'type': 'status',
                    'payload': {'status': 'model_downloading', 'message': f'Downloading {self.model_name}. This might '
                                                                          f'take a while.'}
                })
                self.download_with_progress()

            # TODO: model_kwargs is passed to the model and for some reason messes with openai/gpt-oss-20b and openai/gpt-oss-120b
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )

            print("INFO: Model loading complete.")

            # Backend communication status
            self.status_queue.put({
                'type': 'local_llm_model_ready',
                'payload': True  # Signal that the model is ready
            })
            # Use the queue again for the final status update
            self.status_queue.put({
                'type': 'status',
                'payload': {'status': 'model_ready', 'message': f'{self.model_name} ready.'}
            })

        except Exception as e:
            print(f"ERROR: Failed to load local model '{model_name}'. Error: {e}")
            # Backend communication status
            self.status_queue.put({
                'type': 'local_llm_model_ready',
                'payload': False  # Signal error for backend
            })
            self.status_queue.put({
                'type': 'status',
                'payload': {'status': 'model_error', 'message': f'Error loading {self.model_name}.'}
            })

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

    def generate_narrative(self, prompt, stopping_criteria=None, **gen_overrides):
        """
        :param prompt:
        :param stopping_criteria: A list of StoppingCriteria to halt generation.
        :param gen_overrides:
        :return:
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        gen_cfg = {**self.generation_defaults, **gen_overrides}

        # Add stopping criteria to the generation config if provided
        if stopping_criteria:
            gen_cfg["stopping_criteria"] = stopping_criteria

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_cfg)

        gen_tokens = output_ids[0, inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        print("INFO: Narrative generated successfully.")
        # print("DEBUG: Local LLM's answer: " + text)
        return text.strip()

    def is_model_cached(self):
        try:
            # Try resolving config.json locally
            hf_hub_download(
                repo_id=self.model_name,
                filename="config.json",
                local_files_only=True
            )
            return True
        except Exception:
            return False

    # TODO: not used
    def custom_progress_callback(self, percent):
        print(f"Downloaded {percent}%")
        self.status_queue.put({
            'type': 'status',
            'payload': {
                'status': 'model_downloading',
                'message': f'{self.model_name}: {percent}% downloaded.'
            }
        })

    def download_with_progress(self):
        """
        Manually downloads all files, capturing console output to report progress
        for older versions of the huggingface_hub library.
        """
        try:
            api = HfApi()
            repo_tree = api.list_repo_tree(repo_id=self.model_name)
            repo_files = [item for item in repo_tree if isinstance(item, RepoFile)]

            # Can't easily do overall progress with this method, so we focus on per-file
            print(f"INFO: Starting download of {len(repo_files)} files.")
            self.status_queue.put({
            'type': 'status',
            'payload': {
                'status': 'model_downloading',
                'message': f'Starting download of {len(repo_files)} files.'
            }
            })

            # Loop through each file and download it inside our progress redirector
            for file_info in repo_files:
                print(f"INFO: Downloading file: {file_info.path} ({file_info.size / 1e6:.2f} MB)")
                self.status_queue.put({
                    'type': 'status',
                    'payload': {
                        'status': 'model_downloading',
                        'message': f'Downloading file: {file_info.path} ({file_info.size / 1e6:.2f} MB)'
                    }
                })

                # Use the context manager to capture progress for this specific download
                with ProgressRedirector(self.status_queue, file_info):
                    hf_hub_download(
                        repo_id=self.model_name,
                        filename=file_info.path,
                        resume_download=True,
                        # No tqdm_class argument here
                    )
                # After the 'with' block, stdout is automatically restored to normal
                print(f"\nINFO: Finished downloading {file_info.path}")
                self.status_queue.put({
                    'type': 'status',
                    'payload': {
                        'status': 'model_downloading',
                        'message': f'Finished downloading {file_info.path}.'
                    }
                })

        except Exception as e:
            # Ensure stdout is restored even if an error occurs somewhere else
            # This is a fallback, the context manager should handle it
            if isinstance(sys.stdout, TqdmProgressCapturer):
                sys.stdout = sys.stdout.original_stdout

            print(f"ERROR during model download: {e}")
            self.status_queue.put({
                'type': 'status',
                'payload': {
                    'status': 'error',
                    'message': f'Download failed: {str(e)}'
                }
            })


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
        # print("DEBUG: Prompt for LLM:\n\t" + prompt)
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

        summary_payload[p_id] = {
            'id': p_id,
            'name': person_name,
            'narrative_summary': narrative,
            'distribution': distribution,
            'total_detections': total,
            'average_engagement': avg_engagement
        }
    return summary_payload
