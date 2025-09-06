import os
import sys
import json
import base64
import numpy as np
from flask import render_template
from flask_socketio import emit
import cv2
import torch
import threading
import multiprocessing as mp

# --- Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# --- Local Imports ---
from model.model import EmotionCNN
from server.analysis import FaceReIDTracker, RemoteLLM, analyze_frame, llm_process_worker, generate_summary_payload, \
    LocalLLM
# At the top
import server.extensions as extensions
from server.extensions import socketio, app, APP_STATE, DEFAULT_LLM_API_URL, DEFAULT_LLM_API_MODEL, DEFAULT_LOCAL_LLM_MODEL_NAME



# --- Global State for the Worker Thread ---
frame_lock = threading.Lock()
latest_frame_data = None

# --- Global State for LLM Process Management ---
llm_process = None
task_queue = mp.Queue()
result_queue = mp.Queue()
status_queue = mp.Queue()
llm_process_lock = threading.RLock() # To protect access to the llm_process object, RLock to allow nested calls for restart_and_summarize

# --- State for pending summary after model reload ---
pending_summary_task = None
pending_summary_lock = threading.Lock() # Protects access to the pending_summary_task variable

# --- Global LLM instances ---
local_llm = None
remote_llm = None
llm_lock = threading.Lock()

# --- Global State for Cancellable Summary Generation ---
summary_thread = None
cancel_summary_flag = threading.Event()
summary_lock = threading.Lock()

# --- State for background model loading ---
model_loader_thread = None
is_model_loading = threading.Event()

# --- Global Settings Configuration ---
CURRENT_LLM_API_URL = DEFAULT_LLM_API_URL
CURRENT_LLM_API_MODEL = DEFAULT_LLM_API_MODEL
CURRENT_LOCAL_LLM_MODEL_NAME = DEFAULT_LOCAL_LLM_MODEL_NAME


def create_app():
    """
    Creates and configures the Flask application and SocketIO server.

    Sets up:
      - Flask app and SocketIO integration
      - Emotion model loading
      - Person tracker initialization
      - Background analysis worker thread
      - SocketIO event handlers for status updates, frame processing,
        model settings, and summary generation

    :return: A tuple (socketio, app) with the configured SocketIO server and Flask application.
    """

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


    def result_monitor():
        """Monitors the result queue and emits results back to the client via SocketIO."""
        print("INFO: Result queue monitor thread started.")
        while True:
            try:
                result = result_queue.get()

                person_id = result.get('person_id')
                summary_text = result.get('summary')
                error = result.get('error')

                if error:
                    print(f"ERROR from LLM worker: {error}", file=sys.stderr)
                    socketio.emit('summary_status',
                                  {'status': 'error', 'message': f'LLM Worker Error: An error occurred.'})
                    continue

                if person_id is not None and person_id in tracking_data:
                    # Look up the person's metadata from the tracker:
                    person_metadata = next((meta for meta in tracker.known_face_metadata if meta['id'] == person_id),
                                           None)
                    person_name = person_metadata['name'] if person_metadata else f"Person {person_id}"

                    # Retrieve the raw data collected for this person:
                    person_data = tracking_data[person_id]
                    emotions = person_data.get('emotions', [])
                    engagements = person_data.get('engagement', [])

                    # Calculate the required aggregated values:

                    # Total detections is the count of recorded emotions.
                    total_detections = len(emotions)

                    # Average engagement is the sum of engagement scores divided by the count.
                    avg_engagement = round(sum(engagements) / len(engagements), 2) if engagements else 0.0

                    # Emotion distribution is the frequency of each emotion.
                    # Assumes 7 emotion categories, indexed 0 through 6.
                    distribution_counts = [0] * 7
                    for emotion_index in emotions:
                        if 0 <= emotion_index < 7:
                            distribution_counts[emotion_index] += 1

                    # Normalize the counts to get a distribution between 0.0 and 1.0.
                    distribution = [round(count / total_detections, 4) for count in
                                    distribution_counts] if total_detections > 0 else [0.0] * 7

                    # Assemble the final payload in the desired format:
                    final_payload = {
                        person_id: {
                            'id': person_id,
                            'name': person_name,
                            'narrative_summary': summary_text,
                            'distribution': distribution,
                            'total_detections': total_detections,
                            'average_engagement': avg_engagement
                        }
                    }

                    # print("Emitted json.dumps for local LLM: " + str(final_payload))
                    socketio.emit('tracking_summary', json.dumps(final_payload))

                    # Use the person_name variable for the success message.
                    socketio.emit('summary_status', {
                        'status': 'success',
                        'message': f'Summary for {person_name} generated.'
                    })

            except Exception as e:
                print(f"Error in result monitor thread: {e}", file=sys.stderr)
            socketio.sleep(0.1)

    # --- Worker Thread for Frame Processing ---
    def analysis_worker():
        """
        Runs in a background thread, continuously processing the latest available frame.
        :return:
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

    # Helper function to gather and emit the current status of all settings
    def emit_status_update():
        """
        Gathers all current statuses and emits them to the client.
        :return:
        """
        status = {
            "api_key_present": extensions.LLM_API_KEY is not None and extensions.LLM_API_KEY != "",
            "api_url_present": CURRENT_LLM_API_URL is not None and CURRENT_LLM_API_URL != "",
            "local_model_present": CURRENT_LOCAL_LLM_MODEL_NAME is not None and CURRENT_LOCAL_LLM_MODEL_NAME != "",
            "cuda_available": torch.cuda.is_available(),
            "api_url": CURRENT_LLM_API_URL,
            "api_model_name": CURRENT_LLM_API_MODEL,
            "local_model_name": CURRENT_LOCAL_LLM_MODEL_NAME,
            "default_api_url": DEFAULT_LLM_API_URL,
            "default_local_model_name": DEFAULT_LOCAL_LLM_MODEL_NAME,
            "local_model_ready": APP_STATE.get("local_model_ready")
        }
        socketio.emit('status_update', status)

    def _invalidate_remote_llm():
        """Helper function to safely invalidate the remote LLM instance."""
        global remote_llm
        with llm_lock:
            remote_llm = None
        print("INFO: Remote LLM instance invalidated due to configuration change.")


    @socketio.on('connect')
    def on_connect():
        if not hasattr(app, 'analysis_thread_started'):
            socketio.start_background_task(target=analysis_worker)
            app.analysis_thread_started = True
            print("INFO: Client connected, background worker started.")

        emit_status_update()

    # Handler for the client to request a status update at any time
    @socketio.on('get_status')
    def on_get_status():
        emit_status_update()

    # --- Settings Management Handlers ---
    @socketio.on('set_api_key')
    def on_set_api_key(data):
        """Client is setting a new API key."""
        new_key = data.get('key')

        if new_key and new_key.strip():
            extensions.LLM_API_KEY = new_key
            print("INFO: API Key has been set by the user.")
        else:
            extensions.LLM_API_KEY = None
            print("INFO: API Key has been cleared.")

        _invalidate_remote_llm()
        emit_status_update()  # Send a full status update after changing the key

    # Handler for setting the API URL
    @socketio.on('set_api_url')
    def on_set_api_url(data):

        global CURRENT_LLM_API_URL, remote_llm
        new_url = data.get('url', '').strip()

        if new_url:
            CURRENT_LLM_API_URL = new_url
            print(f"INFO: API URL set to: {CURRENT_LLM_API_URL}")
        else:
            CURRENT_LLM_API_URL = DEFAULT_LLM_API_URL
            print(f"INFO: API URL cleared. Reverting to default: {CURRENT_LLM_API_URL}")

        _invalidate_remote_llm() # Invalidate to force recreation
        emit_status_update()

    # Handler for setting the API Model
    @socketio.on('set_api_model')
    def on_set_api_model(data):

        global CURRENT_LLM_API_MODEL, remote_llm
        new_model = data.get('api_model', '').strip()

        if new_model:
            CURRENT_LLM_API_MODEL = new_model
            print(f"INFO: API Model set to: {CURRENT_LLM_API_MODEL}")
        else:
            CURRENT_LLM_API_MODEL = DEFAULT_LLM_API_MODEL
            print(f"INFO: API Model reset. Reverting to default: {CURRENT_LLM_API_MODEL}")

        _invalidate_remote_llm()
        emit_status_update()

    @socketio.on('set_local_model')
    def on_set_local_model(data):
        global llm_process, CURRENT_LOCAL_LLM_MODEL_NAME

        with llm_process_lock:
            new_model_name = data.get('model_name', '').strip()
            if not new_model_name:
                new_model_name = DEFAULT_LOCAL_LLM_MODEL_NAME # This currently prevents clearing the input

            # Terminate the old process if it exists
            if llm_process and llm_process.is_alive():
                print(f"INFO: Terminating old LLM worker process (PID: {llm_process.pid}).")
                llm_process.terminate()
                llm_process.join(timeout=5)  # Wait a bit for it to clean up
                if llm_process.is_alive():
                    print(f"WARNING: Process {llm_process.pid} did not terminate gracefully. Killing.")
                    llm_process.kill()  # Force kill if terminate fails

                # Drain the queues to prevent old tasks/results from interfering
                while not task_queue.empty(): task_queue.get_nowait()
                while not result_queue.empty(): result_queue.get_nowait()

            CURRENT_LOCAL_LLM_MODEL_NAME = new_model_name
            print(f"INFO: Starting new LLM worker process for model: {new_model_name}")
            socketio.emit('summary_status',
                 {'status': 'initializing', 'message': f'Starting new LLM worker process for model: {new_model_name}.'})

            # Create and start the new worker process
            llm_process = mp.Process(
                target=llm_process_worker,
                args=(task_queue, result_queue, status_queue, new_model_name)
            )
            llm_process.start()

        socketio.start_background_task(target=queue_listener, queue=status_queue)
        socketio.start_background_task(target=result_monitor)
        emit_status_update()

    # Listener that runs in the main process using IPC
    def queue_listener(queue):
        """
        Listens to the result queue and emits socketio messages.
        This runs in a background THREAD in the main process.
        """
        global pending_summary_task

        print("INFO: Queue listener started.")
        while True:
            try:
                message = queue.get()  # Blocks until a message is available
                message_type = message.get('type')
                # Status update
                if message_type == 'status':
                    print(f"INFO: Received status from worker via IPC: {message['payload']}")
                    socketio.emit('summary_status', message['payload'])
                # Backend local llm model ready update
                elif message_type == 'local_llm_model_ready':
                    is_ready = message.get('payload', False)
                    print(f"INFO: Backend communication received model readiness update from worker. Model ready: {is_ready}")
                    APP_STATE["local_model_ready"] = is_ready
                    emit_status_update()
                    # Model is ready after pending summary tasks
                    if is_ready:
                        with pending_summary_lock:
                            if pending_summary_task is not None:
                                print("INFO: Model is ready after restart, executing pending summary task.")
                                # Call the summary function with the stored data
                                _execute_local_summary()
                                # Clear the task so it doesn't run again
                                pending_summary_task = None
                # Percentage update for progress bar
                elif message_type == 'model_downloading':
                    socketio.emit('model_downloading', message['payload'])
            except Exception as e:
                print(f"ERROR in queue_listener: {e}")

    def _execute_local_summary():
        """
        Contains the core logic for generating a summary with a local LLM.
        """
        with llm_process_lock:
            if not llm_process or not llm_process.is_alive():
                socketio.emit('summary_status',
                              {'status': 'error', 'message': 'Local LLM process is not running. Please set a model.'})
                return

        tasks_sent = 0
        for person_id, p_data in tracking_data.items():
            if len(p_data.get('emotions', [])) >= 5:
                person_metadata = next((meta for meta in tracker.known_face_metadata if meta['id'] == person_id), None)
                if not person_metadata:
                    print(f"WARNING: Could not find name for person_id {person_id}. Skipping summary.")
                    continue

                person_name = person_metadata['name']
                task = {
                    "person_name": person_name,
                    "emotions_sequence": p_data['emotions'],
                    "emotion_labels": emotion_labels,
                    "engagement_sequence": p_data.get('engagement'),
                    "person_id": person_id
                }
                task_queue.put(task)
                tasks_sent += 1

        if tasks_sent > 0:
            print(f"INFO: Sent {tasks_sent} summary task(s) to the LLM worker process.")
            socketio.emit('setting_validated', {'type': 'local_model', 'value': CURRENT_LOCAL_LLM_MODEL_NAME})
            socketio.emit('summary_status',
                          {'status': 'generating', 'message': f'Sent {tasks_sent} task(s) to LLM worker.'})
        else:
            socketio.emit('summary_status', {'status': 'error', 'message': 'No one had enough data for a summary.'})

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
        # The frontend sends a list of sources, a single target, and the final name
        source_ids = [int(sid) for sid in data['source_ids']]
        target_id = int(data['target_id'])
        new_name = data['name']

        # The first ID in the merged list is the target, others are sources
        if tracker.merge_persons(source_ids, target_id, new_name, tracking_data):
            # Notify clients about the merge so they can update their local state
            emit('merge_notification', {'source_ids': source_ids, 'target_id': target_id}, broadcast=True)
            # Send the final, updated list of all known faces
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
        """
        Handles the client's request to generate a summary.
        Delegates the task based on the selected mode.
        """
        use_llm_mode = int(data.get("use_llm", 0))

        # --- Mode 1: Local LLM via Multiprocessing ---
        if use_llm_mode == 1:
            _execute_local_summary()

        # --- Modes 0 & 2: Heuristic and Remote API via Threading ---
        else:
            def threaded_summary_worker():
                try:
                    active_llm = None
                    if use_llm_mode == 2:
                        global remote_llm
                        with llm_lock:
                            if not extensions.LLM_API_KEY:
                                raise ValueError("API Key is not set.")
                            if remote_llm is None:
                                socketio.emit('summary_status',
                                              {'status': 'calling_api', 'message': 'Initializing remote LLM client...'})
                                remote_llm = RemoteLLM(api_key=extensions.LLM_API_KEY, api_url=CURRENT_LLM_API_URL, model=CURRENT_LLM_API_MODEL)
                                print("INFO: Remote LLM client is ready.")
                                socketio.emit('setting_validated',
                                     {'type': 'api_model', 'value': CURRENT_LLM_API_MODEL})
                                socketio.emit('setting_validated',
                                     {'type': 'api_url', 'value': CURRENT_LLM_API_URL})
                        active_llm = remote_llm

                    socketio.emit('summary_status', {'status': 'generating', 'message': 'Generating summary...'})

                    summary_payload = generate_summary_payload(
                        tracking_data, tracker, emotion_labels, llm=active_llm
                    )

                    socketio.emit('tracking_summary', json.dumps(summary_payload))

                    success_message = "Heuristic summary generated successfully."
                    if use_llm_mode == 2:
                        success_message = "Summary generated from remote API successfully."

                    socketio.emit('summary_status', {'status': 'success', 'message': success_message})
                    print("INFO: Heuristic/Remote summary generated and sent.")

                except Exception as e:
                    print(f"ERROR: Error in threaded summary worker: {e}", file=sys.stderr)
                    socketio.emit('summary_status', {'status': 'error', 'message': f'An error occurred: {str(e)}'})

            socketio.start_background_task(target=threaded_summary_worker)

    @socketio.on('cancel_summary')
    def on_cancel_summary():
        with llm_process_lock:
            if llm_process and llm_process.is_alive():
                print(f"INFO: User requested cancellation. Terminating LLM worker process (PID: {llm_process.pid}).")
                llm_process.terminate()
                llm_process.join(timeout=2)
                emit('summary_status', {'status': 'cancelled', 'message': 'LLM process has been terminated.'})
            else:
                print("INFO: User requested cancellation, but no LLM process was running.")

    @socketio.on('restart_and_summarize')
    def on_restart_and_summarize(data):
        """
        Handles a request to generate a summary, reloading the local model only if necessary.
        """
        global pending_summary_task
        use_llm_mode = int(data.get("use_llm", 0))

        # Proceed to the summary directly if heuristic or remote API summary is selected
        if use_llm_mode != 1:
            print("INFO: Received summarize request for non-local LLM. Forwarding.")
            on_get_summary(data)
            return

        # Local LLM Logic
        with llm_process_lock:
            # Check if a process is already running and the model has confirmed it's ready
            if llm_process and llm_process.is_alive() and APP_STATE.get("local_model_ready"):
                print("INFO: LLM process is already ready. Reusing for new summary.")
                on_get_summary(data)
                return

        # Reload logic
        print("INFO: LLM process not ready. Starting reload and queuing summary task.")
        with pending_summary_lock:
            pending_summary_task = {"use_llm": use_llm_mode}

        on_set_local_model({'model_name': data.get('model_name', DEFAULT_LOCAL_LLM_MODEL_NAME)})


    # --- Start the result monitor thread when the app starts up ---
    if not hasattr(app, 'result_monitor_started'):
        socketio.start_background_task(target=result_monitor)
        app.result_monitor_started = True

    return socketio, app


if __name__ == '__main__':
    # --- IMPORTANT: Set start method for multiprocessing ---
    # Must be 'spawn' for CUDA compatibility and placed inside the main guard
    mp.set_start_method('spawn', force=True)

    socketio, app = create_app()
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)