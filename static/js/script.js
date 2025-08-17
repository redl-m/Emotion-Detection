document.addEventListener('DOMContentLoaded', () => {
    // ---------------------------------------------------
    // A. Constants & State
    // ---------------------------------------------------
    const EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'];
    const socket = io();
    const D3_COLORS = d3.scaleOrdinal(d3.schemeCategory10);
    const FRAME_SEND_INTERVAL_MS = 200; // ~5 FPS
    const MAX_TIMESERIES_POINTS = 100;
    const poseEstimator = new HeadPoseEstimator();
    let useLLM;

    let state = {
        isTracking: false,
        videoSource: null,
        sendIntervalId: null,
        lastFrameResults: [],
        participants: {}, // { id: { color, history: [...] }}
        participantNames: {}, // { id: "Name" }, synced with server
        selectedParticipantId: 'average',
        isMergeMode: false, // Controls the modal visibility
        mergeSelection: [], // Stores selected IDs for merging. First element is the target.
    };

    // ---------------------------------------------------
    // B. DOM Element Selectors
    // ---------------------------------------------------
    const dom = {
        videoWrapper: document.getElementById('video-wrapper'),
        overlayCanvas: document.getElementById('overlayCanvas'),
        overlayCtx: document.getElementById('overlayCanvas').getContext('2d'),
        placeholder: document.getElementById('placeholder'),
        placeholderContent: document.getElementById('placeholder-content'),
        loader: document.getElementById('loader'),
        uploadInput: document.getElementById('videoUpload'),
        liveBtn: document.getElementById('liveBtn'),
        trackBtn: document.getElementById('trackBtn'),
        trackBtnText: document.getElementById('trackBtnText'),
        mergeBtn: document.getElementById('mergeBtn'),
        mergeModal: document.getElementById('mergeModal'),
        closeMergeModalBtn: document.getElementById('closeMergeModalBtn'),
        cancelMergeBtn: document.getElementById('cancelMergeBtn'),
        confirmMergeBtn: document.getElementById('confirmMergeBtn'),
        mergeParticipantList: document.getElementById('mergeParticipantList'),
        mergeNamePrompt: document.getElementById('mergeNamePrompt'),
        newPersonNameInput: document.getElementById('newPersonName'),
        controlsBottom: document.querySelector('.controls-bottom'),
        analysisSection: document.getElementById('analysis-section'),
        participantSelectors: {
            bar: document.getElementById('participant-selector-bar'),
            ts: document.getElementById('participant-selector-ts'),
            attn: document.getElementById('participant-selector-attn')
        },
        summaryContainer: document.getElementById('summary-container'),
        summaryReport: document.getElementById('summary-report'),
    };

    const apiKeyStatusText = document.getElementById('apiKeyStatusText');
    const apiKeyStatusIndicator = document.getElementById('apiKeyStatusIndicator');
    const apiUrlStatusText = document.getElementById('apiUrlStatusText');
    const apiUrlStatusIndicator = document.getElementById('apiUrlStatusIndicator');
    const apiGroupStatus = document.getElementById('apiGroupStatus');

    const localLlmStatusText = document.getElementById('localLlmStatusText');
    const localLlmStatusIndicator = document.getElementById('localLlmStatusIndicator');
    const cudaStatusText = document.getElementById('cudaStatusText');
    const cudaStatusIndicator = document.getElementById('cudaStatusIndicator');
    const localLlmGroupStatus = document.getElementById('localLlmGroupStatus');

    const apiKeyInput = document.getElementById('apiKeyInput');
    const setApiKeyBtn = document.getElementById('setApiKeyBtn');
    const apiUrlInput = document.getElementById('apiUrlInput');
    const setApiUrlBtn = document.getElementById('setApiUrlBtn');
    const localModelInput = document.getElementById('localModelInput');
    const setLocalModelBtn = document.getElementById('setLocalModelBtn');

    // ---------------------------------------------------
    // C. Setup & Initialization
    // ---------------------------------------------------

    let chartManager;
    let attentionChart;


    /**
     * Initializes chart managers,  adds event listeners and connects to the server.
     */
    function init() {

        chartManager = new ChartManager({
            barChartSelector: '#prob-bar-chart',
            timeSeriesSelector: '#chart',
            legendSelector: '#legend',
            emotions: EMOTIONS,
            colors: D3_COLORS
        });
        chartManager.init();

        attentionChart = new AttentionChart({
            chartSelector: '#attention-chart-container',
        });
        attentionChart.init();

        addEventListeners();
        poseEstimator.initialize();
        socket.emit('client_ready');
        requestAnimationFrame(drawLoop);

        if (typeof socket === 'undefined') {
            console.error('Socket.IO instance not found. Make sure settings.js is loaded after the main socket script.');
        }
    }

    socket.on("connect", () => {
        console.log("Connected to server");
    });


    /**
     * Adds event listeners to html elements and socket.
     */
    function addEventListeners() {

        dom.uploadInput.addEventListener('change', handleFileUpload);
        dom.liveBtn.addEventListener('click', handleLiveFeed);
        dom.trackBtn.addEventListener('click', toggleTracking);
        dom.placeholder.addEventListener('click', () => dom.uploadInput.click());
        dom.mergeBtn.addEventListener('click', openMergeModal);
        dom.closeMergeModalBtn.addEventListener('click', closeMergeModal);
        dom.cancelMergeBtn.addEventListener('click', closeMergeModal);
        dom.confirmMergeBtn.addEventListener('click', handleConfirmMerge);

        setApiKeyBtn.addEventListener('click', () => {
            const key = apiKeyInput.value;
            // The backend handles empty strings as "clearing" the key
            socket.emit('set_api_key', {key: key});
            apiKeyInput.value = ''; // Clear input for security
            apiKeyInput.placeholder = "API Key has been set";
            setTimeout(() => {
                apiKeyInput.placeholder = "Enter new API Key";
            }, 2000);
        });

        setApiUrlBtn.addEventListener('click', () => {
            const url = apiUrlInput.value.trim();
            if (url) {
                socket.emit('set_api_url', {url: url});
            } else {
                alert('API URL field cannot be empty.');
            }
        });

        setLocalModelBtn.addEventListener('click', () => {
            const model = localModelInput.value.trim();
            if (model) {
                socket.emit('set_local_model', {model_name: model});
            } else {
                alert('Local LLM Model field cannot be empty.');
            }
        });

        // Socket listeners
        socket.on('known_faces_update', onKnownFacesUpdate);
        socket.on('frame_data', onFrameData);
        socket.on('tracking_summary', renderSummary);
        socket.on('merge_notification', onMergeNotification);
        // Listener for the unified status update from the backend
        socket.on('status_update', (data) => {
            console.log('Received status update:', data);
            updateApiStatus(data.api_key_present, data.api_url_present);
            updateLocalLlmStatus(data.local_model_present, data.cuda_available);
            // Populate the input fields with current values from the server
            if (data.api_url) apiUrlInput.value = data.api_url;
            if (data.local_model_name) localModelInput.value = data.local_model_name;
        });
    }

    // ---------------------------------------------------
    // D. Core App Logic (Video & Tracking)
    // ---------------------------------------------------

    /**
     * Toggles between showing the placeholder or content.
     * @param isShowing True, if placeholder should be invisible and content shown, false otherwise.
     */
    function showLoader(isShowing) {

        dom.loader.classList.toggle('hidden', !isShowing);
        dom.placeholderContent.classList.toggle('hidden', isShowing);
    }


    /**
     * Resets the app by clearing all object sates, setting buttons to defaults
     * and clearing all charts.
     */
    function resetApp() {

        if (state.isTracking) toggleTracking();
        if (state.videoSource) {
            if (state.videoSource.srcObject) {
                state.videoSource.srcObject.getTracks().forEach(track => track.stop());
            }
            state.videoSource.remove();
        }

        Object.assign(state, {
            isTracking: false,
            videoSource: null,
            sendIntervalId: null,
            lastFrameResults: [],
            isMergeMode: false,
            mergeSelection: []
        });

        dom.placeholder.classList.replace('placeholder-hidden', 'placeholder-active');
        showLoader(false);

        if (dom.trackBtn) dom.trackBtn.disabled = true;
        if (dom.mergeBtn) {
            dom.mergeBtn.disabled = true;
            dom.mergeBtn.classList.remove('active');
        }

        dom.controlsBottom.querySelector('#video-controls')?.remove();
        dom.overlayCtx.clearRect(0, 0, dom.overlayCanvas.width, dom.overlayCanvas.height);
        dom.summaryContainer.classList.add('hidden');
        dom.summaryReport.innerHTML = '';
        renderParticipantSelectors();
        if (chartManager) chartManager.clear();
        if (attentionChart) attentionChart.clear();
    }


    /**
     * Initializes and starts a live video feed from the user's camera.
     * @returns {Promise<void>} Resolves when the video starts playing.
     * @throws {Error} If the camera cannot be accessed or permissions are denied.
     */
    async function handleLiveFeed() {

        try {
            resetApp();
            const stream = await navigator.mediaDevices.getUserMedia({video: true});
            const liveVideoEl = document.createElement('video');
            liveVideoEl.srcObject = stream;
            liveVideoEl.autoplay = true;
            liveVideoEl.muted = true;
            liveVideoEl.playsInline = true;
            setupVideoSource(liveVideoEl, true);
            await liveVideoEl.play();
        } catch (err) {
            console.error("Error accessing camera:", err);
            alert("Could not access the camera. Please check permissions.");
            resetApp();
        }
    }


    /**
     * Handles file uploads for video analysis.
     * @param event The upload event to handle.
     */
    function handleFileUpload(event) {

        const file = event.target.files[0];
        if (file) {
            resetApp();
            showLoader(true);
            const url = URL.createObjectURL(file);
            const videoEl = document.createElement('video');
            videoEl.src = url;
            videoEl.playsInline = true;
            videoEl.volume = 0.5;
            setupVideoSource(videoEl, false);
            event.target.value = '';
        }
    }


    /**
     * Sets up the video source based on the given video element.
     * @param videoElement The video element to analyze.
     * @param isLive True, if live analysis is used, false otherwise.
     */
    function setupVideoSource(videoElement, isLive) {

        state.videoSource = videoElement;
        dom.videoWrapper.prepend(videoElement);
        dom.placeholder.classList.replace('placeholder-active', 'placeholder-hidden');
        dom.trackBtn.disabled = false;
        dom.mergeBtn.disabled = false;
        showLoader(false);

        const resizeCanvas = () => {
            dom.overlayCanvas.width = videoElement.videoWidth;
            dom.overlayCanvas.height = videoElement.videoHeight;
        };
        if (isLive) {
            videoElement.addEventListener('playing', resizeCanvas, {once: true});
        } else {
            videoElement.addEventListener('loadedmetadata', resizeCanvas, {once: true});
            createVideoControls();
        }
    }


    /**
     * Toggles tracking state of the program and clears user history.
     */
    function toggleTracking() {

        state.isTracking = !state.isTracking;
        dom.trackBtn.classList.toggle('active', state.isTracking);
        dom.trackBtnText.textContent = state.isTracking ? 'Stop & Summarize' : 'Start Tracking';

        if (state.isTracking) {
            // Clear history for all existing participants but keep their names/IDs
            for (const id in state.participants) {
                state.participants[id].history = [];
                state.participants[id].attentionHistory = [];
            }
            dom.summaryContainer.classList.add('hidden');
            dom.summaryReport.innerHTML = '';
            socket.emit('start_tracking');
            state.sendIntervalId = setInterval(sendFrame, FRAME_SEND_INTERVAL_MS);
        } else {
            clearInterval(state.sendIntervalId);
            state.sendIntervalId = null;
            socket.emit("get_summary", {use_llm: useLLM});
        }
    }


    /**
     * Emits the current frame for the backend to process.
     * @returns {Promise<void>} A promise that resolves when the frame has been processed
     * and sent, without returning any value.
     */
    async function sendFrame() {

        if (!state.videoSource || state.videoSource.paused || state.videoSource.ended || state.videoSource.videoWidth === 0) return;

        // 1. Asynchronously process the frame for head pose
        await poseEstimator.processFrame(state.videoSource);

        // 2. Get the latest pose data from the estimator
        const headPoseData = poseEstimator.getLatestPoseData();

        // 3. Send frame and pose data to the server
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = state.videoSource.videoWidth;
        tempCanvas.height = state.videoSource.videoHeight;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(state.videoSource, 0, 0, tempCanvas.width, tempCanvas.height);

        socket.emit('frame', {
            data_url: tempCanvas.toDataURL('image/jpeg', 0.7),
            head_pose: headPoseData // Use the data from our module
        });
    }


    /**
     * Opens the merge faces modal window.
     */
    function openMergeModal() {

        state.isMergeMode = true;
        state.mergeSelection = [];
        dom.mergeModal.style.display = 'flex';
        dom.mergeBtn.classList.add('active');
        renderMergeModalContent();
    }


    /**
     * Closes the merge faces modal window and resets state.
     */
    function closeMergeModal() {

        state.isMergeMode = false;
        state.mergeSelection = [];
        dom.mergeModal.style.display = 'none';
        dom.mergeBtn.classList.remove('active');
        // Clean up just in case
        dom.mergeNamePrompt.style.display = 'none';
        dom.newPersonNameInput.value = '';
    }


    /**
     * Renders the content of the merge modal, showing all available participants.
     */
    function renderMergeModalContent() {

        const container = dom.mergeParticipantList;
        container.innerHTML = '';

        const participantIds = Object.keys(state.participants);

        participantIds.forEach(id => {
            const name = state.participantNames[id] || `P${id}`;
            const personDiv = document.createElement('div');
            personDiv.className = 'participant-item';
            personDiv.dataset.id = id;

            const colorSwatch = document.createElement('div');
            colorSwatch.className = 'color-swatch';
            colorSwatch.style.backgroundColor = state.participants[id].color;

            const nameSpan = document.createElement('span');
            nameSpan.textContent = name;

            personDiv.appendChild(colorSwatch);
            personDiv.appendChild(nameSpan);

            // Add selection classes
            if (state.mergeSelection.includes(id)) {
                personDiv.classList.add('selected');
                if (state.mergeSelection[0] === id) {
                    personDiv.classList.add('target'); // Special class for the target
                }
            }

            personDiv.addEventListener('click', () => handleMergeParticipantSelect(id));
            container.appendChild(personDiv);
        });

        // Update the confirm button state
        dom.confirmMergeBtn.disabled = state.mergeSelection.length < 2;
    }


    /**
     * Handles the selection of a participant inside the merge modal.
     * @param {string} id - The ID of the participant who was clicked.
     */
    function handleMergeParticipantSelect(id) {

        const index = state.mergeSelection.indexOf(id);
        if (index > -1) {
            state.mergeSelection.splice(index, 1); // Deselect
        } else {
            state.mergeSelection.push(id); // Select
        }
        // Re-render the modal to update styles
        renderMergeModalContent();
    }


    /**
     * Handles the final merge confirmation, including smart naming logic.
     */
    function handleConfirmMerge() {

        if (state.mergeSelection.length < 2) return;

        if (dom.mergeNamePrompt.style.display !== 'none') {
            const newName = dom.newPersonNameInput.value.trim();
            if (!newName) {
                alert('Please enter a name for the merged person.');
                return;
            }
            emitMergeEvent(newName);
            return;
        }

        const selectedNames = state.mergeSelection.map(id => getParticipantName(id));

        const customNames = selectedNames.filter(name => !/^P\d+$/.test(name));
        const uniqueCustomNames = new Set(customNames);

        let finalName = '';

        if (uniqueCustomNames.size === 0) {
            const targetId = state.mergeSelection[0];
            // The final name is the default name of the target.
            finalName = getParticipantName(targetId);
            emitMergeEvent(finalName);
        } else if (uniqueCustomNames.size === 1) {
            finalName = uniqueCustomNames.values().next().value;
            emitMergeEvent(finalName);
        } else {
            dom.mergeNamePrompt.style.display = 'block';
            dom.newPersonNameInput.focus();
        }
    }


    /**
     * Emits the merge event to the server after the final name has been determined.
     * @param {string} finalName - The name to be assigned to the merged person.
     */
    function emitMergeEvent(finalName) {

        const target_id = state.mergeSelection[0];
        const source_ids = state.mergeSelection.slice(1);

        socket.emit('merge_persons', {
            source_ids: source_ids,
            target_id: target_id,
            name: finalName
        });

        closeMergeModal();
    }


    /**
     * Gets the display name for a participant.
     * Returns the custom name if it exists, otherwise generates the default name (e.g., "P1").
     * @param {string} id The participant's ID.
     * @returns {string} The display name.
     */
    function getParticipantName(id) {
        // If a custom name is stored in our state, use it. Otherwise, generate the default.
        return state.participantNames[id] || `P${id}`;
    }


    // ---------------------------------------------------
    // E. Real-time Data Handling & Visualization
    // ---------------------------------------------------

    /**
     * Updates the list of known participants based on metadata from the server.
     * This function synchronizes the local state with the server state, adding new
     * participants and removing old ones (e.g., after a merge).
     * @param {Array<{id: number, name: string}>} metadata - Array of participant metadata objects from the server.
     */
    function onKnownFacesUpdate(metadata) {

        const newNames = {};
        const serverIds = new Set();

        metadata.forEach(p => {
            const id = String(p.id);
            serverIds.add(id);

            // Check if the name received from the server is just a default placeholder (e.g., "P0" for id 0).
            if (/^P\d+$/.test(p.name) && p.name === `P${id}`) {
                newNames[id] = undefined; // Do not store the default name as a "real" name.
            } else {
                newNames[id] = p.name; // Store the actual custom name.
            }

            if (!state.participants[id]) {
                state.participants[id] = {
                    color: D3_COLORS(id),
                    history: [],
                    attentionHistory: []
                };
            }
        });

        for (const localId in state.participants) {
            if (!serverIds.has(localId)) {
                delete state.participants[localId];
            }
        }

        state.participantNames = newNames;
        renderParticipantSelectors();

        if (state.selectedParticipantId !== 'average' && !state.participants[state.selectedParticipantId]) {
            state.selectedParticipantId = 'average';
            renderParticipantSelectors();
            updateActiveCharts();
        }
    }


    /**
     * Merges multiple participant records into one based on server notification.
     * @param {{source_ids: string[], target_id: string}} data - Object containing source and target IDs.
     */
    function onMergeNotification({source_ids, target_id}) {

        if (state.participants[target_id]) {
            source_ids.forEach(source_id => {
                if (state.participants[source_id]) {
                    // Merge history data
                    state.participants[target_id].history.push(...state.participants[source_id].history);

                    // Merge attention history if it exists
                    const sourceAttnHistory = state.participants[source_id].attentionHistory || [];
                    if (!state.participants[target_id].attentionHistory) {
                        state.participants[target_id].attentionHistory = [];
                    }
                    state.participants[target_id].attentionHistory.push(...sourceAttnHistory);

                    // Delete the old source participant
                    delete state.participants[source_id];
                }
            });

            // Sort the merged histories by timestamp to ensure correctness
            state.participants[target_id].history.sort((a, b) => a.timestamp - b.timestamp);
            if (state.participants[target_id].attentionHistory) {
                state.participants[target_id].attentionHistory.sort((a, b) => a.timestamp - b.timestamp);
            }

            // If a merged person was selected, switch to the target person's view
            if (source_ids.includes(state.selectedParticipantId)) {
                state.selectedParticipantId = target_id;
                updateActiveCharts();
            }
        }
        // Final state consistency will be ensured by the 'known_faces_update' event that follows.
    }


    /**
     * Processes incoming frame analysis data from the server.
     * Updates participant histories, creates new participants when detected
     * and refreshes charts and selectors.
     * @param {string} msg - JSON string containing frame analysis results from the backend.
     */
    function onFrameData(msg) {

        if (!state.isTracking) return;
        const results = JSON.parse(msg);
        state.lastFrameResults = results;
        const now = new Date();
        let newParticipantDetected = false;

        results.forEach(p => {
            if (!state.participants[p.id]) {
                newParticipantDetected = true;
                // Names are now handled by onKnownFacesUpdate, just create the participant object
                state.participants[p.id] = {
                    color: D3_COLORS(p.id),
                    history: [],
                    attentionHistory: []
                };
                // The name might not be known yet, server will send it via 'known_faces_update'
            }
            state.participants[p.id].history.push({timestamp: now, probs: p.probs});
            if (state.participants[p.id].history.length > MAX_TIMESERIES_POINTS) {
                state.participants[p.id].history.shift();
            }

            // Store attention history
            if (p.engagement !== undefined) {
                state.participants[p.id].attentionHistory.push({timestamp: now, engagement: p.engagement});
                if (state.participants[p.id].attentionHistory.length > MAX_TIMESERIES_POINTS) {
                    state.participants[p.id].attentionHistory.shift();
                }
            }
        });
        if (newParticipantDetected) {
            renderParticipantSelectors();
        }
        updateActiveCharts();
    }


    /**
     * Continuously renders bounding boxes, names, and emotion/confidence labels.
     */
    function drawLoop() {

        dom.overlayCtx.clearRect(0, 0, dom.overlayCanvas.width, dom.overlayCanvas.height);
        if (state.isTracking && state.lastFrameResults.length > 0) {
            state.lastFrameResults.forEach(p => {
                const [x, y, w, h] = p.bbox;
                const color = state.participants[p.id]?.color || '#FFFFFF';
                const name = state.participantNames[p.id] || `P${p.id}`;
                dom.overlayCtx.strokeStyle = color;
                dom.overlayCtx.lineWidth = 3;
                dom.overlayCtx.strokeRect(x, y, w, h);
                dom.overlayCtx.font = 'bold 16px ' + getComputedStyle(document.body).fontFamily;
                const text = `${name}: ${EMOTIONS[p.emotion]} (${Math.round(p.confidence * 100)}%)`;
                const textMetrics = dom.overlayCtx.measureText(text);
                dom.overlayCtx.fillStyle = 'rgba(0,0,0,0.6)';
                dom.overlayCtx.fillRect(x, y - 22, textMetrics.width + 10, 22);
                dom.overlayCtx.fillStyle = color;
                dom.overlayCtx.fillText(text, x + 5, y - 5);
            });
        }
        requestAnimationFrame(drawLoop);
    }

    // ---------------------------------------------------
    // F. Charting, Summary & Participant UI
    // ---------------------------------------------------

    /**
     * Renders the participant selector buttons under each chart header.
     * The merge-specific UI has been removed from this function.
     */
    function renderParticipantSelectors() {

        ['bar', 'ts', 'attn'].forEach(key => {
            const container = dom.participantSelectors[key];
            container.innerHTML = '';
            const selector = document.createElement('div');
            selector.className = 'participant-selector';

            const ids = ['average', ...Object.keys(state.participants)];

            ids.forEach(id => {
                const name = (id === 'average') ? 'Avg' : getParticipantName(id);
                const btn = document.createElement('button');
                btn.className = 'p-btn';
                btn.textContent = name;
                btn.dataset.id = id;

                if (state.selectedParticipantId === id) {
                    btn.classList.add('selected');
                }

                btn.addEventListener('click', (e) => handleParticipantClick(e, id, name, btn));
                selector.appendChild(btn);
            });
            container.appendChild(selector);
        });
    }


    /**
     * Sets up the summary method slider with dynamic options.
     * @param onChangeCallback Defines the behavior on switch.
     */
    function setupSummaryMethodSlider(onChangeCallback) {

        const options = document.querySelectorAll(".llm-slider-option");
        const indicator = document.querySelector(".llm-slider-indicator");

        if (!options.length || !indicator) {
            console.error("Slider elements not found!");
            return;
        }

        const numOptions = options.length;
        const optionWidthPercent = 100 / numOptions;

        indicator.style.width = `calc(${optionWidthPercent}% - 3px)`;

        options.forEach((opt, index) => {
            opt.addEventListener("click", () => {
                const mode = opt.getAttribute("data-mode");

                // Update visual state
                options.forEach(o => o.classList.remove("selected"));
                opt.classList.add("selected");
                indicator.style.left = `calc(${index * optionWidthPercent}% + 2px)`;


                // Call the callback function with the new mode
                if (typeof onChangeCallback === "function") {
                    onChangeCallback(parseInt(mode));
                }
            });
        });

        // Set initial state
        if (options[0]) {
            options[0].classList.add("selected");
            indicator.style.left = "2px";
        }
    }


    /**
     * Sets up the summary method slider.
     */
    setupSummaryMethodSlider(newMode => {
        console.log("Summary mode changed to:", newMode);
        useLLM = newMode;
    });


    /**
     * Handles click interactions on a participant selector button.
     *
     * - If "average" is clicked, selects the average view and updates charts.
     * - If merge mode is active, toggles participant selection for merging.
     * - On double-click, replaces the button with an input to rename the participant.
     * - On single-click, selects the participant and updates charts.
     *
     * @param {MouseEvent} event - The click event object.
     * @param {string} id - The participant ID ("average" for group average).
     * @param {string} name - The current participant name.
     * @param {HTMLElement} btnElement - The button element clicked.
     */
    function handleParticipantClick(event, id, name, btnElement) {

        if (state.isMergeMode) return;

        if (id === 'average') {
            state.selectedParticipantId = id;
            renderParticipantSelectors();
            updateActiveCharts();
            return;
        }

        if (event.detail === 2) { // Double-click to rename
            const input = document.createElement('input');
            input.type = 'text';
            input.value = name;
            input.className = 'p-btn-rename-input';
            btnElement.replaceWith(input);
            input.focus();
            input.select();

            const saveName = () => {
                const newName = input.value.trim() || `P${id}`;
                socket.emit('rename_person', {id: parseInt(id, 10), name: newName});
            };

            input.addEventListener('blur', saveName);
            input.addEventListener('keydown', e => {
                if (e.key === 'Enter') input.blur();
                if (e.key === 'Escape') {
                    input.replaceWith(btnElement);
                }
            });
        } else { // Single-click to select
            state.selectedParticipantId = id;
            renderParticipantSelectors();
            updateActiveCharts();
        }
    }


    /**
     * Renders the session summary view with charts and narrative descriptions
     * based on summary data received from the backend.
     *
     * @param {string} summaryMsg - JSON string containing summary statistics for each participant.
     */
    function renderSummary(summaryMsg) {

        const summaryData = JSON.parse(summaryMsg);
        if (Object.keys(summaryData).length === 0) return;

        dom.summaryContainer.classList.remove('hidden');
        dom.summaryReport.innerHTML = '';
        dom.analysisSection.scrollTo({top: dom.analysisSection.scrollHeight, behavior: 'smooth'});

        for (const id in summaryData) {
            const p_data = summaryData[id];
            const card = document.createElement('div');
            card.className = 'summary-card';
            card.innerHTML = `
                <div class="donut-chart" id="donut-${id}"></div>
                <div class="summary-details">
                    <h3>${p_data.name}</h3>
                    <div class="narrative-summary">${p_data.narrative_summary}</div>
                    <p class="frame-count">Detected in <strong>${p_data.total_detections}</strong> frames.</p>
                </div>`;
            dom.summaryReport.appendChild(card);
            ChartManager.createDonut(`#donut-${id}`, p_data.distribution, {
                emotions: EMOTIONS,
                colors: D3_COLORS
            });
        }
    }


    /**
     * Updates the emotion and attention charts based on the selected participant.
     * Supports both "average" view and individual participant view.
     * Calculates time-synced averages for group view and updates chart components.
     */
    function updateActiveCharts() {

        let emotionHistory = [];
        let attentionHistory = [];

        if (state.selectedParticipantId === 'average') {
            const allEmotionHistory = Object.values(state.participants).flatMap(p => p.history);
            if (allEmotionHistory.length > 0) {
                const timeMapEmotions = new Map();
                allEmotionHistory.forEach(d => {
                    const key = d.timestamp.getTime();
                    if (!timeMapEmotions.has(key)) timeMapEmotions.set(key, {
                        count: 0,
                        probs: new Array(EMOTIONS.length).fill(0)
                    });
                    const entry = timeMapEmotions.get(key);
                    entry.count++;
                    d.probs.forEach((p, i) => entry.probs[i] += p);
                });
                const avgEmotionHistory = [];
                timeMapEmotions.forEach((value, key) => {
                    avgEmotionHistory.push({timestamp: new Date(key), probs: value.probs.map(p => p / value.count)});
                });
                emotionHistory = avgEmotionHistory.sort((a, b) => a.timestamp - b.timestamp);
            }

            const allAttentionHistory = Object.values(state.participants).flatMap(p => p.attentionHistory || []);
            if (allAttentionHistory.length > 0) {
                const timeMapAttention = new Map();
                allAttentionHistory.forEach(d => {
                    // Guard against malformed data entries
                    if (!d || !d.timestamp) return;

                    const key = d.timestamp.getTime();
                    if (!timeMapAttention.has(key)) {
                        timeMapAttention.set(key, {count: 0, totalEngagement: 0});
                    }
                    const entry = timeMapAttention.get(key);
                    entry.count++;
                    entry.totalEngagement += d.engagement;
                });

                const avgAttentionHistory = [];
                timeMapAttention.forEach((value, key) => {
                    avgAttentionHistory.push({
                        timestamp: new Date(key),
                        engagement: value.totalEngagement / value.count
                    });
                });

                // assigns the calculated average to the variable.
                attentionHistory = avgAttentionHistory.sort((a, b) => a.timestamp - b.timestamp);
            }

        } else if (state.participants[state.selectedParticipantId]) {
            // This part was already correct. It fetches data for a single participant.
            emotionHistory = state.participants[state.selectedParticipantId].history;
            attentionHistory = state.participants[state.selectedParticipantId].attentionHistory || [];
        }

        // These calls will now receive the correct data for both 'average' and individual views.
        chartManager.update(emotionHistory, FRAME_SEND_INTERVAL_MS / 2);
        attentionChart.update(attentionHistory, FRAME_SEND_INTERVAL_MS / 2);
    }


    /**
     * Creates and attaches video playback controls for the current video source
     * and wires them to update based on video events.
     */
    function createVideoControls() {

        const controlsContainer = document.createElement('div');
        controlsContainer.id = 'video-controls';

        const playPauseBtn = document.createElement('button');
        playPauseBtn.id = 'playPauseBtn';
        playPauseBtn.className = 'video-control-btn';

        const seekBar = document.createElement('input');
        seekBar.type = 'range';
        seekBar.id = 'seekBar';
        seekBar.className = 'seek-bar';
        seekBar.value = 0;
        seekBar.step = 0.1;

        const timeDisplay = document.createElement('div');
        timeDisplay.id = 'timeDisplay';
        timeDisplay.className = 'time-display';

        controlsContainer.append(playPauseBtn, seekBar, timeDisplay);
        dom.controlsBottom.append(controlsContainer);

        const video = state.videoSource;
        playPauseBtn.addEventListener('click', () => video.paused ? video.play() : video.pause());
        seekBar.addEventListener('input', (e) => video.currentTime = (e.target.value / 100) * video.duration);
        ['timeupdate', 'play', 'pause', 'loadeddata', 'durationchange'].forEach(evt => {
            video.addEventListener(evt, () => updateVideoPlayerUI(video, playPauseBtn, seekBar, timeDisplay));
        });
        updateVideoPlayerUI(video, playPauseBtn, seekBar, timeDisplay);
    }


    /**
     * Updates the state of the custom video player controls to match the current
     * playback position and state.
     *
     * @param {HTMLVideoElement} video - The video element being controlled.
     * @param {HTMLButtonElement} playPauseBtn - The play/pause button element.
     * @param {HTMLInputElement} seekBar - The range input representing playback position.
     * @param {HTMLDivElement} timeDisplay - The element displaying current and total time.
     */
    function updateVideoPlayerUI(video, playPauseBtn, seekBar, timeDisplay) {

        const formatTime = (s) => new Date(1000 * s).toISOString().substr(14, 5);
        seekBar.value = (video.currentTime / video.duration) * 100 || 0;
        timeDisplay.textContent = `${formatTime(video.currentTime)} / ${formatTime(video.duration || 0)}`;
        const icon = video.paused ? 'play' : 'pause';
        playPauseBtn.innerHTML = `<svg class="icon" xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="currentColor">${icon === 'play' ? '<polygon points="5 3 19 12 5 21 5 3"></polygon>' : '<rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect>'}</svg>`;
    }


    // ---------------------------------------------------
    // G. Status Section
    // ---------------------------------------------------

    /**
     * Updates the status text and visual indicator for a tile element.
     *
     * @param {HTMLElement} textEl - The element where the status text is displayed.
     * @param {HTMLElement} indicatorEl - The element whose classes indicate the status.
     * @param {boolean} isPresent - True if present, false otherwise.
     * @param {string} [presentText='Present'] - Text shown if isPresent is true.
     * @param {string} [notPresentText='Not Present'] - Text shown if isPresent is false.
     */
    function updateTileStatus(textEl, indicatorEl, isPresent, presentText = 'Present', notPresentText = 'Not Present') {
        textEl.textContent = isPresent ? presentText : notPresentText;
        indicatorEl.classList.remove('status-present', 'status-not-present', 'status-warning');
        indicatorEl.classList.add(isPresent ? 'status-present' : 'status-not-present');
    }


    /**
     * Updates the API status indicators for API key and URL presence,
     * and enables/disables the corresponding summary option.
     *
     * @param {boolean} keyPresent - Whether an API key is available.
     * @param {boolean} urlPresent - Whether an API URL is set.
     */
    function updateApiStatus(keyPresent, urlPresent) {

        updateTileStatus(apiKeyStatusText, apiKeyStatusIndicator, keyPresent);
        updateTileStatus(apiUrlStatusText, apiUrlStatusIndicator, urlPresent);
        const isApiReady = keyPresent && urlPresent;
        setSummaryOptionEnabled(2, isApiReady);
        apiGroupStatus.classList.remove('status-present', 'status-not-present');
        apiGroupStatus.classList.add(isApiReady ? 'status-present' : 'status-not-present');
    }


    /**
     * Updates the status indicators for a local LLM model and CUDA availability.
     * Applies special handling if a model is present but CUDA is unavailable.
     *
     * @param {boolean} modelPresent - Whether a local LLM model is available.
     * @param {boolean} cudaAvailable - Whether CUDA is available on the system.
     */
    function updateLocalLlmStatus(modelPresent, cudaAvailable) {

        updateTileStatus(localLlmStatusText, localLlmStatusIndicator, modelPresent);
        updateTileStatus(cudaStatusText, cudaStatusIndicator, cudaAvailable, 'Available', 'Not Available');
        localLlmGroupStatus.classList.remove('status-present', 'status-not-present', 'status-warning');
        setSummaryOptionEnabled(1, modelPresent);
        if (modelPresent && cudaAvailable) {
            localLlmGroupStatus.classList.add('status-present');
        } else if (modelPresent && !cudaAvailable) {
            localLlmGroupStatus.classList.add('status-warning');
        } else {
            localLlmGroupStatus.classList.add('status-not-present');
        }
    }


    /**
     * Enables or disables a summary method option in the selector UI.
     * If the currently selected option is disabled, resets the selection to the default (mode 0).
     *
     * @param {number} mode - The numeric identifier of the summary method option:
     * @param {boolean} enabled - Whether the option should be enabled (true) or disabled (false).
     */
    function setSummaryOptionEnabled(mode, enabled) {

        const option = document.querySelector(`#summaryMethodSelector .llm-slider-option[data-mode="${mode}"]`);
        if (!option) return;

        if (enabled) {
            option.classList.remove("disabled");
            option.setAttribute("aria-disabled", "false");
        } else {
            option.classList.add("disabled");
            option.setAttribute("aria-disabled", "true");

            // If the currently selected option is being disabled, reset to default (mode 0)
            if (option.classList.contains("selected")) {
                option.classList.remove("selected");
                const defaultOption = document.querySelector(`#summaryMethodSelector .llm-slider-option[data-mode="0"]`);
                if (defaultOption) defaultOption.classList.add("selected");
            }
        }
    }


    // ---------------------------------------------------
    // H. Start the application
    // ---------------------------------------------------
    init();
});