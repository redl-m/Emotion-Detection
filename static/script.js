document.addEventListener('DOMContentLoaded', () => {
    // ---------------------------------------------------
    // A. Constants & State
    // ---------------------------------------------------
    const EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'];
    const socket = io();
    const D3_COLORS = d3.scaleOrdinal(d3.schemeCategory10);
    const FRAME_SEND_INTERVAL_MS = 200; // ~5 FPS
    const MAX_TIMESERIES_POINTS = 100;

    let state = {
        isTracking: false,
        videoSource: null, // The <video> element
        sendIntervalId: null,
        lastFrameResults: [],
        participants: {}, // { internalId: { color, history: [{ts, probs}] }}
        participantNames: {}, // { internalId: "Participant 1" }
        selectedParticipantId: 'average', // 'average' or a specific internalId
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
        controlsBottom: document.querySelector('.controls-bottom'),
        analysisSection: document.getElementById('analysis-section'),
        participantSelectors: {
            bar: document.getElementById('participant-selector-bar'),
            ts: document.getElementById('participant-selector-ts')
        },
        summaryContainer: document.getElementById('summary-container'),
        summaryReport: document.getElementById('summary-report'),
    };

    const charts = {}; // To hold D3 chart elements

    // ---------------------------------------------------
    // C. Setup & Initialization
    // ---------------------------------------------------
    function init() {
        setupCharts();
        addEventListeners();
        requestAnimationFrame(drawLoop);
    }

    function addEventListeners() {
        dom.uploadInput.addEventListener('change', handleFileUpload);
        dom.liveBtn.addEventListener('click', handleLiveFeed);
        dom.trackBtn.addEventListener('click', toggleTracking);
        dom.placeholder.addEventListener('click', () => dom.uploadInput.click()); // Make placeholder clickable
        socket.on('frame_data', onFrameData);
        socket.on('tracking_summary', renderSummary);
    }

    // ---------------------------------------------------
    // D. Core App Logic (Video & Tracking)
    // ---------------------------------------------------
    function showLoader(isShowing) {
        dom.loader.classList.toggle('hidden', !isShowing);
        dom.placeholderContent.classList.toggle('hidden', isShowing);
    }

    function resetApp() {
        if (state.isTracking) toggleTracking();

        if (state.videoSource) {
            if (state.videoSource.srcObject) {
                state.videoSource.srcObject.getTracks().forEach(track => track.stop());
            }
            state.videoSource.remove();
        }

        Object.assign(state, {
            isTracking: false, videoSource: null, sendIntervalId: null,
            lastFrameResults: [], participants: {}, participantNames: {}, selectedParticipantId: 'average'
        });

        dom.placeholder.classList.replace('placeholder-hidden', 'placeholder-active');
        showLoader(false);
        dom.trackBtn.disabled = true;
        dom.controlsBottom.querySelector('#video-controls')?.remove();
        dom.overlayCtx.clearRect(0, 0, dom.overlayCanvas.width, dom.overlayCanvas.height);
        dom.summaryContainer.classList.add('hidden');
        dom.summaryReport.innerHTML = '';
        renderParticipantSelectors();
        clearCharts();
    }

    async function handleLiveFeed() {
        try {
            resetApp();
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
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

    function handleFileUpload(event) {
        const file = event.target.files[0];
        if (file) {
            resetApp();
            showLoader(true); // Show loader while browser processes the file
            const url = URL.createObjectURL(file);
            const videoEl = document.createElement('video');
            videoEl.src = url;
            videoEl.playsInline = true;
            videoEl.volume = 0.5;
            setupVideoSource(videoEl, false);
            event.target.value = '';
        }
    }

    function setupVideoSource(videoElement, isLive) {
        state.videoSource = videoElement;
        dom.videoWrapper.prepend(videoElement);

        // Immediately update the UI for a more responsive feel, especially for live feeds.
        dom.placeholder.classList.replace('placeholder-active', 'placeholder-hidden');
        dom.trackBtn.disabled = false;
        showLoader(false);

        const resizeCanvas = () => {
            dom.overlayCanvas.width = videoElement.videoWidth;
            dom.overlayCanvas.height = videoElement.videoHeight;
        };

        // Use the appropriate event for each video type to set canvas dimensions.
        if (isLive) {
            // For live streams, 'playing' is the most reliable event to get video dimensions.
            videoElement.addEventListener('playing', resizeCanvas, { once: true });
        } else {
            // For uploaded files, 'loadedmetadata' is correct.
            videoElement.addEventListener('loadedmetadata', resizeCanvas, { once: true });
            createVideoControls();
        }
        // -----------------------
    }

    function toggleTracking() {
        state.isTracking = !state.isTracking;
        dom.trackBtn.classList.toggle('active', state.isTracking);
        dom.trackBtnText.textContent = state.isTracking ? 'Stop & Summarize' : 'Start Tracking';

        if (state.isTracking) {
            state.participants = {};
            state.participantNames = {};
            state.selectedParticipantId = 'average';
            dom.summaryContainer.classList.add('hidden');
            dom.summaryReport.innerHTML = '';
            socket.emit('start_tracking');
            state.sendIntervalId = setInterval(sendFrame, FRAME_SEND_INTERVAL_MS);
        } else {
            clearInterval(state.sendIntervalId);
            state.sendIntervalId = null;
            socket.emit('get_summary');
        }
    }

    function sendFrame() {
        if (!state.videoSource || state.videoSource.paused || state.videoSource.ended) return;

        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = state.videoSource.videoWidth;
        tempCanvas.height = state.videoSource.videoHeight;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(state.videoSource, 0, 0, tempCanvas.width, tempCanvas.height);
        socket.emit('frame', tempCanvas.toDataURL('image/jpeg', 0.7));
    }

    // ---------------------------------------------------
    // E. Real-time Data Handling & Visualization
    // ---------------------------------------------------
    function onFrameData(msg) {
        if (!state.isTracking) return;
        const results = JSON.parse(msg);
        state.lastFrameResults = results;
        const now = new Date();
        let newParticipantDetected = false;

        results.forEach(p => {
            const internalId = p.id;
            if (!state.participants[internalId]) {
                newParticipantDetected = true;
                state.participants[internalId] = {
                    color: D3_COLORS(internalId),
                    history: []
                };
                // Assign a default name
                const participantNum = Object.keys(state.participantNames).length + 1;
                state.participantNames[internalId] = `Participant ${participantNum}`;
            }
            state.participants[internalId].history.push({ timestamp: now, probs: p.probs });
            if (state.participants[internalId].history.length > MAX_TIMESERIES_POINTS) {
                state.participants[internalId].history.shift();
            }
        });

        if (newParticipantDetected) {
            renderParticipantSelectors();
        }

        updateCharts();
    }

    function drawLoop() {
        dom.overlayCtx.clearRect(0, 0, dom.overlayCanvas.width, dom.overlayCanvas.height);
        if (state.isTracking && state.lastFrameResults.length > 0) {
            state.lastFrameResults.forEach(p => {
                const [x, y, w, h] = p.bbox;
                const internalId = p.id;
                const color = state.participants[internalId]?.color || '#FFFFFF';
                const name = state.participantNames[internalId] || `P${internalId}`;

                dom.overlayCtx.strokeStyle = color;
                dom.overlayCtx.lineWidth = 3;
                dom.overlayCtx.strokeRect(x, y, w, h);

                dom.overlayCtx.font = 'bold 16px ' + getComputedStyle(document.body).fontFamily;
                const text = `${name}: ${EMOTIONS[p.emotion]} (${Math.round(p.confidence*100)}%)`;
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
    // F. Charting & Summary
    // ---------------------------------------------------
    function setupCharts() {
        // --- Legend ---
        const legend = d3.select('#legend');
        EMOTIONS.forEach((name, i) => {
            const item = legend.append('div').attr('class', 'legend-item');
            item.append('div').attr('class', 'legend-color').style('background-color', D3_COLORS(i));
            item.append('span').text(name);
        });

        // --- Bar Chart ---
        const bcMargin = { top: 5, right: 20, bottom: 30, left: 65 };
        const bcContainer = d3.select('#prob-bar-chart');
        const bcWidth = bcContainer.node().getBoundingClientRect().width - bcMargin.left - bcMargin.right;
        const bcHeight = 180 - bcMargin.top - bcMargin.bottom;
        charts.bcSvg = bcContainer.append('svg').attr('width', '100%').attr('height', '100%')
            .attr('viewBox', `0 0 ${bcWidth + bcMargin.left + bcMargin.right} ${bcHeight + bcMargin.top + bcMargin.bottom}`)
            .append('g').attr('transform', `translate(${bcMargin.left}, ${bcMargin.top})`);
        charts.yB = d3.scaleBand().domain(EMOTIONS).range([0, bcHeight]).padding(0.2);
        charts.xB = d3.scaleLinear().domain([0, 1]).range([0, bcWidth]);
        charts.bcSvg.append('g').attr('class', 'y-axis-b').call(d3.axisLeft(charts.yB).tickSize(0)).select('.domain').remove();
        charts.bcSvg.append('g').attr('class', 'x-axis-b').attr('transform', `translate(0, ${bcHeight})`)
            .call(d3.axisBottom(charts.xB).ticks(5).tickFormat(d3.format('.0%')));
        charts.bars = charts.bcSvg.selectAll('.bar').data(EMOTIONS.map(e => ({ emotion: e, value: 0 }))).enter()
            .append('rect').attr('class', 'bar-b').attr('y', d => charts.yB(d.emotion)).attr('x', 0)
            .attr('height', charts.yB.bandwidth()).attr('width', 0).attr('rx', 3).style('fill', (d, i) => D3_COLORS(i));

        // --- Time Series ---
        const tsMargin = { top: 5, right: 20, bottom: 30, left: 40 };
        const tsContainer = d3.select('#chart');
        const tsWidth = tsContainer.node().getBoundingClientRect().width - tsMargin.left - tsMargin.right;
        const tsHeight = 180 - tsMargin.top - tsMargin.bottom;
        charts.tsSvg = tsContainer.append('svg').attr('width', '100%').attr('height', '100%')
            .attr('viewBox', `0 0 ${tsWidth + tsMargin.left + tsMargin.right} ${tsHeight + tsMargin.top + tsMargin.bottom}`)
            .append('g').attr('transform', `translate(${tsMargin.left},${tsMargin.top})`);
        charts.xTS = d3.scaleTime().range([0, tsWidth]);
        charts.yTS = d3.scaleLinear().domain([0, 1]).range([tsHeight, 0]);
        charts.tsSvg.append('g').attr('class', 'x-axis-ts').attr('transform', `translate(0,${tsHeight})`);
        charts.tsSvg.append('g').attr('class', 'y-axis-ts').call(d3.axisLeft(charts.yTS).ticks(5).tickFormat(d3.format('.0%')));
        charts.lineGen = EMOTIONS.map((_, i) => d3.line().x(d => charts.xTS(d.timestamp)).y(d => charts.yTS(d.probs[i])).curve(d3.curveMonotoneX));
        charts.lines = charts.tsSvg.selectAll('.line-ts').data(EMOTIONS).enter().append('path')
            .attr('class', 'line-ts').style('stroke', (d, i) => D3_COLORS(i)).style('fill', 'none').style('stroke-width', 2.5);
    }

    function updateCharts() {
        let history = [];
        if (state.selectedParticipantId === 'average') {
            const allHistory = Object.values(state.participants).flatMap(p => p.history);
            if (allHistory.length > 0) {
                 const timeMap = new Map();
                allHistory.forEach(d => {
                    const key = d.timestamp.getTime();
                    if (!timeMap.has(key)) timeMap.set(key, { count: 0, probs: new Array(EMOTIONS.length).fill(0) });
                    const entry = timeMap.get(key);
                    entry.count++;
                    d.probs.forEach((p, i) => entry.probs[i] += p);
                });
                const avgHistory = [];
                timeMap.forEach((value, key) => {
                    avgHistory.push({ timestamp: new Date(key), probs: value.probs.map(p => p / value.count) });
                });
                history = avgHistory.sort((a, b) => a.timestamp - b.timestamp);
            }
        } else if (state.participants[state.selectedParticipantId]) {
            history = state.participants[state.selectedParticipantId].history;
        }

        const latestProbs = history.length > 0 ? history[history.length - 1].probs : new Array(EMOTIONS.length).fill(0);
        const barData = EMOTIONS.map((e, i) => ({ emotion: e, value: latestProbs[i] }));
        charts.bars.data(barData).transition().duration(FRAME_SEND_INTERVAL_MS / 2).attr('width', d => charts.xB(d.value));

        if (history.length < 2) {
            charts.lines.attr('d', null);
            return;
        };
        charts.xTS.domain(d3.extent(history, d => d.timestamp));
        charts.tsSvg.select('.x-axis-ts').transition().duration(FRAME_SEND_INTERVAL_MS / 2).call(d3.axisBottom(charts.xTS).ticks(5));
        charts.lines.data(EMOTIONS).attr('d', (d, i) => charts.lineGen[i](history));
    }

    function clearCharts() {
        const emptyData = EMOTIONS.map(e => ({ emotion: e, value: 0 }));
        charts.bars.data(emptyData).transition().duration(100).attr('width', 0);
        charts.lines.attr('d', null);
    }

    function renderParticipantSelectors() {
        ['bar', 'ts'].forEach(key => {
            const container = dom.participantSelectors[key];
            container.innerHTML = '';
            const selector = document.createElement('div');
            selector.className = 'participant-selector';

            const ids = ['average', ...Object.keys(state.participants)];

            ids.forEach(id => {
                const name = (id === 'average') ? 'Avg' : (state.participantNames[id] || `P${id}`);
                const btn = document.createElement('button');
                btn.className = 'p-btn';
                btn.textContent = name;
                btn.dataset.id = id;
                if (state.selectedParticipantId === id) btn.classList.add('selected');

                btn.addEventListener('click', () => {
                    if (id !== 'average') { // Allow renaming
                        const input = document.createElement('input');
                        input.type = 'text';
                        input.value = name;
                        input.className = 'p-btn-rename-input';
                        btn.replaceWith(input);
                        input.focus();
                        input.select();

                        const saveName = () => {
                            state.participantNames[id] = input.value.trim() || `Participant ${id}`;
                            state.selectedParticipantId = id;
                            renderParticipantSelectors();
                            updateCharts();
                        };
                        input.addEventListener('blur', saveName);
                        input.addEventListener('keydown', e => e.key === 'Enter' && input.blur());
                    } else { // Just select 'average'
                        state.selectedParticipantId = id;
                        renderParticipantSelectors();
                        updateCharts();
                    }
                });
                selector.appendChild(btn);
            });
            container.appendChild(selector);
        });
    }

    function renderSummary(summaryMsg) {
        const summaryData = JSON.parse(summaryMsg);
        if (Object.keys(summaryData).length === 0) return;

        dom.summaryContainer.classList.remove('hidden');
        dom.summaryReport.innerHTML = '';
        dom.analysisSection.scrollTo({ top: dom.analysisSection.scrollHeight, behavior: 'smooth' });

        for (const internalId in summaryData) {
            const p_data = summaryData[internalId];
            const name = state.participantNames[internalId] || `Participant ${parseInt(internalId) + 1}`;

            const card = document.createElement('div');
            card.className = 'summary-card';

            card.innerHTML = `
                <div class="donut-chart" id="donut-${internalId}"></div>
                <div class="summary-details">
                    <h3>${name}</h3>
                    <p>Primarily felt <strong>${p_data.dominant_emotion}</strong>.</p>
                    <p>Detected in <strong>${p_data.total_detections}</strong> frames.</p>
                </div>
            `;
            dom.summaryReport.appendChild(card);
            createDonutChart(`#donut-${internalId}`, p_data.distribution);
        }
    }

    function createDonutChart(selector, distribution) {
        const data = distribution.map((value, i) => ({ value, name: EMOTIONS[i] })).filter(d => d.value > 0);
        const width = 80, height = 80, margin = 5;
        const radius = Math.min(width, height) / 2 - margin;

        const svg = d3.select(selector).append("svg")
            .attr("class", "donut-chart-svg")
            .attr("viewBox", `0 0 ${width} ${height}`)
            .append("g")
            .attr("transform", `translate(${width / 2},${height / 2})`);

        const pie = d3.pie().value(d => d.value).sort(null);
        const arc = d3.arc().innerRadius(radius * 0.5).outerRadius(radius);

        svg.selectAll('path').data(pie(data)).enter().append('path')
            .attr('d', arc)
            .attr('fill', d => D3_COLORS(EMOTIONS.indexOf(d.data.name)));
    }


    // ---------------------------------------------------
    // G. Video Player Controls
    // ---------------------------------------------------
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

    function updateVideoPlayerUI(video, playPauseBtn, seekBar, timeDisplay) {
        const formatTime = (s) => new Date(1000 * s).toISOString().substr(14, 5);
        seekBar.value = (video.currentTime / video.duration) * 100 || 0;
        timeDisplay.textContent = `${formatTime(video.currentTime)} / ${formatTime(video.duration || 0)}`;
        const icon = video.paused ? 'play' : 'pause';
        playPauseBtn.innerHTML = `<svg class="icon" xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="currentColor">${icon === 'play' ? '<polygon points="5 3 19 12 5 21 5 3"></polygon>' : '<rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect>'}</svg>`;
    }

    // ---------------------------------------------------
    // H. Start the application
    // ---------------------------------------------------
    init();
});