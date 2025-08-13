class HeadPoseEstimator {
    constructor() {
        this.faceMesh = null;
        this.videoElement = null;
        this.latestPoseData = {}; // Stores { id: { pitch, yaw, roll, engagement } }
        this.isReady = false;
    }

    /**
     * Initializes the MediaPipe FaceMesh model and sets up the callback.
     * Must be called before any processing.
     * @returns {Promise<void>}
     */
    async initialize() {

        this.faceMesh = new FaceMesh({
            locateFile: (file) => {
                return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
            }
        });


        this.faceMesh.setOptions({
            selfieMode: true,
            maxNumFaces: 1,
            refineLandmarks: true,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });


        this.faceMesh.onResults(this._onResults.bind(this));
        this.isReady = true;
        console.log("HeadPoseEstimator initialized.");
    }

    /**
     * Main processing function. Sends a video frame to FaceMesh for analysis.
     * @param {HTMLVideoElement} videoElement The video element to process.
     * @returns {Promise<void>}
     */
    async processFrame(videoElement) {
        if (!this.isReady || !videoElement) return;
        this.videoElement = videoElement;
        await this.faceMesh.send({image: videoElement});
    }

    /**
     * Returns the most recently calculated pose data.
     * @returns {object} The latest pose data.
     */
    getLatestPoseData() {
        return this.latestPoseData;
    }

    /**
     * [Private] Callback function for when FaceMesh has results.
     * @param {object} results - The results from FaceMesh.
     */
    _onResults(results) {
        if (!window.cvReady || !this.videoElement || !results.multiFaceLandmarks) {
            this.latestPoseData = {}; // Clear old data
            return;
        }

        const newPoseData = {};
        const {videoWidth, videoHeight} = this.videoElement;

        results.multiFaceLandmarks.forEach((landmarks, index) => {
            // NOTE: Using the landmark index as a temporary ID. Your backend tracker will assign a persistent ID.
            const id = index;
            try {
                const [rawPitch, yaw, roll] = this._estimateHeadPose(landmarks, videoWidth, videoHeight);
                const pitch = rawPitch + 180; // offset to normalize
                const engagement = this._computeEngagement(pitch, yaw);
                newPoseData[id] = {pitch, yaw, roll, engagement, rawPitch};

            } catch (e) {
                console.error(`Error estimating head pose for face ${id}:`, e);
            }
        });

        this.latestPoseData = newPoseData;
    }

    /**
     * [Private] Estimates head pose angles from facial landmarks using OpenCV.js.
     * Ensures OpenCV is ready, validates landmarks, and uses correct matrix formats.
     */
    _estimateHeadPose(landmarks, width, height) {
        // Wait for OpenCV.js to be ready
        if (!window.cvReady || typeof cv === "undefined") {
            throw new Error("OpenCV.js not ready yet");
        }

        // Landmark indices for pose estimation
        const landmarkIndices = [1, 152, 263, 33, 287, 57];
        const imagePoints = landmarkIndices.map(i => landmarks[i]);

        // Validate landmarks
        if (imagePoints.some(p => !p || isNaN(p.x) || isNaN(p.y))) {
            throw new Error("Missing or invalid facial landmarks");
        }

        // Convert normalized landmark coords to pixel coords
        const imgPtsArray = imagePoints.flatMap(pt => [pt.x * width, pt.y * height]);

        // 3D model points in mm
        const modelPoints = [
            [0.0, 0.0, 0.0],         // Nose tip
            [0.0, -63.6, -12.5],     // Chin
            [43.3, 32.7, -26.0],     // Right eye outer
            [-43.3, 32.7, -26.0],    // Left eye outer
            [28.9, -28.9, -24.1],    // Right mouth
            [-28.9, -28.9, -24.1]    // Left mouth
        ];

        // Camera internals
        const focalLength = width;
        const center = [width / 2, height / 2];

        // Build OpenCV matrices
        const cameraMatrix = cv.matFromArray(3, 3, cv.CV_64F, [
            focalLength, 0, center[0],
            0, focalLength, center[1],
            0, 0, 1
        ]);
        const distCoeffs = cv.Mat.zeros(4, 1, cv.CV_64F);

        const imageMat = cv.matFromArray(6, 2, cv.CV_64F, imgPtsArray);
        const modelMat = cv.matFromArray(6, 3, cv.CV_64F, modelPoints.flat());

        // Output vectors
        const rvec = new cv.Mat();
        const tvec = new cv.Mat();
        const rotMat = new cv.Mat();

        // Solve PnP
        const success = cv.solvePnP(modelMat, imageMat, cameraMatrix, distCoeffs, rvec, tvec, false, cv.SOLVEPNP_ITERATIVE);
        if (!success) {
            [cameraMatrix, distCoeffs, imageMat, modelMat, rvec, tvec, rotMat].forEach(m => m.delete());
            throw new Error("solvePnP failed");
        }

        // Rodrigues to rotation matrix
        cv.Rodrigues(rvec, rotMat);
        const m = rotMat.data64F;

        // Extract Euler angles
        const sy = Math.hypot(m[0], m[3]);
        let x, y, z;
        if (sy > 1e-6) {
            x = Math.atan2(m[7], m[8]);
            y = Math.atan2(-m[6], sy);
            z = Math.atan2(m[3], m[0]);
        } else {
            x = Math.atan2(-m[5], m[4]);
            y = Math.atan2(-m[6], sy);
            z = 0;
        }

        // Clean up
        [cameraMatrix, distCoeffs, imageMat, modelMat, rvec, tvec, rotMat].forEach(m => m.delete());

        // Convert to degrees
        return [x, y, z].map(rad => rad * 180 / Math.PI);
    }


    /**
     * [Private] Calculates a simple engagement score from pitch and yaw.
     * @param {number} pitch - The pitch angle in degrees.
     * @param {number} yaw - The yaw angle in degrees.
     * @returns {number} An engagement score between 0 and 1.
     */
    _computeEngagement(pitch, yaw) {

        // "Comfort zones" before penalty kicks in
        const yawComfort = 15;   // degrees with no penalty
        const pitchComfort = 10; // degrees with no penalty

        // Max considered angles before "full disengagement"
        const yawMax = 60;
        const pitchMax = 40;

        // Compute smooth penalties between 0 and 1
        const yawPenalty = Math.pow(
            Math.min(Math.max(Math.abs(yaw) - yawComfort, 0) / (yawMax - yawComfort), 1),
            1.5 // exponent controls how fast the drop feels
        );

        const pitchPenalty = Math.pow(
            Math.min(Math.max(Math.abs(pitch) - pitchComfort, 0) / (pitchMax - pitchComfort), 1),
            1.5
        );

        // Weight yaw less than pitch if desired
        const yawWeight = 0.4;
        const pitchWeight = 0.6;

        // Engagement drops based on weighted penalties
        const engagement = 1 - (yawWeight * yawPenalty + pitchWeight * pitchPenalty);

        // Clamp and round
        console.log("Actual Engagement minus penalties" + engagement);
        console.log("Engagement for chart: " + +Math.max(0, Math.min(1, engagement)).toFixed(2));
        return +Math.max(0, Math.min(1, engagement)).toFixed(2);
    }
}