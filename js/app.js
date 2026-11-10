import {
  HandLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

/* ── DOM refs ────────────────────────────────────────────── */
const $ = (id) => document.getElementById(id);
const video = $("video");
const canvas = $("overlay");
const ctx = canvas.getContext("2d");
const drawingUtils = new DrawingUtils(ctx);
const output = $("output");
const cameraButton = $("btnCamera");
const statusEl = $("status");
const letterDisplay = $("letterDisplay");
const confidenceDisplay = $("confidenceDisplay");
const detectedLetterEl = $("detectedLetter");
const dwellBar = $("dwellBar");

/* ── Config ──────────────────────────────────────────────── */
const config = {
  dwellMs: 700,
  minProbability: 0.75,
  voteWindow: 7,
  sampleIntervalMs: 100,
  recordDurationMs: 3000, // Record samples for 3 seconds per click
};

/* ── State ───────────────────────────────────────────────── */
let landmarker;
let model = null;
let stream = null;
let recording = false;
let recordTimeout = null;
let lastFrameTime = -1;
let history = [];
let candidate = { letter: null, since: 0 };
let lockedLetter = null;
let collectedSamples = []; // { label, features[] }
let currentProgress = 0;

/* ── Model loading ───────────────────────────────────────── */
async function loadModel() {
  try {
    const response = await fetch("model.json");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    model = await response.json();
    console.log(
      `Model loaded: ${model.classes.length} classes (${model.classes.join(", ")})`,
    );
    statusEl.textContent = "Model loaded — start camera to begin";
  } catch (error) {
    console.warn("No model.json found — running in data-collection mode.", error);
    statusEl.textContent = "No model — data collection mode";
  }
}

/* ── MediaPipe setup ─────────────────────────────────────── */
async function setupLandmarker() {
  if (landmarker) return;
  statusEl.textContent = "Loading hand tracker…";
  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm",
  );
  const options = {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 1,
  };

  try {
    landmarker = await HandLandmarker.createFromOptions(fileset, options);
  } catch (gpuError) {
    console.warn("MediaPipe GPU setup failed; retrying with CPU.", gpuError);
    landmarker = await HandLandmarker.createFromOptions(fileset, {
      ...options,
      baseOptions: { ...options.baseOptions, delegate: "CPU" },
    });
  }
  statusEl.textContent = "Hand tracker ready";
}

/* ── Camera ──────────────────────────────────────────────── */
async function startCamera() {
  cameraButton.disabled = true;
  try {
    await setupLandmarker();
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    video.srcObject = stream;
    await new Promise((resolve) => {
      video.addEventListener("loadedmetadata", resolve, { once: true });
    });
    await video.play();
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    cameraButton.textContent = "Stop camera";
    statusEl.textContent = "Camera active — show your hand";
    requestAnimationFrame(loop);
  } catch (error) {
    const message =
      error.name === "NotAllowedError"
        ? "Camera access was denied. Allow camera access and try again."
        : `Unable to start camera: ${error.message}`;
    statusEl.textContent = message;
    console.error(error);
  } finally {
    cameraButton.disabled = false;
  }
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }
  video.srcObject = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  cameraButton.textContent = "Start camera";
  detectedLetterEl.style.display = "none";
  dwellBar.style.width = "0%";
  statusEl.textContent = model ? "Camera stopped" : "No model — data collection mode";
  history = [];
  candidate = { letter: null, since: 0 };
  lockedLetter = null;
}

/* ── Feature extraction ──────────────────────────────────── */
function toFeatures(landmarks) {
  const [wristX, wristY, wristZ] = [
    landmarks[0].x,
    landmarks[0].y,
    landmarks[0].z,
  ];
  const relative = landmarks.map((point) => [
    point.x - wristX,
    point.y - wristY,
    point.z - wristZ,
  ]);
  const scale =
    Math.max(
      ...relative.map(([x, y, z]) => Math.hypot(x, y, z)),
    ) || 1;
  return relative.flat().map((value) => +(value / scale).toFixed(5));
}

/* ── Prediction (logistic regression) ────────────────────── */
function predict(features) {
  if (!model) return null;
  const logits = model.coef.map((weights, index) =>
    weights.reduce(
      (sum, weight, featureIndex) => sum + weight * features[featureIndex],
      model.intercept[index],
    ),
  );
  const maxLogit = Math.max(...logits);
  const exponentials = logits.map((logit) => Math.exp(logit - maxLogit));
  const sum = exponentials.reduce((total, value) => total + value, 0);
  const probabilities = exponentials.map((value) => value / sum);
  const bestIndex = probabilities.indexOf(Math.max(...probabilities));
  return {
    letter: model.classes[bestIndex],
    probability: probabilities[bestIndex],
  };
}

/* ── Data collection ─────────────────────────────────────── */
function collectSample(features) {
  const label = $("letterSelect").value;
  collectedSamples.push({ label, features });
  updateSampleCounts();
}

function updateSampleCounts() {
  const counts = {};
  for (const sample of collectedSamples) {
    counts[sample.label] = (counts[sample.label] || 0) + 1;
  }
  const countsEl = $("counts");
  if (collectedSamples.length === 0) {
    countsEl.textContent = "No samples collected yet.";
    return;
  }
  const entries = Object.entries(counts)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([letter, count]) => `${letter}: ${count}`)
    .join("  ·  ");
  countsEl.textContent = `${collectedSamples.length} total — ${entries}`;
}

function exportCSV() {
  if (collectedSamples.length === 0) {
    alert("No samples collected yet.");
    return;
  }
  const header =
    "label," + Array.from({ length: 63 }, (_, i) => `f${i}`).join(",");
  const rows = collectedSamples.map(
    (s) => `${s.label},${s.features.join(",")}`,
  );
  const csv = [header, ...rows].join("\n");
  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "asl_landmarks.csv";
  a.click();
  URL.revokeObjectURL(url);
}

function clearData() {
  if (
    collectedSamples.length > 0 &&
    !confirm(`Delete all ${collectedSamples.length} collected samples?`)
  )
    return;
  collectedSamples = [];
  updateSampleCounts();
}

/* ── Frame processing ────────────────────────────────────── */
function handleFrame(letter, prediction) {
  const now = performance.now();
  history.push(letter);
  if (history.length > config.voteWindow) history.shift();

  const counts = new Map();
  for (const item of history) {
    counts.set(item, (counts.get(item) ?? 0) + 1);
  }
  const vote =
    [...counts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] ?? null;

  if (vote !== lockedLetter) lockedLetter = null;
  if (vote !== candidate.letter) candidate = { letter: vote, since: now };

  let progress = 0;
  if (vote) {
    progress = lockedLetter
      ? 1
      : Math.min(1, (now - candidate.since) / config.dwellMs);
    if (progress >= 1 && !lockedLetter) {
      output.value += vote;
      lockedLetter = vote;
    }
  }

  // Update visual feedback
  currentProgress = progress;
  dwellBar.style.width = `${(progress * 100).toFixed(0)}%`;

  if (prediction && prediction.probability >= config.minProbability) {
    detectedLetterEl.style.display = "block";
    letterDisplay.textContent = prediction.letter;
    confidenceDisplay.textContent = `${(prediction.probability * 100).toFixed(0)}%`;
    statusEl.textContent = `Detected: ${prediction.letter} (${(prediction.probability * 100).toFixed(0)}%)`;
  } else if (letter === null && prediction) {
    detectedLetterEl.style.display = "block";
    letterDisplay.textContent = prediction.letter;
    letterDisplay.style.opacity = "0.4";
    confidenceDisplay.textContent = `${(prediction.probability * 100).toFixed(0)}% (low)`;
    statusEl.textContent = "Low confidence";
    setTimeout(() => (letterDisplay.style.opacity = "1"), 50);
  } else {
    detectedLetterEl.style.display = "none";
    statusEl.textContent = stream ? "No hand detected" : "Camera stopped";
  }
}

/* ── Render loop ─────────────────────────────────────────── */
function loop() {
  if (!stream) return;

  if (video.readyState >= 2 && video.currentTime !== lastFrameTime) {
    lastFrameTime = video.currentTime;
    const result = landmarker.detectForVideo(video, performance.now());
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (result.landmarks.length) {
      const landmarks = result.landmarks[0];
      drawingUtils.drawConnectors(
        landmarks,
        HandLandmarker.HAND_CONNECTIONS,
        { color: "#1683a5", lineWidth: 3 },
      );
      drawingUtils.drawLandmarks(landmarks, {
        color: "#d84b58",
        radius: 3,
      });
      const features = toFeatures(landmarks);
      if (recording) collectSample(features);
      const prediction = predict(features);
      handleFrame(
        prediction && prediction.probability >= config.minProbability
          ? prediction.letter
          : null,
        prediction,
      );
    } else {
      handleFrame(null, null);
    }
  }
  requestAnimationFrame(loop);
}

/* ── Event listeners ─────────────────────────────────────── */
cameraButton.addEventListener("click", () => {
  if (stream) {
    stopCamera();
  } else {
    void startCamera();
  }
});

// Text controls
$("btnClearText").addEventListener("click", () => {
  output.value = "";
});

$("btnAddSpace").addEventListener("click", () => {
  output.value += " ";
});

$("btnBackspace").addEventListener("click", () => {
  output.value = output.value.slice(0, -1);
});

// Data collection controls
$("btnRecord").addEventListener("click", () => {
  if (!stream) {
    alert("Start the camera first.");
    return;
  }
  const btn = $("btnRecord");
  if (recording) {
    recording = false;
    clearTimeout(recordTimeout);
    btn.textContent = "Record samples";
    btn.style.background = "";
    statusEl.textContent = "Recording stopped";
  } else {
    recording = true;
    btn.textContent = `Recording ${$("letterSelect").value}…`;
    btn.style.background = "#ef4444";
    btn.style.color = "white";
    statusEl.textContent = `Recording letter ${$("letterSelect").value}…`;
    recordTimeout = setTimeout(() => {
      recording = false;
      btn.textContent = "Record samples";
      btn.style.background = "";
      btn.style.color = "";
      statusEl.textContent = "Recording complete";
    }, config.recordDurationMs);
  }
});

$("btnExport").addEventListener("click", exportCSV);
$("btnClearData").addEventListener("click", clearData);

/* ── Init ────────────────────────────────────────────────── */
loadModel();
