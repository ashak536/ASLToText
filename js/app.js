import {
  HandLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const $ = (id) => document.getElementById(id);
const video = $("video");
const canvas = $("overlay");
const ctx = canvas.getContext("2d");
const drawingUtils = new DrawingUtils(ctx);
const output = $("output");
const cameraButton = $("btnCamera");

const config = {
  dwellMs: 700,
  minProbability: 0.75,
  voteWindow: 7,
  sampleIntervalMs: 100,
};

let landmarker;
let model = null;
let stream = null;
let recording = false;
let lastFrameTime = -1;
let history = [];
let candidate = { letter: null, since: 0 };
let lockedLetter = null;

async function setupLandmarker() {
  if (landmarker) return;
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
}

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
    requestAnimationFrame(loop);
  } catch (error) {
    const message =
      error.name === "NotAllowedError"
        ? "Camera access was denied. Allow camera access and try again."
        : `Unable to start camera: ${error.message}`;
    console.error(error);
  } finally {
    cameraButton.disabled = false;
  }
}

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
  return { letter: model.classes[bestIndex], probability: probabilities[bestIndex] };
}

function handleFrame(letter) {
  const now = performance.now();
  history.push(letter);
  if (history.length > config.voteWindow) history.shift();

  const counts = new Map();
  for (const item of history) {
    counts.set(item, (counts.get(item) ?? 0) + 1);
  }
  const vote = [...counts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] ?? null;

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
}

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
      );
    } else {
      handleFrame(null);
    }
  }
  requestAnimationFrame(loop);
}

$("btnCamera").addEventListener("click", () => {
  if (stream) {
    cameraButton.textContent = "Start camera";
  } else {
    void startCamera();
  }
});

