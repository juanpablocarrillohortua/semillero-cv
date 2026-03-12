// MediaPipe Explorer — Semillero CV, Externado
// Uses MediaPipe Tasks Vision JS SDK (WASM, runs in browser)

import {
  PoseLandmarker,
  HandLandmarker,
  FaceLandmarker,
  GestureRecognizer,
  ObjectDetector,
  ImageSegmenter,
  FilesetResolver,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs";

// ── Model registry ──────────────────────────────────────────────────────────

const MODEL_BASE = "https://storage.googleapis.com/mediapipe-models";

const MODELS = {
  pose: {
    name: "Pose Estimation",
    description:
      "Detecta 33 puntos del cuerpo humano en 3D: cabeza, hombros, codos, muñecas, caderas, rodillas, tobillos. Base de biomecánica deportiva y análisis de movimiento.",
    path: `${MODEL_BASE}/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task`,
    create: (vision, path, conf) =>
      PoseLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: path, delegate: "GPU" },
        runningMode: "VIDEO",
        numPoses: 2,
        minPoseDetectionConfidence: conf,
        minTrackingConfidence: conf,
      }),
  },
  hands: {
    name: "Hand Tracking",
    description:
      "Detecta 21 puntos 3D por mano (hasta 2 manos). Identifica cada articulación de cada dedo, la muñeca y la palma. Útil para reconocimiento de gestos.",
    path: `${MODEL_BASE}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`,
    create: (vision, path, conf) =>
      HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: path, delegate: "GPU" },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: conf,
        minTrackingConfidence: conf,
      }),
  },
  face: {
    name: "Face Mesh",
    description:
      "Detecta 478 puntos faciales en 3D incluyendo contorno, ojos, cejas, nariz, labios e iris. Permite análisis de expresiones faciales y tracking de mirada.",
    path: `${MODEL_BASE}/face_landmarker/face_landmarker/float16/latest/face_landmarker.task`,
    create: (vision, path, conf) =>
      FaceLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: path, delegate: "GPU" },
        runningMode: "VIDEO",
        numFaces: 2,
        minFaceDetectionConfidence: conf,
        minTrackingConfidence: conf,
        outputFaceBlendshapes: false,
      }),
  },
  gesture: {
    name: "Gesture Recognition",
    description:
      "Reconoce gestos de la mano: puño cerrado, palma abierta, pulgar arriba/abajo, señalar, victoria, amor. Combina detección de mano + clasificación de gesto.",
    path: `${MODEL_BASE}/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task`,
    create: (vision, path, conf) =>
      GestureRecognizer.createFromOptions(vision, {
        baseOptions: { modelAssetPath: path, delegate: "GPU" },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: conf,
        minTrackingConfidence: conf,
      }),
  },
  object: {
    name: "Object Detection",
    description:
      "Detecta 80 categorías de objetos (COCO): personas, pelotas, botellas, sillas, celulares, etc. Devuelve bounding boxes con clase y confianza.",
    path: `${MODEL_BASE}/object_detector/efficientdet_lite0/float16/latest/efficientdet_lite0.tflite`,
    create: (vision, path, conf) =>
      ObjectDetector.createFromOptions(vision, {
        baseOptions: { modelAssetPath: path, delegate: "GPU" },
        runningMode: "VIDEO",
        maxResults: 10,
        scoreThreshold: conf,
      }),
  },
  segment: {
    name: "Segmentation",
    description:
      "Segmenta la persona del fondo pixel a pixel. Genera una máscara binaria en tiempo real. Útil para efectos de fondo, análisis de silueta y tracking corporal.",
    path: `${MODEL_BASE}/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite`,
    create: (vision, path, conf) =>
      ImageSegmenter.createFromOptions(vision, {
        baseOptions: { modelAssetPath: path, delegate: "GPU" },
        runningMode: "VIDEO",
        outputCategoryMask: true,
        outputConfidenceMasks: false,
      }),
  },
};

// ── DOM elements ────────────────────────────────────────────────────────────

const modelSelect = document.getElementById("model-select");
const confidenceSlider = document.getElementById("confidence");
const confidenceValue = document.getElementById("confidence-value");
const sourceSelect = document.getElementById("source-select");
const btnStart = document.getElementById("btn-start");
const fileInput = document.getElementById("file-input");
const video = document.getElementById("webcam");
const canvas = document.getElementById("output-canvas");
const uploadedImage = document.getElementById("uploaded-image");
const loadingOverlay = document.getElementById("loading-overlay");
const fpsEl = document.getElementById("fps");
const currentModelEl = document.getElementById("current-model");
const detectionsEl = document.getElementById("detections");
const inferenceTimeEl = document.getElementById("inference-time");
const infoTitle = document.getElementById("info-title");
const infoDescription = document.getElementById("info-description");

const ctx = canvas.getContext("2d");

// ── State ───────────────────────────────────────────────────────────────────

let vision = null;
let activeTask = null;
let activeModelKey = null;
let isRunning = false;
let animFrameId = null;
let lastFrameTime = 0;
let fpsBuffer = [];

// ── Init MediaPipe WASM ─────────────────────────────────────────────────────

async function initVision() {
  if (vision) return vision;
  vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"
  );
  return vision;
}

// ── Load model ──────────────────────────────────────────────────────────────

async function loadModel(key) {
  const conf = parseFloat(confidenceSlider.value);
  const model = MODELS[key];

  loadingOverlay.style.display = "flex";

  // Close previous task
  if (activeTask) {
    activeTask.close();
    activeTask = null;
  }

  await initVision();

  try {
    activeTask = await model.create(vision, model.path, conf);
    activeModelKey = key;
    currentModelEl.textContent = model.name;
    infoTitle.textContent = model.name;
    infoDescription.textContent = model.description;
  } catch (err) {
    console.error("Error loading model:", err);
    // Fallback: retry without GPU delegate
    try {
      const fallbackCreate = (v, p, c) => {
        const opts = {
          baseOptions: { modelAssetPath: p },
          runningMode: "VIDEO",
        };
        switch (key) {
          case "pose":
            return PoseLandmarker.createFromOptions(v, {
              ...opts, numPoses: 2,
              minPoseDetectionConfidence: c, minTrackingConfidence: c,
            });
          case "hands":
            return HandLandmarker.createFromOptions(v, {
              ...opts, numHands: 2,
              minHandDetectionConfidence: c, minTrackingConfidence: c,
            });
          case "face":
            return FaceLandmarker.createFromOptions(v, {
              ...opts, numFaces: 2,
              minFaceDetectionConfidence: c, minTrackingConfidence: c,
              outputFaceBlendshapes: false,
            });
          case "gesture":
            return GestureRecognizer.createFromOptions(v, {
              ...opts, numHands: 2,
              minHandDetectionConfidence: c, minTrackingConfidence: c,
            });
          case "object":
            return ObjectDetector.createFromOptions(v, {
              ...opts, maxResults: 10, scoreThreshold: c,
            });
          case "segment":
            return ImageSegmenter.createFromOptions(v, {
              ...opts, outputCategoryMask: true, outputConfidenceMasks: false,
            });
        }
      };
      activeTask = await fallbackCreate(vision, model.path, conf);
      activeModelKey = key;
      currentModelEl.textContent = model.name;
      infoTitle.textContent = model.name;
      infoDescription.textContent = model.description;
    } catch (err2) {
      console.error("Fallback also failed:", err2);
      infoTitle.textContent = "Error";
      infoDescription.textContent = `No se pudo cargar ${model.name}: ${err2.message}`;
    }
  }

  loadingOverlay.style.display = "none";
}

// ── Drawing functions ───────────────────────────────────────────────────────

const COLORS = {
  primary: "#4a9eff",
  secondary: "#ff6b6b",
  green: "#4aff9e",
  yellow: "#ffe14a",
};

function clearCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function drawLandmarks(landmarks, connections, color = COLORS.primary, radius = 3) {
  if (!landmarks || landmarks.length === 0) return;

  const w = canvas.width;
  const h = canvas.height;

  // Draw connections
  if (connections) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.globalAlpha = 0.7;
    for (const [start, end] of connections) {
      if (start < landmarks.length && end < landmarks.length) {
        const s = landmarks[start];
        const e = landmarks[end];
        ctx.beginPath();
        ctx.moveTo(s.x * w, s.y * h);
        ctx.lineTo(e.x * w, e.y * h);
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1.0;
  }

  // Draw points
  ctx.fillStyle = color;
  for (const lm of landmarks) {
    ctx.beginPath();
    ctx.arc(lm.x * w, lm.y * h, radius, 0, 2 * Math.PI);
    ctx.fill();
  }
}

function drawBoundingBox(x, y, w, h, label, score, color = COLORS.primary) {
  const cw = canvas.width;
  const ch = canvas.height;
  const bx = x * cw;
  const by = y * ch;
  const bw = w * cw;
  const bh = h * ch;

  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.strokeRect(bx, by, bw, bh);

  const text = `${label} ${(score * 100).toFixed(0)}%`;
  ctx.font = "bold 14px system-ui";
  const textWidth = ctx.measureText(text).width;
  ctx.fillStyle = color;
  ctx.fillRect(bx, by - 22, textWidth + 10, 22);
  ctx.fillStyle = "#000";
  ctx.fillText(text, bx + 5, by - 6);
}

function drawSegmentationMask(mask) {
  if (!mask) return;

  const w = canvas.width;
  const h = canvas.height;
  const imageData = ctx.createImageData(w, h);
  const maskData = mask.getAsUint8Array();

  // Scale mask to canvas if needed
  const mw = mask.width;
  const mh = mask.height;

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const mx = Math.floor((x / w) * mw);
      const my = Math.floor((y / h) * mh);
      const maskVal = maskData[my * mw + mx];
      const idx = (y * w + x) * 4;

      if (maskVal > 0) {
        // Person: semi-transparent blue
        imageData.data[idx] = 74;
        imageData.data[idx + 1] = 158;
        imageData.data[idx + 2] = 255;
        imageData.data[idx + 3] = 100;
      } else {
        // Background: darken
        imageData.data[idx] = 0;
        imageData.data[idx + 1] = 0;
        imageData.data[idx + 2] = 0;
        imageData.data[idx + 3] = 150;
      }
    }
  }

  ctx.putImageData(imageData, 0, 0);
}

function drawGestureLabel(gesture, handedness, x, y) {
  const text = `${gesture} (${handedness})`;
  ctx.font = "bold 18px system-ui";
  const tw = ctx.measureText(text).width;
  ctx.fillStyle = COLORS.yellow;
  ctx.fillRect(x - 5, y - 24, tw + 10, 30);
  ctx.fillStyle = "#000";
  ctx.fillText(text, x, y);
}

// ── Pose connections (MediaPipe standard) ───────────────────────────────────

const POSE_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],
  [9,10],[11,12],[11,13],[13,15],[15,17],[15,19],[15,21],
  [17,19],[12,14],[14,16],[16,18],[16,20],[16,22],[18,20],
  [11,23],[12,24],[23,24],[23,25],[24,26],[25,27],[26,28],
  [27,29],[28,30],[29,31],[30,32],[27,31],[28,32],
];

const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],       // thumb
  [0,5],[5,6],[6,7],[7,8],       // index
  [0,9],[9,10],[10,11],[11,12],  // middle
  [0,13],[13,14],[14,15],[15,16],// ring
  [0,17],[17,18],[18,19],[19,20],// pinky
  [5,9],[9,13],[13,17],          // palm
];

// ── Process frame ───────────────────────────────────────────────────────────

function processFrame(source, timestamp) {
  if (!activeTask || !activeModelKey) return 0;

  const key = activeModelKey;
  let detectionCount = 0;

  try {
    switch (key) {
      case "pose": {
        const result = activeTask.detectForVideo(source, timestamp);
        if (result.landmarks) {
          for (const landmarks of result.landmarks) {
            drawLandmarks(landmarks, POSE_CONNECTIONS, COLORS.primary, 4);
          }
          detectionCount = result.landmarks.length;
        }
        break;
      }
      case "hands": {
        const result = activeTask.detectForVideo(source, timestamp);
        if (result.landmarks) {
          const colors = [COLORS.green, COLORS.secondary];
          for (let i = 0; i < result.landmarks.length; i++) {
            drawLandmarks(result.landmarks[i], HAND_CONNECTIONS, colors[i % 2], 3);
          }
          detectionCount = result.landmarks.length;
        }
        break;
      }
      case "face": {
        const result = activeTask.detectForVideo(source, timestamp);
        if (result.faceLandmarks) {
          for (const landmarks of result.faceLandmarks) {
            // Draw only points (no connections — too many for face mesh)
            const w = canvas.width;
            const h = canvas.height;
            ctx.fillStyle = COLORS.primary;
            ctx.globalAlpha = 0.6;
            for (const lm of landmarks) {
              ctx.beginPath();
              ctx.arc(lm.x * w, lm.y * h, 1.2, 0, 2 * Math.PI);
              ctx.fill();
            }
            ctx.globalAlpha = 1.0;
          }
          detectionCount = result.faceLandmarks.length;
        }
        break;
      }
      case "gesture": {
        const result = activeTask.recognizeForVideo(source, timestamp);
        if (result.landmarks) {
          for (let i = 0; i < result.landmarks.length; i++) {
            drawLandmarks(result.landmarks[i], HAND_CONNECTIONS, COLORS.green, 3);
            if (result.gestures && result.gestures[i] && result.gestures[i].length > 0) {
              const gesture = result.gestures[i][0].categoryName;
              const handedness = result.handedness?.[i]?.[0]?.categoryName || "";
              const wrist = result.landmarks[i][0];
              drawGestureLabel(gesture, handedness, wrist.x * canvas.width, wrist.y * canvas.height - 30);
            }
          }
          detectionCount = result.landmarks.length;
        }
        break;
      }
      case "object": {
        const result = activeTask.detectForVideo(source, timestamp);
        if (result.detections) {
          for (const det of result.detections) {
            const bb = det.boundingBox;
            if (bb && det.categories.length > 0) {
              const cat = det.categories[0];
              drawBoundingBox(
                bb.originX / source.videoWidth,
                bb.originY / source.videoHeight,
                bb.width / source.videoWidth,
                bb.height / source.videoHeight,
                cat.categoryName,
                cat.score,
                COLORS.yellow
              );
            }
          }
          detectionCount = result.detections.length;
        }
        break;
      }
      case "segment": {
        const result = activeTask.segmentForVideo(source, timestamp);
        if (result.categoryMask) {
          drawSegmentationMask(result.categoryMask);
          result.categoryMask.close();
          detectionCount = 1;
        }
        break;
      }
    }
  } catch (err) {
    // Skip frame on error (can happen during model switch)
    console.warn("Frame processing error:", err.message);
  }

  return detectionCount;
}

// ── Animation loop ──────────────────────────────────────────────────────────

function loop() {
  if (!isRunning) return;

  const now = performance.now();

  // Match canvas to video resolution
  if (video.videoWidth && video.videoHeight) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }

  clearCanvas();

  const t0 = performance.now();
  const detections = processFrame(video, now);
  const inferenceMs = performance.now() - t0;

  // FPS calculation
  fpsBuffer.push(now);
  fpsBuffer = fpsBuffer.filter((t) => now - t < 1000);
  const fps = fpsBuffer.length;

  // Update stats
  fpsEl.textContent = fps;
  detectionsEl.textContent = detections;
  inferenceTimeEl.textContent = inferenceMs.toFixed(1);

  animFrameId = requestAnimationFrame(loop);
}

// ── Webcam ──────────────────────────────────────────────────────────────────

async function startWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: false,
    });
    video.srcObject = stream;
    video.style.display = "block";
    uploadedImage.style.display = "none";
    await video.play();

    await loadModel(modelSelect.value);

    isRunning = true;
    btnStart.textContent = "Detener";
    btnStart.classList.add("active");
    loop();
  } catch (err) {
    console.error("Webcam error:", err);
    infoTitle.textContent = "Error de webcam";
    infoDescription.textContent = `No se pudo acceder a la cámara: ${err.message}. Intenta con "Subir imagen".`;
  }
}

function stopWebcam() {
  isRunning = false;
  if (animFrameId) cancelAnimationFrame(animFrameId);

  const stream = video.srcObject;
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    video.srcObject = null;
  }

  clearCanvas();
  btnStart.textContent = "Iniciar Webcam";
  btnStart.classList.remove("active");
  fpsEl.textContent = "--";
  detectionsEl.textContent = "--";
  inferenceTimeEl.textContent = "--";
}

// ── Image upload ────────────────────────────────────────────────────────────

async function handleImageUpload(file) {
  const url = URL.createObjectURL(file);

  return new Promise((resolve) => {
    const img = uploadedImage;
    img.onload = async () => {
      video.style.display = "none";
      img.style.display = "block";
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;

      // Load model in IMAGE mode
      const key = modelSelect.value;
      const conf = parseFloat(confidenceSlider.value);
      const model = MODELS[key];

      loadingOverlay.style.display = "flex";
      await initVision();

      if (activeTask) {
        activeTask.close();
        activeTask = null;
      }

      try {
        // Create in IMAGE mode
        const opts = {
          baseOptions: { modelAssetPath: model.path },
          runningMode: "IMAGE",
        };

        switch (key) {
          case "pose":
            activeTask = await PoseLandmarker.createFromOptions(vision, {
              ...opts, numPoses: 2,
              minPoseDetectionConfidence: conf,
            });
            break;
          case "hands":
            activeTask = await HandLandmarker.createFromOptions(vision, {
              ...opts, numHands: 2,
              minHandDetectionConfidence: conf,
            });
            break;
          case "face":
            activeTask = await FaceLandmarker.createFromOptions(vision, {
              ...opts, numFaces: 2,
              minFaceDetectionConfidence: conf,
              outputFaceBlendshapes: false,
            });
            break;
          case "gesture":
            activeTask = await GestureRecognizer.createFromOptions(vision, {
              ...opts, numHands: 2,
              minHandDetectionConfidence: conf,
            });
            break;
          case "object":
            activeTask = await ObjectDetector.createFromOptions(vision, {
              ...opts, maxResults: 10, scoreThreshold: conf,
            });
            break;
          case "segment":
            activeTask = await ImageSegmenter.createFromOptions(vision, {
              ...opts, outputCategoryMask: true, outputConfidenceMasks: false,
            });
            break;
        }

        activeModelKey = key;
        currentModelEl.textContent = model.name;
        infoTitle.textContent = model.name;
        infoDescription.textContent = model.description;
      } catch (err) {
        console.error("Error:", err);
      }

      loadingOverlay.style.display = "none";

      // Process single image
      clearCanvas();
      const t0 = performance.now();
      let detections = 0;

      try {
        switch (key) {
          case "pose": {
            const r = activeTask.detect(img);
            if (r.landmarks) {
              for (const lm of r.landmarks) drawLandmarks(lm, POSE_CONNECTIONS, COLORS.primary, 4);
              detections = r.landmarks.length;
            }
            break;
          }
          case "hands": {
            const r = activeTask.detect(img);
            if (r.landmarks) {
              const c = [COLORS.green, COLORS.secondary];
              for (let i = 0; i < r.landmarks.length; i++)
                drawLandmarks(r.landmarks[i], HAND_CONNECTIONS, c[i % 2], 3);
              detections = r.landmarks.length;
            }
            break;
          }
          case "face": {
            const r = activeTask.detect(img);
            if (r.faceLandmarks) {
              for (const fl of r.faceLandmarks) {
                ctx.fillStyle = COLORS.primary;
                ctx.globalAlpha = 0.6;
                for (const lm of fl) {
                  ctx.beginPath();
                  ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 1.2, 0, 2 * Math.PI);
                  ctx.fill();
                }
                ctx.globalAlpha = 1.0;
              }
              detections = r.faceLandmarks.length;
            }
            break;
          }
          case "gesture": {
            const r = activeTask.recognize(img);
            if (r.landmarks) {
              for (let i = 0; i < r.landmarks.length; i++) {
                drawLandmarks(r.landmarks[i], HAND_CONNECTIONS, COLORS.green, 3);
                if (r.gestures?.[i]?.[0]) {
                  const g = r.gestures[i][0].categoryName;
                  const h = r.handedness?.[i]?.[0]?.categoryName || "";
                  const wr = r.landmarks[i][0];
                  drawGestureLabel(g, h, wr.x * canvas.width, wr.y * canvas.height - 30);
                }
              }
              detections = r.landmarks.length;
            }
            break;
          }
          case "object": {
            const r = activeTask.detect(img);
            if (r.detections) {
              for (const det of r.detections) {
                const bb = det.boundingBox;
                if (bb && det.categories.length > 0) {
                  const cat = det.categories[0];
                  drawBoundingBox(
                    bb.originX / img.naturalWidth, bb.originY / img.naturalHeight,
                    bb.width / img.naturalWidth, bb.height / img.naturalHeight,
                    cat.categoryName, cat.score, COLORS.yellow
                  );
                }
              }
              detections = r.detections.length;
            }
            break;
          }
          case "segment": {
            const r = activeTask.segment(img);
            if (r.categoryMask) {
              drawSegmentationMask(r.categoryMask);
              r.categoryMask.close();
              detections = 1;
            }
            break;
          }
        }
      } catch (err) {
        console.error("Detection error:", err);
      }

      const inferenceMs = performance.now() - t0;
      fpsEl.textContent = "N/A";
      detectionsEl.textContent = detections;
      inferenceTimeEl.textContent = inferenceMs.toFixed(1);

      resolve();
    };
    img.src = url;
  });
}

// ── Event listeners ─────────────────────────────────────────────────────────

btnStart.addEventListener("click", () => {
  if (sourceSelect.value === "upload") {
    fileInput.click();
    return;
  }
  if (isRunning) {
    stopWebcam();
  } else {
    startWebcam();
  }
});

fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) handleImageUpload(file);
});

modelSelect.addEventListener("change", async () => {
  const info = MODELS[modelSelect.value];
  infoTitle.textContent = info.name;
  infoDescription.textContent = info.description;

  if (isRunning) {
    // Reload model while webcam is active
    await loadModel(modelSelect.value);
  }
});

confidenceSlider.addEventListener("input", () => {
  confidenceValue.textContent = confidenceSlider.value;
});

sourceSelect.addEventListener("change", () => {
  if (sourceSelect.value === "upload") {
    btnStart.textContent = "Subir Imagen";
    if (isRunning) stopWebcam();
  } else {
    btnStart.textContent = isRunning ? "Detener" : "Iniciar Webcam";
  }
});

// ── Init ────────────────────────────────────────────────────────────────────

const initInfo = MODELS[modelSelect.value];
infoTitle.textContent = initInfo.name;
infoDescription.textContent = initInfo.description;
