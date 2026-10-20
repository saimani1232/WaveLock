// WaveLock mobile: the camera (counterpart of web_backend/camera.py).
//
// Every camera frame is drawn MIRRORED onto a working canvas (the desktop
// flips frames with cv2.flip before MediaPipe), scaled so that its longer
// side is at most 640 px. On that canvas:
//   - MediaPipe Hand Landmarker (web build of the same hand model family,
//     one hand, detection 0.7 / tracking 0.5 as gesture_capture.py) runs on
//     every frame;
//   - YuNet face detection (face_worker.js) runs every 0.25 s, continuously
//     while a liveness challenge is answered, and not during a gesture
//     recording (the desktop pauses its face thread while recording).
// A display canvas shows the frame with the hand skeleton and the face box.
//
// Jobs: a gesture recording (hand landmarks of every frame for `duration`
// seconds + copies of 6 frames at evenly spaced times for the face check), a
// face capture (frames at least 0.35 s apart, each checked by YuNet and
// embedded by SFace) and a per-frame hook used by the liveness challenge.

import { FilesetResolver, HandLandmarker } from "../../vendor/mediapipe/vision_bundle.mjs";
import { Rpc } from "./rpc.js";

const BASE = new URL("../../", import.meta.url).href;
const MAX_SIDE = 640;
const FACE_INTERVAL_MS = 250;            // web_backend/camera.py FACE_DETECT_INTERVAL_SEC
const BRIGHTNESS_INTERVAL_MS = 500;
const FACE_CAPTURE_TIMEOUT_MS = 15000;   // web_backend/camera.py FACE_CAPTURE_TIMEOUT_SEC
const HAND_CONNECTIONS = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10],
  [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];

export class CameraError extends Error {}

export class CameraService {
  constructor() {
    this.state = "stopped";
    this.error = null;
    this.facing = "user";
    this.fps = 0;
    this.processMs = 0;
    this.faceMs = 0;
    this.handVisible = false;
    this.faceRow = null;
    this.faceAt = 0;
    this.brightness = null;
    this.width = 0;
    this.height = 0;
    this.video = document.createElement("video");
    this.video.muted = true;
    this.video.playsInline = true;
    this.video.setAttribute("playsinline", "");
    this.work = document.createElement("canvas");
    this.workCtx = this.work.getContext("2d", { willReadFrequently: true });
    this.display = document.createElement("canvas");
    this.display.className = "feed";
    this.displayCtx = this.display.getContext("2d");
    this.small = document.createElement("canvas");
    this.small.width = 40; this.small.height = 30;
    this.smallCtx = this.small.getContext("2d", { willReadFrequently: true });
    this.stream = null;
    this.landmarker = null;
    this.face = new Rpc(new Worker(new URL("./face_worker.js", import.meta.url), { type: "module" }));
    this.faceBusy = false;
    this.lastFace = 0;
    this.lastBrightness = 0;
    this.lastFrame = 0;
    this.lastTs = 0;
    this.loopToken = 0;
    this.recording = null;
    this.faceCapture = null;
    this.frameHook = null;               // liveness challenge: called on every frame
    this.faceHook = null;                // liveness challenge: called with every face result
    this.frameListeners = new Set();     // camera panels repaint on new frames
  }

  // ---------------------------------------------------------------- lifecycle
  async start(facing = this.facing) {
    if (this.state === "running" && facing === this.facing) return this.status();
    this.stop();
    this.facing = facing;
    this.state = "starting";
    this.error = null;
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new CameraError("This browser cannot open the camera (a secure https:// page is required).");
      }
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: facing, width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
      });
      this.video.srcObject = this.stream;
      await this.video.play();
      await this._waitForSize();
      if (!this.landmarker) this.landmarker = await this._createLandmarker();
      this.state = "running";
      this.face.call({ type: "warm" }).catch(() => {});
      this._loop(++this.loopToken);
    } catch (e) {
      this.stop();
      this.state = "error";
      this.error = e instanceof CameraError ? e.message
        : e && e.name === "NotAllowedError" ? "Camera permission was denied. Allow the camera for this site in the browser settings."
          : e && e.name === "NotFoundError" ? "No camera was found on this device."
            : `Could not open the camera (${(e && e.message) || e}).`;
    }
    return this.status();
  }

  stop() {
    this.loopToken += 1;
    this.cancelJobs("camera stopped");
    if (this.stream) for (const t of this.stream.getTracks()) t.stop();
    this.stream = null;
    this.video.srcObject = null;
    this.handVisible = false;
    this.faceRow = null;
    if (this.state !== "error") this.state = "stopped";
  }

  async _waitForSize() {
    const end = Date.now() + 8000;
    while (!this.video.videoWidth && Date.now() < end) await new Promise((r) => setTimeout(r, 50));
    const vw = this.video.videoWidth, vh = this.video.videoHeight;
    if (!vw) throw new CameraError("The camera did not deliver a picture.");
    const scale = Math.min(1, MAX_SIDE / Math.max(vw, vh));
    this.width = Math.round(vw * scale);
    this.height = Math.round(vh * scale);
    for (const c of [this.work, this.display]) { c.width = this.width; c.height = this.height; }
  }

  async _createLandmarker() {
    const fileset = await FilesetResolver.forVisionTasks(BASE + "vendor/mediapipe/wasm");
    const options = (delegate) => ({
      baseOptions: { modelAssetPath: BASE + "models/hand_landmarker.task", delegate },
      runningMode: "VIDEO",
      numHands: 1,
      minHandDetectionConfidence: 0.7,       // gesture_capture.DETECTION_CONFIDENCE
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,            // gesture_capture.TRACKING_CONFIDENCE
    });
    try {
      return await HandLandmarker.createFromOptions(fileset, options("GPU"));
    } catch (e) {
      return HandLandmarker.createFromOptions(fileset, options("CPU"));
    }
  }

  // ---------------------------------------------------------------- frames
  _loop(token) {
    const next = () => {
      if (token !== this.loopToken) return;
      if (this.video.requestVideoFrameCallback) this.video.requestVideoFrameCallback(step);
      else requestAnimationFrame(step);
    };
    const step = () => {
      if (token !== this.loopToken) return;
      try { this._frame(); } catch (e) { console.error(e); }
      next();
    };
    next();
  }

  _frame() {
    const now = performance.now();
    const t0 = now;
    const ctx = this.workCtx, w = this.width, h = this.height;
    ctx.setTransform(-1, 0, 0, 1, w, 0);               // mirror, like cv2.flip(frame, 1)
    ctx.drawImage(this.video, 0, 0, w, h);
    ctx.setTransform(1, 0, 0, 1, 0, 0);

    const ts = Math.max(this.lastTs + 1, Math.round(now));
    this.lastTs = ts;
    const res = this.landmarker.detectForVideo(this.work, ts);
    const lm = res && res.landmarks && res.landmarks.length ? res.landmarks[0].map((p) => [p.x, p.y, p.z]) : null;
    this.handVisible = lm !== null;
    this.processMs = ema(this.processMs, performance.now() - t0);
    if (this.lastFrame) this.fps = ema(this.fps, 1000 / Math.max(1, now - this.lastFrame), 0.1);
    this.lastFrame = now;

    if (now - this.lastBrightness >= BRIGHTNESS_INTERVAL_MS) {
      this.smallCtx.drawImage(this.work, 0, 0, 40, 30);
      const px = this.smallCtx.getImageData(0, 0, 40, 30).data;
      let sum = 0;
      for (let i = 0; i < px.length; i += 4) sum += 0.114 * px[i + 2] + 0.587 * px[i + 1] + 0.299 * px[i];
      this.brightness = sum / (px.length / 4);
      this.lastBrightness = now;
    }

    const rec = this.recording;
    if (rec) this._feedRecording(rec, lm, now);
    else {
      if (this.frameHook) this.frameHook({ hand: lm, aspect: w / h, now: Date.now() / 1000 });
      const due = this.frameHook || this.faceCapture || now - this.lastFace >= FACE_INTERVAL_MS;
      if (due && !this.faceBusy) this._faceStep(now);
    }
    this._draw(lm);
    for (const fn of this.frameListeners) fn();
  }

  _image() {
    const img = this.workCtx.getImageData(0, 0, this.width, this.height);
    return { data: img.data, width: img.width, height: img.height };
  }

  async _faceStep(now) {
    this.faceBusy = true;
    this.lastFace = now;
    const image = this._image();
    const fcap = this.faceCapture;
    const wantEmbed = fcap && Date.now() - fcap.lastSample >= fcap.gapMs;
    const t0 = performance.now();
    try {
      const out = await this.face.call({ type: wantEmbed ? "detect_embed" : "detect", image });
      this.faceMs = ema(this.faceMs, performance.now() - t0);
      this.faceRow = out.row;
      this.faceAt = performance.now();
      if (this.faceHook) this.faceHook({ row: out.row, image, now: Date.now() / 1000 });
      if (wantEmbed && fcap === this.faceCapture && out.row && out.embedding) {
        fcap.lastSample = Date.now();
        fcap.embeddings.push(out.embedding);
        if (fcap.embeddings.length >= fcap.required) this._finishFaceCapture(fcap, null);
      }
    } catch (e) {
      this.faceRow = null;
    } finally {
      this.faceBusy = false;
    }
  }

  _draw(lm) {
    const ctx = this.displayCtx, w = this.width, h = this.height;
    ctx.drawImage(this.work, 0, 0);
    const row = this.faceRow && performance.now() - this.faceAt < 1000 ? this.faceRow : null;
    if (row) {
      ctx.strokeStyle = "rgba(120, 220, 210, 0.9)";
      ctx.lineWidth = Math.max(1.5, w / 320);
      ctx.strokeRect(row[0], row[1], row[2], row[3]);
    }
    if (lm) {
      ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
      ctx.lineWidth = Math.max(1.5, w / 280);
      ctx.beginPath();
      for (const [a, b] of HAND_CONNECTIONS) {
        ctx.moveTo(lm[a][0] * w, lm[a][1] * h);
        ctx.lineTo(lm[b][0] * w, lm[b][1] * h);
      }
      ctx.stroke();
      ctx.fillStyle = "rgba(230, 60, 60, 0.95)";
      const r = Math.max(2, w / 200);
      for (const p of lm) { ctx.beginPath(); ctx.arc(p[0] * w, p[1] * h, r, 0, 2 * Math.PI); ctx.fill(); }
    }
  }

  // ---------------------------------------------------------------- status
  status() {
    return {
      state: this.state,
      error: this.error,
      camera_index: this.facing === "user" ? "front" : "back",
      facing: this.facing,
      fps: round1(this.fps),
      read_ms: 0,
      process_ms: round1(this.processMs),
      face_ms: round1(this.faceMs),
      hand_visible: this.state === "running" && this.handVisible,
      face_in_view: this.state === "running" && this.faceRow !== null && performance.now() - this.faceAt < 1000,
      frame_aspect: this.width && this.height ? this.width / this.height : null,
      brightness: this.brightness === null ? null : Math.round(this.brightness),
    };
  }

  get running() { return this.state === "running"; }

  // ---------------------------------------------------------------- jobs
  _requireRunning() {
    if (this.state !== "running") throw new CameraError("The camera is not running.");
    if (this.recording || this.faceCapture) throw new Error("The camera is busy with another recording.");
  }

  // Gesture recording: requires a visible hand at the start (as the desktop).
  startRecording(durationSec, { faceSamples = 0 } = {}) {
    this._requireRunning();
    if (!this.handVisible) throw new Error("No hand detected. Hold your hand in front of the camera.");
    let resolve;
    const done = new Promise((r) => { resolve = r; });
    const rec = {
      duration: durationSec, started: null, landmarks: [], framesSeen: 0, faceFrames: [],
      faceSamples, aspect: this.width / this.height, error: null, done, resolve,
      progress() { return this.started === null ? 0 : Math.min(1, (performance.now() - this.started) / (1000 * this.duration)); },
    };
    this.recording = rec;
    return rec;
  }

  _feedRecording(rec, lm, now) {
    if (rec.started === null) rec.started = now;
    rec.framesSeen += 1;
    if (lm) rec.landmarks.push(lm);
    const elapsed = (now - rec.started) / 1000;
    // face samples at evenly spaced times (0, T/(n-1), ..., T): a phone cannot
    // keep every frame like the desktop, which samples evenly by frame index
    if (rec.faceSamples > 0) {
      const k = rec.faceFrames.length;
      const due = rec.faceSamples === 1 ? 0 : (k * rec.duration) / (rec.faceSamples - 1);
      if (k < rec.faceSamples && (elapsed >= due || elapsed >= rec.duration)) rec.faceFrames.push(this._image());
    }
    if (elapsed >= rec.duration) {
      while (rec.faceSamples > 0 && rec.faceFrames.length < rec.faceSamples) rec.faceFrames.push(this._image());
      this.recording = null;
      rec.resolve(rec);
    }
  }

  startFaceCapture(required, gapSec) {
    this._requireRunning();
    let resolve;
    const done = new Promise((r) => { resolve = r; });
    const fcap = { required, gapMs: gapSec * 1000, embeddings: [], lastSample: 0, error: null, done, resolve };
    fcap.timer = setTimeout(() => this._finishFaceCapture(fcap,
      "No face was captured in time. Face the camera in good light and try again."), FACE_CAPTURE_TIMEOUT_MS);
    this.faceCapture = fcap;
    return fcap;
  }

  _finishFaceCapture(fcap, error) {
    if (this.faceCapture !== fcap) return;
    clearTimeout(fcap.timer);
    fcap.error = error;
    this.faceCapture = null;
    fcap.resolve(fcap);
  }

  cancelJobs(reason) {
    const rec = this.recording;
    if (rec) { rec.error = reason; this.recording = null; rec.resolve(rec); }
    if (this.faceCapture) this._finishFaceCapture(this.faceCapture, reason);
    this.frameHook = null;
    this.faceHook = null;
  }

  // Detection + embedding of a stored frame (sign-in face samples, liveness bind).
  async detectEmbed(image) {
    const out = await this.face.call({ type: "detect_embed", image });
    return out;
  }

  async embed(image, row) {
    return (await this.face.call({ type: "embed", image, row })).embedding;
  }

  warmSface() {
    return this.face.call({ type: "warm", sface: true });
  }
}

function ema(prev, value, k = 0.2) {
  return prev ? prev * (1 - k) + value * k : value;
}

function round1(x) {
  return Math.round(x * 10) / 10;
}
