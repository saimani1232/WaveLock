// WaveLock mobile: the local service (counterpart of web_backend/service.py).
//
// The web pages call the same API as with the Python server (api.js), and
// get the same JSON shapes. Every decision, calibration, attempt count and
// stored file comes from the desktop Python code running in py_worker.js
// (mobile_core.py); this file only sequences the steps in time, as the
// service's worker threads do on the desktop. Nothing leaves the device.

import { Rpc } from "./rpc.js";
import { CameraService, CameraError } from "./camera_service.js";

export class ApiError extends Error {
  constructor(status, message, code = null) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

const SETTINGS_KEY = "wavelock-mobile-settings";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const nowSec = () => Date.now() / 1000;

function loadSettings() {
  const defaults = { debug: false, liveness: true, facing: "user" };
  try { return Object.assign(defaults, JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}")); } catch (e) { return defaults; }
}

export class LocalService {
  constructor() {
    this.settings = loadSettings();
    this.camera = new CameraService();
    this.camera.facing = this.settings.facing;
    this.progressText = "Starting";
    this.progressListeners = new Set();
    this.py = new Rpc(new Worker(new URL("./py_worker.js", import.meta.url)), (t) => {
      this.progressText = t;
      for (const fn of this.progressListeners) fn(t);
    });
    this.ready = null;
    this.info = null;
    this.C = null;                       // constants from the Python modules
    this.activity = null;                // null | "auth" | "register" | "face"
    this.auth = null;
    this.reg = null;
    this.face = null;
  }

  boot() {
    if (!this.ready) {
      this.ready = (async () => {
        this.info = await this.py.call({ method: "__boot__",
          kwargs: { debug: this.settings.debug, liveness: this.settings.liveness } });
        this.C = await this.call("constants");
        return this.info;
      })();
    }
    return this.ready;
  }

  async call(method, kwargs = {}) {
    const r = await this.py.call({ method, kwargs });
    if (r && r.__error__) throw new ApiError(r.__error__.status, r.__error__.message, r.__error__.code);
    return r;
  }

  saveSettings(patch) {
    Object.assign(this.settings, patch);
    try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(this.settings)); } catch (e) { /* private mode */ }
  }

  // ------------------------------------------------------------ helpers (service.py)
  _claim(name) {
    if (this.activity === "auth" && name !== "auth" && this.auth && ["ready", "result"].includes(this.auth.phase)) {
      this.call("auth_end", { username: this.auth.username }).catch(() => {});
      this.auth = null;
      this.activity = null;
    }
    if (this.activity !== null && this.activity !== name) {
      const where = { auth: "A sign-in attempt is running. Wait a few seconds.",
        register: "A registration is in progress. Open Register to continue or cancel it.",
        face: "A face enrolment is in progress. Wait for it to finish." };
      throw new ApiError(409, where[this.activity], "busy");
    }
    this.activity = name;
  }

  _release(name) {
    if (this.activity === name) this.activity = null;
  }

  _startRecording(withFace) {
    try {
      return this.camera.startRecording(this.C.recording_sec, { faceSamples: withFace ? 6 : 0 });
    } catch (e) {
      if (e instanceof CameraError) throw new ApiError(503, e.message, "camera");
      throw new ApiError(409, e.message, /hand/i.test(e.message) ? "no_hand" : "busy");
    }
  }

  _startFaceCapture() {
    try {
      return this.camera.startFaceCapture(this.C.face_required_frames, this.C.face_sample_gap_sec);
    } catch (e) {
      throw new ApiError(e instanceof CameraError ? 503 : 409, e.message, e instanceof CameraError ? "camera" : "busy");
    }
  }

  // ------------------------------------------------------------ overview
  async system() {
    await this.boot();
    const info = await this.call("system");
    info.debug = this.settings.debug;
    info.liveness = this.settings.liveness;
    info.camera = this.camera.status();
    info.activity = this.activity;
    info.models = { yunet: true, sface: true, hands: true };
    info.python = this.info.python;
    info.mobile = { numpy: this.info.numpy, scipy: this.info.scipy, facing: this.settings.facing };
    delete info.templates_dir;
    return info;
  }

  async users() { await this.boot(); return this.call("users"); }

  async activityLog(limit = 50, user = null) { await this.boot(); return this.call("activity", { limit, user }); }

  async status() {
    return {
      camera: this.camera.status(),
      activity: this.activity,
      auth: this._authState(),
      register: this._regState(),
      face: this._faceState(),
      bench: null,
    };
  }

  // ------------------------------------------------------------ camera
  async cameraStart(index = null) {
    const facing = index === "back" ? "environment" : index === "front" ? "user" : this.settings.facing;
    if (index !== null) this.saveSettings({ facing });
    return this.camera.start(facing);
  }

  async cameraStop() {
    if (this.activity !== null) {
      throw new ApiError(409, "Finish or cancel the current activity before stopping the camera.", "busy");
    }
    this.camera.stop();
    return this.camera.status();
  }

  // ------------------------------------------------------------ sign-in
  async authBegin(rawUsername) {
    await this.boot();
    const info = await this.call("auth_begin", { raw_username: rawUsername });
    this._claim("auth");
    const locked = await this.call("locked_for", { username: info.username });
    this.auth = { username: info.username, info, phase: "ready", result: null, error: null, attempt: 0,
      notice: null, lockedUntil: Date.now() + 1000 * locked, challenge: null, readyUntil: null };
    return this._authState();
  }

  async authRecord() {
    const a = this.auth;
    if (!a || this.activity !== "auth") throw new ApiError(409, "Enter a username to start signing in.", "no_session");
    if (a.phase === "challenge" && a.readyUntil) {
      a.go = true;                                   // liveness passed: the user starts the recording
      return this._authState();
    }
    if (["challenge", "recording", "processing"].includes(a.phase)) throw new ApiError(409, "An attempt is already running.", "busy");
    a.notice = null;
    const left = await this.call("locked_for", { username: a.username });
    a.lockedUntil = Date.now() + 1000 * left;
    if (left > 0) throw new ApiError(423, `Too many failed attempts. Try again in ${left.toFixed(0)} s.`, "locked");
    if (this.settings.liveness) return this._authChallenge(a);
    const rec = this._startRecording(a.info.face_enrolled);
    Object.assign(a, { phase: "recording", result: null, error: null, rec });
    a.attempt += 1;
    this._authWorker(a, rec);
    return this._authState();
  }

  async _authChallenge(a) {
    if (!this.camera.running) throw new ApiError(503, "The camera is not running.", "camera");
    const pub = await this.call("challenge_new", { username: a.username, now: nowSec() });
    Object.assign(a, { phase: "challenge", result: null, error: null, challenge: pub, readyUntil: null,
      go: false, notice: null, bestYaw: Infinity, bestFace: null });
    a.attempt += 1;
    let finished = false;
    let handPending = false;
    const handle = (st) => {
      if (!st || finished || this.auth !== a) return;
      a.challenge = st.public;
      if (st.public.state !== "running") {
        finished = true;
        this.camera.frameHook = null;
        this.camera.faceHook = null;
        this._challengeWorker(a, st.public);
      }
    };
    this.camera.frameHook = ({ hand, aspect, now }) => {
      if (handPending) return;                       // one hand update in flight at a time
      handPending = true;
      this.call("challenge_feed", { hand, frame_aspect: aspect, now, hand_seen: true })
        .then(handle).catch(() => {}).finally(() => { handPending = false; });
    };
    this.camera.faceHook = ({ row, image, now }) => {
      this.call("challenge_feed", { face_row: row, face_seen: true, hand_seen: false, now })
        .then((st) => {
          // keep the most frontal face of the challenge for the identity check
          if (st && st.abs_yaw !== null && st.abs_yaw <= a.bestYaw && row) {
            a.bestYaw = st.abs_yaw;
            a.bestFace = { image, row };
          }
          handle(st);
        }).catch(() => {});
    };
    return this._authState();
  }

  async _challengeWorker(a, pub) {
    let state = pub.state, reason = pub.reason;
    if (reason === "liveness_cancelled") {
      if (this.auth === a) a.phase = "ready";
      return;
    }
    try {
      if (state === "passed" && a.info.face_enrolled) {
        // Bind the challenge to the enrolled face (web_backend.service._challenge_worker).
        let embedding = null;
        if (a.bestFace) embedding = await this.camera.embed(a.bestFace.image, a.bestFace.row);
        const bind = await this.call("challenge_bind", { username: a.username, embedding });
        if (!bind.passed) { state = "failed"; reason = bind.reason; }
      }
    } catch (e) {
      state = "failed"; reason = reason || "liveness_no_face";
    }
    if (this.auth !== a) return;
    if (state !== "passed") { await this._livenessFailed(a, reason || "liveness_failed"); return; }
    // Passed: the user starts the recording when ready (R / Space / button).
    a.readyUntil = Date.now() + 1000 * this.C.record_ready_window_sec;
    while (this.auth === a && !a.go) {
      if (Date.now() >= a.readyUntil) {
        Object.assign(a, { phase: "ready", readyUntil: null, notice: "ready_timeout" });
        return;
      }
      await sleep(50);
    }
    if (this.auth !== a) return;
    const end = Date.now() + 1000 * this.C.hand_wait_after_challenge_sec;
    while (!this.camera.handVisible && Date.now() < end && this.auth === a) await sleep(50);
    if (this.auth !== a) return;
    let rec;
    try {
      rec = this.camera.startRecording(this.C.recording_sec, { faceSamples: a.info.face_enrolled ? 6 : 0 });
    } catch (e) {
      await this._livenessFailed(a, "liveness_no_hand_for_gesture");
      return;
    }
    Object.assign(a, { phase: "recording", rec, readyUntil: null });
    this._authWorker(a, rec);
  }

  async _livenessFailed(a, reason) {
    try {
      const payload = await this.call("liveness_failed", { username: a.username, reason });
      a.lockedUntil = Date.now() + 1000 * (payload.locked_for || 0);
      Object.assign(a, { result: payload, phase: "result" });
    } catch (e) {
      Object.assign(a, { phase: "ready", error: `Evaluation failed: ${e.message}` });
    }
  }

  async _authWorker(a, rec) {
    await rec.done;
    if (this.auth !== a) return;
    if (rec.error) { Object.assign(a, { phase: "ready", error: `Recording stopped (${rec.error}).` }); return; }
    a.phase = "processing";
    try {
      let faceSamples = null;
      if (a.info.face_enrolled) {
        faceSamples = [];
        for (const image of rec.faceFrames) {
          const out = await this.camera.detectEmbed(image);
          faceSamples.push(out.row ? out.embedding : null);
        }
      }
      const payload = await this.call("evaluate", { username: a.username, landmarks: rec.landmarks,
        frame_aspect: rec.aspect, face_samples: faceSamples });
      if (this.auth !== a) return;
      a.lockedUntil = Date.now() + 1000 * (payload.locked_for || 0);
      Object.assign(a, { result: payload, phase: "result" });
    } catch (e) {
      if (this.auth === a) Object.assign(a, { phase: "ready", error: `Evaluation failed: ${e.message}` });
    }
  }

  async authReset() {
    const a = this.auth;
    if (a && ["result", "ready"].includes(a.phase)) Object.assign(a, { phase: "ready", result: null, error: null });
    return this._authState();
  }

  async authEnd() {
    const a = this.auth;
    if (a && ["challenge", "recording"].includes(a.phase)) {
      this.camera.cancelJobs("cancelled");
      this.call("challenge_cancel").catch(() => {});
    }
    if (a) this.call("auth_end", { username: a.username }).catch(() => {});
    this.auth = null;
    this._release("auth");
    return { ok: true };
  }

  _authState() {
    const a = this.auth;
    if (!a) return null;
    const rec = a.rec;
    const state = {
      username: a.username,
      phase: a.phase,
      face_enrolled: a.info.face_enrolled,
      templates: a.info.templates,
      attempt: a.attempt,
      result: a.result,
      error: a.error,
      locked_for: Math.max(0, Math.round((a.lockedUntil - Date.now()) / 100) / 10),
      recording: a.phase === "recording" && rec ? { progress: Math.round(rec.progress() * 1000) / 1000,
        hand_frames: rec.landmarks.length, duration: rec.duration } : null,
      challenge: a.phase === "challenge" && a.challenge && a.challenge.state === "running" ? a.challenge : null,
      record_ready: a.phase === "challenge" && a.readyUntil ? Math.max(0, Math.round((a.readyUntil - Date.now()) / 100) / 10) : null,
      notice: a.notice,
      liveness: this.settings.liveness,
    };
    if (this.settings.debug) state.pose_method = a.info.pose_method;
    return state;
  }

  // ------------------------------------------------------------ registration
  async registerBegin(rawUsername, overwrite = false, poseMethod = "legacy", checkLevel = "standard") {
    await this.boot();
    if (this.reg && this.reg.status === "active") {
      throw new ApiError(409, `A registration for '${this.reg.username}' is already in progress. ` +
        "Open Register to continue or cancel it.", "busy");
    }
    const st = await this.call("reg_begin", { raw_username: rawUsername, overwrite: Boolean(overwrite),
      pose_method: poseMethod, check_level: checkLevel });
    try {
      this._claim("register");
    } catch (e) {
      this.call("reg_cancel").catch(() => {});
      throw e;
    }
    this.reg = { username: st.username, replacing: st.replacing, pose_method: st.pose_method, check_level: checkLevel,
      step: "face", status: "active", face_status: "pending", current_sample: 1, accepted: 0,
      awaiting_record: false, phase: "idle", rejections: 0, restarts: 0, events: [], summary: null,
      error: null, message: null, rec: null, fcap: null, cancelled: false, signal: null };
    return this._regState();
  }

  _requireReg(step = null) {
    const r = this.reg;
    if (!r || this.activity !== "register" || r.status !== "active") throw new ApiError(409, "No registration is in progress.", "no_session");
    if (step !== null && r.step !== step) throw new ApiError(409, `This action belongs to the '${step}' step.`, "wrong_step");
    return r;
  }

  async registerFaceCapture() {
    const r = this._requireReg("face");
    if (r.face_status === "capturing") throw new ApiError(409, "Face capture is already running.", "busy");
    const fcap = this._startFaceCapture();
    Object.assign(r, { face_status: "capturing", fcap, message: null });
    this.camera.warmSface().catch(() => {});
    (async () => {
      await fcap.done;
      r.fcap = null;
      if (r.status !== "active") return;
      if (fcap.error) { Object.assign(r, { face_status: "failed", message: fcap.error }); return; }
      const res = await this.call("reg_face", { embeddings: fcap.embeddings });
      if (!res.ok) { Object.assign(r, { face_status: "failed", message: res.message }); return; }
      Object.assign(r, { face_status: "captured", step: "gesture", faceCount: fcap.embeddings.length });
      this._gestureFlow(r);
    })().catch((e) => Object.assign(r, { face_status: "failed", message: e.message }));
    return this._regState();
  }

  async registerFaceSkip() {
    const r = this._requireReg("face");
    if (r.face_status === "capturing") throw new ApiError(409, "Face capture is running.", "busy");
    Object.assign(r, { face_status: "skipped", step: "gesture" });
    this._gestureFlow(r);
    return this._regState();
  }

  _event(r, type, extra = {}) {
    r.events.push({ ...extra, type, sample: r.current_sample, time: nowSec() });
    r.events = r.events.slice(-40);
  }

  // gesture_capture.collect_gesture_samples, with the events of service._reg_gesture_worker.
  async _gestureFlow(r) {
    const C = this.C;
    try {
      let restarts = 0, sample = 1, lastN = 0;
      while (sample <= C.samples_required) {
        let retries = 0, accepted = false;
        while (!accepted && retries < C.max_retries_per_sample) {
          const res = await this._recordSample(r, sample, lastN);
          lastN = sample;
          if (res === null) return;                                  // cancelled
          if (res.ok) {
            accepted = true;
            r.accepted = res.accepted;
          } else {
            retries += 1;
            r.rejections += 1;
          }
          this._event(r, res.ok ? "accepted" : "rejected",
            { message: res.message, rejections: r.rejections, max_retries: C.max_retries_per_sample });
        }
        if (!accepted) {
          restarts += 1;
          if (restarts > C.max_registration_restarts) {
            this._regFinish(r, "failed", "The recordings stayed inconsistent after " +
              `${C.max_registration_restarts} restarts. Practise the gesture until it feels the same each ` +
              "time, then try again. Any existing profile is unchanged.");
            return;
          }
          await this.call("reg_restart");
          sample = 1;
          continue;
        }
        sample += 1;
      }
      Object.assign(r, { phase: "saving", step: "review" });
      const summary = await this.call("reg_save");
      r.summary = summary;
      this._regFinish(r, "done", null);
    } catch (e) {
      if (r.status === "active") {
        this._regFinish(r, "failed", r.phase === "saving"
          ? `Could not save the registration: ${e.message}. Any existing profile is unchanged.`
          : `Registration failed: ${e.message}`);
      }
    }
  }

  // service._reg_gesture_worker.record_fn + the quality gate of reg_sample.
  async _recordSample(r, n, lastN) {
    if (n < lastN) {
      r.restarts += 1;
      r.accepted = 0;
      this._event(r, "restart", { restart: r.restarts, max_restarts: this.C.max_registration_restarts });
    }
    if (n !== lastN) r.rejections = 0;
    for (;;) {
      Object.assign(r, { current_sample: n, awaiting_record: true, phase: "idle" });
      await new Promise((resolve) => { r.signal = resolve; });
      if (r.cancelled) return null;
      const rec = r.rec;
      await rec.done;
      if (r.cancelled) return null;
      if (rec.error) { this._event(r, "error", { message: `Recording stopped (${rec.error}).` }); continue; }
      r.phase = "checking";
      const res = await this.call("reg_sample", { landmarks: rec.landmarks, frame_aspect: rec.aspect });
      if (res.too_few_frames) {
        this._event(r, "too_few_frames", { frames: res.frames,
          message: `Only ${res.frames} hand frames. Keep your hand in view and record again.` });
        continue;
      }
      return res;
    }
  }

  _regFinish(r, status, message) {
    Object.assign(r, { status, phase: "idle", awaiting_record: false });
    if (message) { r.message = message; r.error = message; }
    if (status === "done") r.step = "review";
    this._release("register");
  }

  async registerRecord() {
    const r = this._requireReg("gesture");
    if (!r.awaiting_record) throw new ApiError(409, "Wait until the current sample has been checked.", "busy");
    const rec = this._startRecording(false);
    Object.assign(r, { rec, awaiting_record: false, phase: "recording", message: null });
    if (r.signal) { const s = r.signal; r.signal = null; s(); }
    return this._regState();
  }

  async registerCancel() {
    const r = this.reg;
    if (!r) return { ok: true };
    r.cancelled = true;
    this.camera.cancelJobs("cancelled");
    if (r.signal) { const s = r.signal; r.signal = null; s(); }
    if (r.status === "active") {
      r.status = "cancelled";
      r.message = "Registration cancelled. Any existing profile is unchanged.";
    }
    this.call("reg_cancel").catch(() => {});
    this._release("register");
    return this._regState();
  }

  async registerClose() {
    const r = this.reg;
    if (r && r.status === "active") await this.registerCancel();
    this.reg = null;
    return { ok: true };
  }

  _regState() {
    const r = this.reg;
    if (!r) return null;
    const C = this.C || {};
    return {
      username: r.username, replacing: r.replacing, pose_method: r.pose_method, step: r.step, status: r.status,
      face_status: r.face_status,
      face_progress: r.fcap ? r.fcap.embeddings.length : (r.face_status === "captured" ? C.face_required_frames : 0),
      face_required: C.face_required_frames,
      samples_required: C.samples_required,
      current_sample: r.current_sample, accepted: r.accepted,
      awaiting_record: r.awaiting_record, phase: r.phase,
      rejections: r.rejections, max_retries: C.max_retries_per_sample,
      restarts: r.restarts, max_restarts: C.max_registration_restarts,
      events: r.events.slice(-12),
      recording: r.phase === "recording" && r.rec ? { progress: Math.round(r.rec.progress() * 1000) / 1000,
        hand_frames: r.rec.landmarks.length, duration: r.rec.duration } : null,
      summary: r.summary, message: r.message,
    };
  }

  // ------------------------------------------------------------ face enrolment (Users page)
  async faceEnroll(rawUsername) {
    await this.boot();
    const { username } = await this.call("face_enroll_check", { raw_username: rawUsername });
    this._claim("face");
    let fcap;
    try { fcap = this._startFaceCapture(); } catch (e) { this._release("face"); throw e; }
    const f = { username, status: "capturing", fcap, message: null };
    this.face = f;
    this.camera.warmSface().catch(() => {});
    (async () => {
      await fcap.done;
      let status, message;
      try {
        if (fcap.error) { status = "failed"; message = fcap.error; }
        else ({ status, message } = await this.call("face_enroll_save", { username, embeddings: fcap.embeddings }));
      } catch (e) {
        status = "failed"; message = `Could not save the face profile: ${e.message}`;
      }
      Object.assign(f, { status, message, fcap: null });
      this._release("face");
    })();
    return this._faceState();
  }

  async faceClose() {
    const f = this.face;
    if (f && f.status === "capturing") this.camera.cancelJobs("cancelled");
    this.face = null;
    this._release("face");
    return { ok: true };
  }

  _faceState() {
    const f = this.face;
    if (!f) return null;
    const req = this.C ? this.C.face_required_frames : 5;
    return { username: f.username, status: f.status, message: f.message,
      progress: f.fcap ? f.fcap.embeddings.length : (f.status === "done" ? req : 0), required: req };
  }

  // ------------------------------------------------------------ settings (mobile)
  async setOptions({ debug, liveness, facing } = {}) {
    const patch = {};
    if (debug !== undefined) patch.debug = Boolean(debug);
    if (liveness !== undefined) patch.liveness = Boolean(liveness);
    if (facing !== undefined) patch.facing = facing === "environment" ? "environment" : "user";
    this.saveSettings(patch);
    await this.boot();
    await this.call("set_options", { debug: this.settings.debug, liveness: this.settings.liveness });
    return this.settings;
  }
}
