// Live camera panel.
//
// Frames are pulled one at a time from /api/frame (a stream that ends while
// the camera warms up or restarts would leave the picture black for good;
// short long-poll requests recover on their own).
//
// Layout: the picture with its overlays (scan ring, corners, hand-placement
// guide, coaching hint, countdown, result banner, start-up / error card) and,
// under it, a strip of status chips, so nothing covers the face or the hand.

import { h, icon, clear, fill, add } from "./ui.js";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const DARK_BRIGHTNESS = 40;      // mean grey level below which we suggest more light
const COACH_DELAY_MS = 1500;     // how long hand / face may be missing before a hint

// Open-hand outline (21-point MediaPipe layout) used as a placement guide.
const GUIDE_SVG =
  '<svg viewBox="0 0 120 150" aria-hidden="true"><g fill="none" stroke="currentColor" stroke-width="2.4" ' +
  'stroke-linecap="round" stroke-linejoin="round" stroke-dasharray="1 5">' +
  '<path d="M60 140 L36 116 L26 92 L20 72"/><path d="M60 140 L44 80 L40 52 L38 30"/>' +
  '<path d="M60 140 L60 76 L60 44 L60 18"/><path d="M60 140 L75 80 L79 52 L81 32"/>' +
  '<path d="M60 140 L88 88 L94 68 L98 52"/><path d="M44 80 L60 76 L75 80 L88 88"/></g></svg>';

export class CameraPanel {
  constructor({ showFace = true, label = "Camera preview" } = {}) {
    this.showFace = showFace;
    this.img = h("img", { class: "feed", alt: label, hidden: true });

    this.ring = h("div", { class: "scan-ring", "aria-hidden": "true" });
    this.ring.innerHTML =
      '<svg viewBox="0 0 100 75" preserveAspectRatio="none">' +
      '<rect class="ring-track" x="1.2" y="1.2" width="97.6" height="72.6" rx="4.2" pathLength="100"/>' +
      '<rect class="ring-bar" x="1.2" y="1.2" width="97.6" height="72.6" rx="4.2" pathLength="100" ' +
      'stroke-dasharray="100" stroke-dashoffset="100"/></svg>';
    this.ringBar = this.ring.querySelector(".ring-bar");

    this.guide = h("div", { class: "hand-guide", hidden: true });
    this.guide.innerHTML = GUIDE_SVG;
    this.countdown = h("div", { class: "cam-countdown", hidden: true, role: "timer", "aria-live": "off" });
    this.coachEl = h("div", { class: "cam-coach", "aria-live": "polite" });
    this.center = h("div", { class: "cam-center" });
    this.result = h("div", { class: "cam-result", hidden: true, role: "status" });
    // Start-up / error card, cleared only by real frames.
    this.boot = h("div", { class: "cam-boot" });

    this.picture = h("div", { class: "camera", role: "img", "aria-label": label },
      this.img,
      h("div", { class: "cam-grid" }),
      ["tl", "tr", "bl", "br"].map((c) => h("span", { class: `corner ${c}` })),
      this.ring, this.guide, this.center, this.countdown, this.coachEl, this.result, this.boot);

    this.chipCam = h("span", { class: "cam-chip" });
    this.chipHand = h("span", { class: "cam-chip" });
    this.chipFace = h("span", { class: "cam-chip" });
    this.chipLight = h("span", { class: "cam-chip warn", hidden: true }, icon("sun"), "Low light");
    this.status = h("div", { class: "cam-status", "aria-label": "Camera status" },
      this.chipCam, this.chipHand, showFace ? this.chipFace : null, this.chipLight);

    this.el = h("div", { class: "cam-panel" }, this.picture, this.status);

    this.mode = null;
    this.token = 0;            // increments on stop(): ends any running frame loop
    this.hasFrame = false;
    this.url = null;
    this.app = null;
    this.coach = null;         // {needHand, needFace, text} set by the view
    this.lastHand = 0;
    this.lastFace = 0;
    this.camera = null;
    this.setMessage("camera", "Starting camera…", { spinner: true });
  }

  // ---------------------------------------------------------------- lifecycle
  async start(app) {
    this.app = app;
    const token = ++this.token;
    this.hasFrame = false;
    this.img.hidden = true;
    this.setMessage("camera", "Starting camera…", { spinner: true });
    let cam;
    try {
      cam = await app.ensureCamera();
    } catch (e) {
      if (token === this.token) this.showError(e.message);
      return false;
    }
    if (token !== this.token) return false;          // stopped while starting
    if (cam.state === "error") {
      this.showError(cam.error || "The camera is not available.");
    } else if (cam.state !== "running") {
      this.setMessage("camera", "The camera is still starting…", { spinner: true });
    }
    this._loop(token);                                // also recovers later
    return true;
  }

  stop() {
    this.token += 1;
    this.img.removeAttribute("src");
    this.img.hidden = true;
    if (this.url) { URL.revokeObjectURL(this.url); this.url = null; }
  }

  async _loop(token) {
    let after = -1, failures = 0;
    while (token === this.token) {
      let res;
      try {
        res = await fetch(`/api/frame?after=${after}`, { cache: "no-store", credentials: "same-origin" });
      } catch (e) {
        failures += 1;                                // server restarting / offline
        if (token !== this.token) return;
        if (failures === 3) this.showError("Lost connection to the WaveLock server. Is web_app.py still running?");
        await sleep(Math.min(2000, 250 * failures));
        continue;
      }
      if (token !== this.token) return;
      if (res.status === 200) {
        const id = Number(res.headers.get("X-Frame-Id"));
        const blob = await res.blob();
        if (token !== this.token) return;
        const prev = this.url;
        this.url = URL.createObjectURL(blob);
        this.img.src = this.url;
        if (prev) setTimeout(() => URL.revokeObjectURL(prev), 1000);
        after = Number.isFinite(id) ? id : after;
        failures = 0;
        if (!this.hasFrame) {
          this.hasFrame = true;
          this.img.hidden = false;
          this.boot.hidden = true;
          this._paintCoach();
        }
      } else if (res.status === 204) {
        // No new frame for ~1 s: say why, keep polling (it resumes by itself).
        const state = res.headers.get("X-Camera-State");
        if (state === "stopped" || state === "error") {
          this.hasFrame = false;
          this.img.hidden = true;
          if (state === "stopped") this.setMessage("cameraOff", "The camera is off.", { retry: true });
          await sleep(800);
        }
      } else {
        failures += 1;
        await sleep(Math.min(2000, 250 * failures));
      }
    }
  }

  // ---------------------------------------------------------------- status
  get live() {
    return this.hasFrame && this.camera !== null && this.camera.state === "running";
  }

  update(camera) {
    if (!camera) return;
    this.camera = camera;
    const now = Date.now();
    if (camera.hand_visible) this.lastHand = now;
    if (camera.face_in_view) this.lastFace = now;
    const running = camera.state === "running";
    this.chipCam.className = `cam-chip ${running ? "on" : camera.state === "error" ? "off" : ""}`;
    fill(this.chipCam, h("span", { class: `live-dot ${running ? "on" : ""}` }),
      running ? `Live · ${Math.round(camera.fps || 0)} fps` : camera.state === "starting" ? "Starting" : "Camera off");
    this.chipHand.className = `cam-chip ${camera.hand_visible ? "on" : ""}`;
    fill(this.chipHand, icon("hand"), camera.hand_visible ? "Hand detected" : "No hand");
    if (this.showFace) {
      this.chipFace.className = `cam-chip ${camera.face_in_view ? "on" : ""}`;
      fill(this.chipFace, icon("face"), camera.face_in_view ? "Face in view" : "No face");
    }
    this.chipLight.hidden = !(running && camera.brightness !== null && camera.brightness !== undefined
      && camera.brightness < DARK_BRIGHTNESS);
    if (camera.state === "error") {
      this.hasFrame = false;
      this.img.hidden = true;
      this.showError(camera.error || "Camera error.");
    }
    this._paintCoach();
  }

  // The view says what the current step needs; the panel turns the live
  // hand / face / light status into one short hint.
  setCoach(coach) {
    this.coach = coach;
    this.lastHand = this.lastFace = Date.now();     // grace period after a change
    this._paintCoach();
  }

  _paintCoach() {
    const c = this.coach, cam = this.camera;
    let text = null, kind = "";
    if (c && this.hasFrame && cam && cam.state === "running") {
      const now = Date.now();
      const noHand = c.needHand && !cam.hand_visible && now - this.lastHand > COACH_DELAY_MS;
      const noFace = c.needFace && !cam.face_in_view && now - this.lastFace > COACH_DELAY_MS;
      const dark = cam.brightness !== null && cam.brightness !== undefined && cam.brightness < DARK_BRIGHTNESS;
      if (this.mode === "recording" && c.needHand && !cam.hand_visible) {
        text = "Keep your hand inside the frame"; kind = "warn";
      } else if (dark) {
        text = "It is quite dark: add light in front of you"; kind = "warn";
      } else if (noFace) {
        text = "Face the camera so your face is visible"; kind = "warn";
      } else if (noHand) {
        text = "Raise your open hand into the frame"; kind = "warn";
      } else if (this.mode !== "recording") {
        text = c.text || null;
      }
      this.guide.hidden = !(c.needHand && !cam.hand_visible && this.mode !== "recording"
        && this.mode !== "processing" && this.mode !== "granted" && this.mode !== "denied");
    } else {
      this.guide.hidden = true;
    }
    if (["processing", "granted", "denied", "challenge"].includes(this.mode)) text = null;
    if (text === this._coachText && kind === this._coachKind) return;
    this._coachText = text; this._coachKind = kind;
    this.coachEl.className = `cam-coach ${kind}`;
    if (text) fill(this.coachEl, icon(kind === "warn" ? "alert" : "info"), text);
    else clear(this.coachEl);
  }

  // ---------------------------------------------------------------- overlays
  setMessage(iconName, text, { error = false, spinner = false, retry = false } = {}) {
    this.boot.hidden = false;
    const actions = (retry || error) && this.app ? h("div", { class: "cam-boot-actions" },
      h("button", { class: "btn sm primary", type: "button", onclick: () => this.start(this.app) },
        icon("refresh"), "Retry"),
      h("a", { class: "btn sm", href: "#/settings" }, icon("settings"), "Choose camera")) : null;
    fill(this.boot, h("div", { class: `cam-message ${error ? "error" : ""}` },
      spinner ? h("div", { class: "spinner" }) : icon(iconName),
      h("div", null, text), actions));
  }

  showError(text) {
    this.setMessage("cameraOff", text, { error: true });
  }

  setMode(mode, data = {}) {
    this.mode = mode;
    this.picture.classList.remove("recording", "granted", "denied", "processing", "challenge");
    if (mode !== "idle") this.picture.classList.add(mode);
    clear(this.center);
    this.countdown.hidden = mode !== "recording";
    this.result.hidden = !(mode === "granted" || mode === "denied");
    this.ringBar.style.transition = "none";
    this.ringBar.setAttribute("stroke-dashoffset", mode === "granted" || mode === "denied" ? "0" : "100");
    void this.ringBar.getBoundingClientRect();        // apply before re-enabling the transition
    this.ringBar.style.transition = "";
    if (mode === "challenge" && data.ready != null) {
      // Challenge passed: the user starts the recording when ready (as at registration).
      add(this.center, h("div", { class: "cam-challenge" },
        h("div", { class: "ch-big" }, icon("check"), icon("wave")),
        h("strong", null, "Passed: start the recording when you are ready"),
        h("small", null, `Then perform your gesture as during registration · ${Math.ceil(data.ready)} s`)));
    } else if (mode === "challenge" && data.step) {
      // Liveness instruction, large and centred: a random request a photo or
      // a recorded video cannot follow.
      const st = data.step;
      const big = st.kind === "fingers"
        ? h("div", { class: "ch-big" }, h("b", null, String(st.count)), icon("hand"))
        : h("div", { class: "ch-big" }, icon(st.direction === "left" ? "arrowLeft" : "arrowRight"));
      const text = st.kind === "fingers"
        ? `Show ${st.count} finger${st.count === 1 ? "" : "s"}`
        : `Turn your head ${st.direction === "left" ? "to the left" : "to the right"}, then back`;
      add(this.center, h("div", { class: "cam-challenge" }, big, h("strong", null, text),
        h("small", null, `Step ${data.index + 1} of ${data.total} · ${Math.ceil(data.remaining || 0)} s`)));
    } else if (mode === "recording") {
      this.setProgress({ progress: 0, duration: data.duration || 3, hand_frames: 0 });
    } else if (mode === "processing") {
      add(this.center, h("div", { class: "cam-message" }, h("div", { class: "spinner" }),
        h("div", null, data.text || "Verifying…")));
    } else if (mode === "granted" || mode === "denied") {
      const ok = mode === "granted";
      fill(this.result, h("span", { class: "r-icon" }, icon(ok ? "unlock" : "lock")),
        h("span", null, h("strong", null, data.title || (ok ? "Verified" : "Not verified")),
          data.sub ? h("small", null, data.sub) : null));
    }
    if (mode === "idle" && data.hint !== undefined) {
      this.coach = Object.assign({}, this.coach || {}, { text: data.hint });
    }
    this._paintCoach();
  }

  setProgress(rec) {
    if (this.mode !== "recording") return;
    const p = Math.max(0, Math.min(1, rec.progress || 0));
    this.ringBar.setAttribute("stroke-dashoffset", String(100 * (1 - p)));
    const left = Math.max(0, (rec.duration || 3) * (1 - p));
    fill(this.countdown, h("span", { class: "rec-dot" }), `REC ${left.toFixed(1)} s`,
      h("small", null, `${rec.hand_frames || 0} hand frames`));
  }
}
