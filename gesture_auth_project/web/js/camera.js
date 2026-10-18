// Live camera panel: frames pulled one at a time from /api/frame plus
// overlays (scanner corners, hand / face chips, recording ring, results).
//
// Why not one endless MJPEG <img> stream: a stream that ends (camera still
// warming up, camera restarted from Settings) leaves the picture black for
// good, and every forgotten stream holds one of the browser's few
// connections. Short long-poll requests recover on their own.

import { h, icon, clear, fill, add } from "./ui.js";

const RING_R = 52;
const RING_C = 2 * Math.PI * RING_R;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export class CameraPanel {
  constructor({ showFace = true, label = "Camera preview" } = {}) {
    this.showFace = showFace;
    this.img = h("img", { class: "feed", alt: label });
    this.chipCam = h("span", { class: "cam-chip" });
    this.chipHand = h("span", { class: "cam-chip" });
    this.chipFace = h("span", { class: "cam-chip" });
    this.chipRec = h("span", { class: "cam-chip rec", hidden: true }, "REC");
    this.center = h("div", { class: "cam-center" });
    this.bottom = h("div", { class: "cam-bottom" });
    // Startup / error overlay: independent of the view's modes, cleared only
    // by real frames, so a view can never hide "Starting camera..." early.
    this.boot = h("div", { class: "cam-center cam-boot" });
    this.el = h("div", { class: "camera", role: "img", "aria-label": label },
      this.img,
      h("div", { class: "cam-grid" }),
      h("div", { class: "scanline" }),
      ["tl", "tr", "bl", "br"].map((c) => h("span", { class: `corner ${c}` })),
      h("div", { class: "cam-chips" }, this.chipCam, this.chipHand,
        showFace ? this.chipFace : null, this.chipRec),
      this.center, this.bottom, this.boot);
    this.mode = null;
    this.token = 0;            // increments on stop(): ends any running frame loop
    this.hasFrame = false;
    this.url = null;
    this.setMessage("camera", "Starting camera…", false, true);
  }

  async start(app) {
    const token = ++this.token;
    this.hasFrame = false;
    this.setMessage("camera", "Starting camera…", false, true);
    let cam;
    try {
      cam = await app.ensureCamera();
    } catch (e) {
      if (token === this.token) this.setMessage("cameraOff", e.message, true);
      return false;
    }
    if (token !== this.token) return false;          // stopped while starting
    if (cam.state === "error") {
      this.setMessage("cameraOff", cam.error || "The camera is not available.", true);
    } else if (cam.state !== "running") {
      this.setMessage("camera", "The camera is still starting…", false, true);
    }
    this._loop(token);                                // also recovers later
    return true;
  }

  stop() {
    this.token += 1;
    this.img.removeAttribute("src");
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
        if (failures === 3) this.setMessage("cameraOff", "Lost connection to the WaveLock server…", true);
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
        if (!this.hasFrame) { this.hasFrame = true; this.boot.hidden = true; }
      } else if (res.status === 204) {
        // No new frame for ~1 s: say why, keep polling (it resumes by itself).
        const state = res.headers.get("X-Camera-State");
        if (state === "stopped" || state === "error") {
          this.hasFrame = false;
          if (state === "stopped") this.setMessage("cameraOff", "The camera is off.", false);
          await sleep(800);
        }
      } else {
        failures += 1;
        await sleep(Math.min(2000, 250 * failures));
      }
    }
  }

  // camera = status.camera
  update(camera) {
    if (!camera) return;
    const live = camera.state === "running";
    this.chipCam.className = `cam-chip ${live ? "on" : "off"}`;
    fill(this.chipCam, icon(live ? "camera" : "cameraOff"),
      live ? `Live · ${Math.round(camera.fps || 0)} fps` : camera.state === "starting" ? "Starting" : "Camera off");
    this.chipHand.className = `cam-chip ${camera.hand_visible ? "on" : ""}`;
    fill(this.chipHand, icon("hand"), camera.hand_visible ? "Hand detected" : "No hand");
    if (this.showFace) {
      this.chipFace.className = `cam-chip ${camera.face_in_view ? "on" : ""}`;
      fill(this.chipFace, icon("face"), camera.face_in_view ? "Face in view" : "No face");
    }
    if (camera.state === "error") {
      this.hasFrame = false;
      this.setMessage("cameraOff", camera.error || "Camera error.", true);
    }
  }

  setMessage(iconName, text, isError = false, spinner = false) {
    this.boot.hidden = false;
    this.boot.className = "cam-center cam-boot";
    fill(this.boot, h("div", { class: `cam-message ${isError ? "error" : ""}` },
      spinner ? h("div", { class: "spinner" }) : icon(iconName), h("div", null, text)));
  }

  setMode(mode, data = {}) {
    this.mode = mode;
    this.el.classList.remove("recording", "granted", "denied");
    this.chipRec.hidden = mode !== "recording";
    clear(this.center);
    clear(this.bottom);
    if (mode === "recording") {
      this.el.classList.add("recording");
      // Corner placement keeps the hand and face visible while recording.
      add(this.bottom, h("span", { class: "spacer" }), this._ring(data));
    } else if (mode === "processing") {
      add(this.center, h("div", { class: "cam-message" }, h("div", { class: "spinner" }),
        h("div", null, data.text || "Verifying…")));
    } else if (mode === "granted" || mode === "denied") {
      this.el.classList.add(mode);
      add(this.center, h("div", { class: `result-burst ${mode === "granted" ? "ok" : "danger"}` },
        icon(mode === "granted" ? "check" : "x")));
    } else if (mode === "idle" && data.hint) {
      add(this.bottom, h("span", { class: "cam-chip" }, icon("info"), data.hint));
    }
  }

  setProgress(rec) {
    if (this.mode !== "recording" || !this.ringBar) return;
    const p = Math.max(0, Math.min(1, rec.progress || 0));
    this.ringBar.setAttribute("stroke-dashoffset", String(RING_C * (1 - p)));
    const left = Math.max(0, (rec.duration || 3) * (1 - p));
    this.ringValue.textContent = left.toFixed(1);
    this.ringSub.textContent = `${rec.hand_frames || 0} hand frames`;
  }

  _ring(data) {
    const wrap = h("div", { class: "ring", role: "progressbar", "aria-label": "Recording progress" });
    wrap.innerHTML =
      `<svg viewBox="0 0 120 120"><circle class="track" cx="60" cy="60" r="${RING_R}" fill="none" stroke-width="8"/>` +
      `<circle class="bar" cx="60" cy="60" r="${RING_R}" fill="none" stroke-width="8" ` +
      `stroke-dasharray="${RING_C}" stroke-dashoffset="${RING_C}"/></svg>`;
    this.ringBar = wrap.querySelector(".bar");
    this.ringValue = h("span", null, (data.duration || 3).toFixed(1));
    this.ringSub = h("small", null, "recording");
    add(wrap, h("div", { class: "ring-label" }, h("div", null, this.ringValue, this.ringSub)));
    return wrap;
  }
}
