// WaveLock mobile: the camera panel of the web UI (web/js/camera.js, copied as
// camera_web.js by build.py), showing the device camera instead of JPEG
// frames from web_app.py. Overlays, coaching hints and status chips are the
// web panel's; only how frames arrive differs.
//
// On a phone the page's action buttons sit far below the camera, so the
// panel adds an action bar right under the picture that MIRRORS the page's
// main button and the one next to it (same label, same enabled state; a tap
// presses the real button). The shared pages stay unchanged.

import { CameraPanel as WebCameraPanel } from "./camera_web.js";
import { service } from "./api.js";
import { h } from "./ui.js";

export class CameraPanel extends WebCameraPanel {
  constructor(opts) {
    super(opts);
    this.actions = h("div", { class: "cam-actions", hidden: true, "aria-label": "Actions" });
    this.el.append(this.actions);
    this.actionKey = "";
    this.observer = null;
    this.mirrorQueued = false;
  }

  async start(app) {
    this.app = app;
    const token = ++this.token;
    this.hasFrame = false;
    this.img.hidden = true;
    this.setMessage("camera", "Starting camera…", { spinner: true });
    this._watchActions();
    let cam;
    try {
      cam = await app.ensureCamera();
    } catch (e) {
      if (token === this.token) this.showError(e.message);
      return false;
    }
    if (token !== this.token) return false;
    if (cam.state !== "running") {
      this.showError(cam.error || "The camera is not available.");
      return false;
    }
    // The live picture is the camera service's canvas (mirrored, with the
    // hand skeleton and face box drawn on it).
    const canvas = service.camera.display;
    if (this.img !== canvas) {
      this.img.replaceWith(canvas);
      this.img = canvas;
    }
    this.picture.style.setProperty("aspect-ratio", `${canvas.width} / ${canvas.height}`);
    canvas.hidden = false;
    this.hasFrame = true;
    this.boot.hidden = true;
    this.update(service.camera.status());
    this._paintCoach();
    return true;
  }

  stop() {
    this.token += 1;
    this.hasFrame = false;
    if (this.img) this.img.hidden = true;
    if (this.observer) { this.observer.disconnect(); this.observer = null; }
    for (const b of document.querySelectorAll(".wl-mirrored")) b.classList.remove("wl-mirrored");
  }

  // ---------------------------------------------------------------- action bar
  _root() {
    return this.el.closest(".modal") || document.getElementById("view");
  }

  _watchActions() {
    if (this.observer) return;
    this.observer = new MutationObserver(() => {
      if (!document.contains(this.el)) { this.observer.disconnect(); this.observer = null; return; }
      if (this.mirrorQueued) return;
      this.mirrorQueued = true;
      requestAnimationFrame(() => { this.mirrorQueued = false; this._mirror(); });
    });
    // the panel is attached to the page right after start(); observe from then on
    requestAnimationFrame(() => {
      const root = this._root();
      if (root && this.observer) {
        this.observer.observe(root, { subtree: true, childList: true, attributes: true,
          attributeFilter: ["disabled", "hidden", "class"], characterData: true });
        this._mirror();
      }
    });
  }

  // The page's main button (outside this panel) and the first other button next to it.
  _sources() {
    const root = this._root();
    if (!root) return [];
    // a source hidden only because the bar mirrors it (.wl-mirrored) still counts
    const shown = (b) => !b.hidden && !b.closest("[hidden]") && (b.offsetParent !== null || b.classList.contains("wl-mirrored"));
    const visible = (b) => !this.el.contains(b) && shown(b);
    const primary = [...root.querySelectorAll("button.btn.primary")].find(visible);
    if (!primary) return [];
    const sibling = [...primary.parentElement.children].find((b) => b !== primary && b.tagName === "BUTTON"
      && b.classList.contains("btn") && shown(b));
    return sibling ? [primary, sibling] : [primary];
  }

  _mirror() {
    const sources = this._sources();
    const label = (b) => [...b.childNodes].filter((n) => !(n.classList && n.classList.contains("kbd")))
      .map((n) => n.textContent).join("").trim();
    // on a phone the originals are hidden while mirrored (mobile.css), so the
    // action is not shown twice; the bar still presses them
    for (const old of document.querySelectorAll(".wl-mirrored")) if (!sources.includes(old)) old.classList.remove("wl-mirrored");
    for (const src of sources) src.classList.add("wl-mirrored");
    // Pages re-render their buttons (e.g. every second of a countdown): the
    // bar is rebuilt only when what it shows changes, so a tap is never lost
    // on a replaced button, and a tap presses the page's CURRENT button.
    const key = sources.map((b) => `${label(b)}|${b.disabled}`).join("||");
    if (key === this.actionKey) return;
    this.actionKey = key;
    this.actions.replaceChildren(...sources.map((src, i) => {
      const btn = src.cloneNode(true);
      for (const k of btn.querySelectorAll(".kbd, .kbd-hint")) k.remove();
      btn.removeAttribute("id");
      btn.removeAttribute("aria-keyshortcuts");
      btn.className = i === 0 ? "btn primary lg" : "btn lg";
      btn.type = "button";
      btn.disabled = src.disabled;
      btn.addEventListener("click", () => {
        const current = this._sources()[i];
        if (current && !current.disabled) current.click();
      });
      return btn;
    }));
    this.actions.hidden = sources.length === 0;
  }
}
