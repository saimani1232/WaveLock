// WaveLock mobile: the camera panel of the web UI (web/js/camera.js, copied as
// camera_web.js by build.py), showing the device camera instead of JPEG
// frames from web_app.py. Overlays, coaching hints and status chips are the
// web panel's; only how frames arrive differs.

import { CameraPanel as WebCameraPanel } from "./camera_web.js";
import { service } from "./api.js";

export class CameraPanel extends WebCameraPanel {
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
  }
}
