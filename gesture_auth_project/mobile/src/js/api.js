// WaveLock mobile: the same API as web/js/api.js, answered on the device by
// the local service (js/local/service.js) instead of web_app.py. The pages of
// the web UI (dashboard, sign-in, register, users) are used unchanged.

import { LocalService, ApiError } from "./local/service.js";

export { ApiError };
export const service = new LocalService();

// Start-up screen while Python, numpy / scipy and the models load (first
// visit: tens of MB, then from the browser cache).
const bootText = document.getElementById("boot-text");
if (bootText) service.progressListeners.add((t) => { bootText.textContent = t; });

function guard(fn) {
  return async (...args) => {
    try {
      return await fn(...args);
    } catch (e) {
      if (e instanceof ApiError) throw e;
      throw new ApiError(500, (e && e.message) || String(e), null);
    }
  };
}

export const api = {
  system: guard(async () => {
    try {
      return await service.system();
    } finally {
      const boot = document.getElementById("boot");
      if (boot) boot.hidden = true;
    }
  }),
  users: guard(() => service.users()),
  activity: guard((limit = 50, user = null) => service.activityLog(limit, user)),
  status: guard(() => service.status()),
  cameras: guard(async () => [
    { index: "front", label: "Front camera (selfie)" },
    { index: "back", label: "Back camera" },
  ]),
  cameraStart: guard((cameraIndex = null) => service.cameraStart(cameraIndex)),
  cameraStop: guard(() => service.cameraStop()),
  authBegin: guard((username) => service.authBegin(username)),
  authRecord: guard(() => service.authRecord()),
  authReset: guard(() => service.authReset()),
  authEnd: guard(() => service.authEnd()),
  registerBegin: guard((username, overwrite, poseMethod, checkLevel = "standard") =>
    service.registerBegin(username, overwrite, poseMethod, checkLevel)),
  registerFaceCapture: guard(() => service.registerFaceCapture()),
  registerFaceSkip: guard(() => service.registerFaceSkip()),
  registerRecord: guard(() => service.registerRecord()),
  registerCancel: guard(() => service.registerCancel()),
  registerClose: guard(() => service.registerClose()),
  faceEnroll: guard((username) => service.faceEnroll(username)),
  faceClose: guard(() => service.faceClose()),
  benchmarkRun: guard(async () => {
    throw new ApiError(403, "The SIMULATED benchmarks run on the desktop (web_app.py --debug), not on the phone.", "debug_only");
  }),
  benchmarkStatus: guard(async () => null),
  setOptions: guard((opts) => service.setOptions(opts)),
};
