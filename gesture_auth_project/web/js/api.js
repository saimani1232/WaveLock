// Thin client for the WaveLock backend (web_backend/server.py).
// Every POST carries the custom header the server requires.

export class ApiError extends Error {
  constructor(status, message, code) {
    super(message);
    this.status = status;
    this.code = code || null;
  }
}

async function call(method, path, body) {
  const opts = { method, headers: {}, cache: "no-store", credentials: "same-origin" };
  if (method !== "GET") {
    opts.headers["Content-Type"] = "application/json";
    opts.headers["X-WaveLock-Client"] = "1";
    opts.body = JSON.stringify(body || {});
  }
  let res;
  try {
    res = await fetch(path, opts);
  } catch (e) {
    throw new ApiError(0, "The WaveLock server is not reachable. Is web_app.py still running?", "offline");
  }
  let payload = null;
  try { payload = await res.json(); } catch (e) { /* non-JSON */ }
  if (!res.ok) {
    throw new ApiError(res.status, (payload && payload.error) || `Request failed (${res.status}).`,
      payload && payload.code);
  }
  return payload ? payload.data : null;
}

export const api = {
  system: () => call("GET", "/api/system"),
  users: () => call("GET", "/api/users"),
  activity: (limit = 50, user = null) =>
    call("GET", `/api/activity?limit=${limit}` + (user ? `&user=${encodeURIComponent(user)}` : "")),
  status: () => call("GET", "/api/status"),
  cameras: () => call("GET", "/api/cameras"),
  cameraStart: (cameraIndex = null) => call("POST", "/api/camera/start", { camera_index: cameraIndex }),
  cameraStop: () => call("POST", "/api/camera/stop"),
  authBegin: (username) => call("POST", "/api/auth/begin", { username }),
  authRecord: () => call("POST", "/api/auth/record"),
  authReset: () => call("POST", "/api/auth/reset"),
  authEnd: () => call("POST", "/api/auth/end"),
  registerBegin: (username, overwrite, poseMethod, checkLevel = "standard") =>
    call("POST", "/api/register/begin", { username, overwrite, pose_method: poseMethod, check_level: checkLevel }),
  registerFaceCapture: () => call("POST", "/api/register/face/capture"),
  registerFaceSkip: () => call("POST", "/api/register/face/skip"),
  registerRecord: () => call("POST", "/api/register/record"),
  registerCancel: () => call("POST", "/api/register/cancel"),
  registerClose: () => call("POST", "/api/register/close"),
  faceEnroll: (username) => call("POST", "/api/face/enroll", { username }),
  faceClose: () => call("POST", "/api/face/close"),
  benchmarkRun: (username, kind) => call("POST", "/api/benchmark/run", { username, kind }),
  benchmarkStatus: () => call("GET", "/api/benchmark/status"),
};
