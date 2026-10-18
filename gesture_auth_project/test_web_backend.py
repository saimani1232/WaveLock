"""
Web UI backend tests — run with:  python test_web_backend.py [name ...]

No webcam needed: a SIMULATED camera replays synthetic landmark sequences
(the same generator as test_wavelock.py) through the real HTTP server,
service and decision code. The simulated parts live only in this file.
"""

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

import cv2
import numpy as np

import test_wavelock as tw
import gesture_auth
import web_backend.service as service_mod
from utils.face_auth import classify_face_score
from web_backend.camera import CameraService, CameraError
from web_backend.service import WaveLockService
from web_backend.server import WaveLockServer

FAST_RECORDING_SEC = 1.0
SIM_FPS = 90
JPEG_SOI = bytes([0xFF, 0xD8])   # JPEG start-of-image marker
SIM_WARMUP_SEC = 1.0      # a real camera's first frame is slow (MediaPipe / YuNet warm-up)


# ------------------------------------------------------------------ simulated camera

class SimSource:
    """Dark frames labelled SIMULATED CAMERA, at SIM_FPS."""

    def __init__(self, index, face_ops):
        if index == 9:
            raise CameraError("Could not open camera 9 (simulated failure).")
        self.face_ops = face_ops

    def read(self):
        time.sleep(1.0 / SIM_FPS)
        frame = np.full((480, 640, 3), (34, 26, 20), np.uint8)
        cv2.putText(frame, "SIMULATED CAMERA", (16, 464), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (110, 120, 130), 1, cv2.LINE_AA)
        if self.face_ops.face_present:
            cv2.ellipse(frame, (325, 120), (52, 66), 0, 0, 360, (90, 110, 130), -1)
            cv2.ellipse(frame, (325, 120), (52, 66), 0, 0, 360, (140, 160, 180), 2)
        # The camera service mirrors every frame (like the CLI); pre-mirror so
        # the label and face position appear as drawn.
        return True, cv2.flip(frame, 1)

    def release(self):
        pass


class SimTracker:
    """Returns scripted raw landmarks: the queued gesture while a recording
    runs, otherwise a still hand (or none when hand_visible is False)."""

    CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                   (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                   (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]

    def __init__(self):
        self.camera = None
        self.queue = []
        self.pos = 0
        self.hand_visible = True
        self.hide_during_recording = False
        self.idle = tw.synth_raw(seed=1)[0]
        self.warm = False

    def play(self, raw):
        self.queue = list(raw)
        self.pos = 0
        self.idle = raw[0]

    def process(self, frame):
        if not self.warm:                 # first frame of a run is slow, like MediaPipe
            time.sleep(SIM_WARMUP_SEC)
            self.warm = True
        recording = self.camera is not None and self.camera._recording is not None
        if recording and self.hide_during_recording:
            return None
        if not self.hand_visible and not recording:
            return None
        if recording and self.queue:
            # Follow recording time, not frame count: the whole gesture spans
            # the recording window whatever the simulated frame rate is.
            rec = self.camera._recording
            progress = rec.progress() if rec is not None else 1.0
            lm = self.queue[min(len(self.queue) - 1, int(progress * (len(self.queue) - 1)))]
        else:
            lm = self.idle
        h, w = frame.shape[:2]
        pts = [(int(x * w), int(y * h)) for x, y, _ in lm]
        for a, b in self.CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (200, 200, 200), 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, p, 4, (60, 60, 230), -1, cv2.LINE_AA)
        return np.asarray(lm, dtype=np.float64)

    def close(self):
        pass


class SimFaceOps:
    def __init__(self):
        self.face_present = True
        rng = np.random.RandomState(11)
        self.identity = rng.randn(128).astype(np.float32)
        self.identity /= np.linalg.norm(self.identity)

    def detect(self, frame):
        if not self.face_present:
            return None, None
        return np.zeros(15, np.float32), (270, 50, 110, 140)

    def embed(self, frame, face_row):
        return self.identity + np.random.normal(0, 0.01, 128).astype(np.float32)

    def verify(self, frame, embedding):
        score = float(np.dot(self.identity, embedding) / (np.linalg.norm(embedding) + 1e-9))
        match, conf, reason = classify_face_score(score)
        return match, score, conf, (270, 50, 110, 140), reason


def make_app(templates_dir, debug=False):
    """Start a real WaveLockServer on a free port with the simulated camera."""
    service_mod.RECORDING_DURATION_SEC = FAST_RECORDING_SEC
    face_ops = SimFaceOps()
    tracker = SimTracker()
    def tracker_factory():
        tracker.warm = False              # every camera start warms up again
        return tracker

    camera = CameraService(source_factory=lambda idx: SimSource(idx, face_ops),
                           tracker_factory=tracker_factory, face_ops=face_ops)
    tracker.camera = camera
    service = WaveLockService(camera, templates_dir=templates_dir, debug=debug)
    server = WaveLockServer(service, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1},
                     daemon=True).start()
    return App(server, service, camera, tracker, face_ops)


class App:
    def __init__(self, server, service, camera, tracker, face_ops):
        self.server, self.service, self.camera = server, service, camera
        self.tracker, self.face_ops = tracker, face_ops
        self.port = server.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"

    def request(self, method, path, body=None, headers=None, raw=False):
        h = {"X-WaveLock-Client": "1", "Content-Type": "application/json"}
        h.update(headers or {})
        data = None if body is None and method == "GET" else json.dumps(body or {}).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=h)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = resp.read()
                return resp.status, (payload if raw else json.loads(payload)), resp.headers
        except urllib.error.HTTPError as e:
            payload = e.read()
            try:
                return e.code, json.loads(payload), e.headers
            except ValueError:
                return e.code, payload, e.headers

    def ok(self, method, path, body=None):
        status, payload, _ = self.request(method, path, body)
        assert status == 200, (path, status, payload)
        return payload["data"]

    def wait(self, predicate, timeout=60, what="condition"):
        end = time.time() + timeout
        while time.time() < end:
            st = self.ok("GET", "/api/status")
            if predicate(st):
                return st
            time.sleep(0.05)
        raise AssertionError(f"timed out waiting for {what}: {st}")

    def close(self):
        self.service.shutdown()
        self.server.shutdown()
        self.server.server_close()


# ------------------------------------------------------------------ flows

def register_via_api(app, username="alice", face=False, pose_method="legacy", overwrite=False):
    app.ok("POST", "/api/camera/start", {})
    app.ok("POST", "/api/register/begin", {"username": username, "overwrite": overwrite,
                                           "pose_method": pose_method})
    if face:
        app.ok("POST", "/api/register/face/capture")
        app.wait(lambda s: s["register"]["face_status"] in ("captured", "failed"), what="face")
    else:
        app.ok("POST", "/api/register/face/skip")
    pool = iter(tw.genuine_set(count=12, seed=100))
    while True:
        st = app.wait(lambda s: s["register"]["status"] != "active"
                      or s["register"]["awaiting_record"], what="awaiting record")
        if st["register"]["status"] != "active":
            break
        app.tracker.play(next(pool))
        app.ok("POST", "/api/register/record")
    reg = st["register"]
    assert reg["status"] == "done", reg
    app.ok("POST", "/api/register/close")
    return reg["summary"]


def sign_in(app, username, raw, hide_hand=False):
    st = app.ok("GET", "/api/status")
    if st["auth"] is None or st["auth"]["username"] != username:
        app.ok("POST", "/api/auth/begin", {"username": username})
    else:
        app.ok("POST", "/api/auth/reset")
    app.tracker.play(raw)
    app.tracker.hide_during_recording = hide_hand
    try:
        status, payload, _ = app.request("POST", "/api/auth/record")
        if status != 200:
            return status, payload
        st = app.wait(lambda s: s["auth"]["phase"] in ("result", "ready"), what="decision")
    finally:
        app.tracker.hide_during_recording = False
    assert st["auth"]["error"] is None, st["auth"]["error"]
    return 200, st["auth"]["result"]


WRONG_ORDER = tw.synth_raw(order=("ring", "pinky", "index", "middle"), seed=7)


# ------------------------------------------------------------------ tests

def test_security_guards():
    with tw.TempDir() as tmp:
        app = make_app(tmp)
        try:
            status, _, headers = app.request("GET", "/", raw=True)
            assert status == 200 and "default-src 'self'" in headers["Content-Security-Policy"]
            assert app.request("GET", "/api/system", headers={"Host": "evil.example:80"})[0] == 403
            assert app.request("GET", "/api/users", headers={"Sec-Fetch-Site": "cross-site"})[0] == 403
            assert app.request("GET", "/api/stream", headers={"Sec-Fetch-Site": "cross-site"})[0] == 403
            assert app.request("POST", "/api/camera/stop", headers={"X-WaveLock-Client": ""})[0] == 403
            assert app.request("POST", "/api/camera/stop", headers={"Content-Type": "text/plain"})[0] == 415
            for path in ("/../gesture_compare.py", "/%2e%2e/gesture_compare.py",
                         "/..%5cgesture_compare.py", "/js/../../web_app.py"):
                assert app.request("GET", path, raw=True)[0] == 404, path
            assert app.request("GET", "/api/nothing")[0] == 404
            assert app.ok("GET", "/api/system")["debug"] is False
        finally:
            app.close()


def test_register_then_sign_in_normal_mode():
    with tw.TempDir() as tmp:
        app = make_app(tmp)
        try:
            summary = register_via_api(app)
            assert summary["samples"] == 5 and summary["face"] == "skipped"
            assert summary["thresholds"]["dtw"] >= 2.0
            users = app.ok("GET", "/api/users")
            assert [u["name"] for u in users] == ["alice"] and "thresholds" not in users[0]

            status, res = sign_in(app, "alice", tw.genuine_set(1, seed=31)[0])
            assert status == 200 and res["granted"] is True, res
            assert "details" not in res and "template_updated" not in res
            status, res = sign_in(app, "alice", WRONG_ORDER)
            assert res["granted"] is False and res["attempts_left"] == 4, res
            assert "details" not in res
            log = app.ok("GET", "/api/activity")
            assert log["today"] == {"total": 2, "granted": 1, "denied": 1}
            assert "reason" not in log["entries"][0]
            app.ok("POST", "/api/auth/end")
        finally:
            app.close()


def test_lockout_after_repeated_failures():
    with tw.TempDir() as tmp:
        app = make_app(tmp)
        try:
            register_via_api(app)
            for i in range(5):
                status, res = sign_in(app, "alice", WRONG_ORDER)
                assert status == 200 and not res["granted"]
            assert res["locked_for"] > 0, res
            status, payload = sign_in(app, "alice", tw.genuine_set(1, seed=31)[0])
            assert status == 423 and payload["code"] == "locked", payload
            assert app.ok("GET", "/api/activity")["locked"][0]["user"] == "alice"
        finally:
            app.close()


def test_too_few_hand_frames_not_counted():
    with tw.TempDir() as tmp:
        app = make_app(tmp)
        try:
            register_via_api(app)
            status, res = sign_in(app, "alice", WRONG_ORDER, hide_hand=True)
            assert res["too_few_frames"] and res["attempts_left"] is None, res
            assert app.ok("GET", "/api/activity")["today"]["total"] == 0
        finally:
            app.close()


def test_debug_mode_shows_scores():
    with tw.TempDir() as tmp:
        app = make_app(tmp, debug=True)
        try:
            register_via_api(app)
            assert "thresholds" in app.ok("GET", "/api/users")[0]
            status, res = sign_in(app, "alice", WRONG_ORDER)
            gates = {g["key"]: g for g in res["details"]["gates"]}
            assert gates["order"]["pass"] is False and "transition_order" in res["details"]["failure_reason"]
            assert "reason" in app.ok("GET", "/api/activity")["entries"][0]
            app.ok("POST", "/api/auth/end")
            run = app.ok("POST", "/api/benchmark/run", {"username": "alice", "kind": "gesture"})
            assert run["state"] == "running"
            end = time.time() + 600
            while app.ok("GET", "/api/benchmark/status")["state"] == "running" and time.time() < end:
                time.sleep(0.5)
            out = app.ok("GET", "/api/benchmark/status")
            assert out["state"] == "done" and "False Acceptance Rate" in out["output"], out
        finally:
            app.close()
    with tw.TempDir() as tmp:
        app = make_app(tmp)                                   # normal mode
        try:
            status, payload, _ = app.request("POST", "/api/benchmark/run",
                                             {"username": "x", "kind": "gesture"})
            assert status == 403 and payload["code"] == "debug_only"
        finally:
            app.close()


def test_existing_user_needs_confirmation_and_cancel_keeps_profile():
    with tw.TempDir() as tmp:
        app = make_app(tmp)
        try:
            register_via_api(app)
            before = sorted(os.listdir(os.path.join(tmp, "alice")))
            status, payload, _ = app.request("POST", "/api/register/begin", {"username": "Alice"})
            assert status == 409 and payload["code"] == "exists"
            st = app.ok("POST", "/api/register/begin", {"username": "alice", "overwrite": True})
            assert st["replacing"] is True
            app.ok("POST", "/api/register/face/skip")
            app.wait(lambda s: s["register"]["awaiting_record"], what="awaiting")
            app.ok("POST", "/api/register/cancel")
            app.ok("POST", "/api/register/close")
            assert sorted(os.listdir(os.path.join(tmp, "alice"))) == before
            # a second tab must not replace a running registration
            app.ok("POST", "/api/register/begin", {"username": "carol"})
            status, payload, _ = app.request("POST", "/api/register/begin", {"username": "dave"})
            assert status == 409 and payload["code"] == "busy", payload
            app.ok("POST", "/api/register/cancel")
            app.ok("POST", "/api/register/close")
            status, payload, _ = app.request("POST", "/api/register/begin", {"username": "../x"})
            assert status == 400 and payload["code"] == "invalid_username"
        finally:
            app.close()


def test_face_enrolment_then_face_is_required():
    real = gesture_auth.verify_face_over_frames
    with tw.TempDir() as tmp:
        app = make_app(tmp)
        try:
            register_via_api(app, face=True)
            assert os.path.exists(os.path.join(tmp, "alice", "face_embedding.npy"))
            app.face_ops.identity = -app.face_ops.identity           # someone else's face
            app.ok("POST", "/api/camera/start", {})
            app.ok("POST", "/api/face/enroll", {"username": "alice"})
            st = app.wait(lambda s: s["face"]["status"] != "capturing", what="face enrol")
            assert st["face"]["status"] == "done", st["face"]
            app.ok("POST", "/api/face/close")
            stored = np.load(os.path.join(tmp, "alice", "face_embedding.npy"))
            assert float(np.dot(stored, app.face_ops.identity)) > 0.99

            def fake_face(match):
                def verify(frames, emb):
                    return {"match": match, "score": 0.8 if match else 0.1,
                            "confidence": 0.9 if match else 0.1,
                            "reason": None if match else "impostor_face_identity",
                            "frames_checked": 6, "faces_detected": 6}
                return verify

            gesture_auth.verify_face_over_frames = fake_face(False)
            status, res = sign_in(app, "alice", tw.genuine_set(1, seed=31)[0])
            assert res["granted"] is False, res
            gesture_auth.verify_face_over_frames = fake_face(True)
            status, res = sign_in(app, "alice", tw.genuine_set(1, seed=31)[0])
            assert res["granted"] is True, res
        finally:
            gesture_auth.verify_face_over_frames = real
            app.close()


def test_busy_no_hand_and_camera_errors():
    with tw.TempDir() as tmp:
        app = make_app(tmp)
        try:
            register_via_api(app)
            # An idle sign-in screen (e.g. a closed tab) must not block other pages.
            app.ok("POST", "/api/auth/begin", {"username": "alice"})
            st = app.ok("POST", "/api/register/begin", {"username": "bob"})
            assert st["username"] == "bob"
            app.ok("POST", "/api/register/cancel")
            app.ok("POST", "/api/register/close")
            # ...but a running sign-in attempt is never interrupted.
            app.ok("POST", "/api/auth/begin", {"username": "alice"})
            app.tracker.play(WRONG_ORDER)
            app.ok("POST", "/api/auth/record")
            status, payload, _ = app.request("POST", "/api/register/begin", {"username": "bob"})
            assert status == 409 and payload["code"] == "busy", payload
            status, payload, _ = app.request("POST", "/api/camera/stop")
            assert status == 409
            app.wait(lambda s: s["auth"]["phase"] == "result", what="attempt done")
            app.tracker.hand_visible = False
            time.sleep(0.2)
            status, payload, _ = app.request("POST", "/api/auth/record")
            assert status == 409 and payload["code"] == "no_hand", payload
            app.tracker.hand_visible = True
            app.ok("POST", "/api/auth/end")
            status, payload, _ = app.request("POST", "/api/auth/begin", {"username": "nobody"})
            assert status == 404
            app.ok("POST", "/api/camera/stop")
            cam = app.ok("POST", "/api/camera/start", {"camera_index": 9})
            assert cam["state"] == "error" and "9" in cam["error"]
            status, _, _ = app.request("POST", "/api/camera/start", {"camera_index": 99})
            assert status == 400
            status, _, _ = app.request("POST", "/api/camera/start", {"camera_index": "x"})
            assert status == 400
        finally:
            app.close()


def test_frames_available_as_soon_as_camera_reports_running():
    # Regression: with a real webcam the first frame arrives ~1 s after the
    # device opens; the stream used to give up before it and stay black.
    with tw.TempDir() as tmp:
        app = make_app(tmp)
        try:
            status, body, headers = app.request("GET", "/api/frame?after=-1", raw=True)
            assert status == 204 and headers["X-Camera-State"] == "stopped"
            cam = app.ok("POST", "/api/camera/start", {})
            assert cam["state"] == "running" and cam["frame_aspect"], cam
            status, body, headers = app.request("GET", "/api/frame?after=-1", raw=True)
            assert status == 200 and body[:2] == JPEG_SOI, status
            first = int(headers["X-Frame-Id"])
            status, body, headers = app.request("GET", f"/api/frame?after={first}", raw=True)
            assert status == 200 and int(headers["X-Frame-Id"]) > first
            # restart: no stale frame, then live again
            app.ok("POST", "/api/camera/stop")
            app.ok("POST", "/api/camera/start", {})
            status, body, headers = app.request("GET", "/api/frame?after=-1", raw=True)
            assert status == 200 and int(headers["X-Frame-Id"]) > first
        finally:
            app.close()


def test_stream_waits_for_a_slow_first_frame():
    with tw.TempDir() as tmp:
        app = make_app(tmp)
        try:
            app.camera.start(None)                     # do not wait for "running"
            req = urllib.request.Request(app.base + "/api/stream")
            with urllib.request.urlopen(req, timeout=15) as resp:
                chunk = resp.read(30000)
            assert JPEG_SOI in chunk, "stream ended before the first frame"
        finally:
            app.close()


def test_stream_delivers_jpeg_frames():
    with tw.TempDir() as tmp:
        app = make_app(tmp)
        try:
            app.ok("POST", "/api/camera/start", {})
            req = urllib.request.Request(app.base + "/api/stream")
            with urllib.request.urlopen(req, timeout=10) as resp:
                assert resp.headers["Content-Type"].startswith("multipart/x-mixed-replace")
                chunk = resp.read(30000)
            assert b"--wavelockframe" in chunk and b"\xff\xd8" in chunk
        finally:
            app.close()


ALL = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    wanted = sys.argv[1:]
    tests = [t for t in ALL if not wanted or any(w in t.__name__ for w in wanted)]
    failed = 0
    for t in tests:
        start = time.time()
        try:
            t()
            print(f"PASS  {t.__name__}  ({time.time() - start:.1f}s)")
        except Exception as e:
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
