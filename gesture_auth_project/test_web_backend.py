"""
Web UI backend tests — run with:  python test_web_backend.py [name ...]

No webcam needed: a SIMULATED camera replays synthetic landmark sequences
(the same generator as test_wavelock.py) through the real HTTP server,
service and decision code. The simulated parts live only in this file.
"""

import json
import collections
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
from utils import attempt_limit as al
from utils.secure_store import load_array
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
    """Frames labelled SIMULATED CAMERA, at SIM_FPS (SimSource.dark: very dark room)."""
    dark = False

    def __init__(self, index, face_ops):
        if index == 9:
            raise CameraError("Could not open camera 9 (simulated failure).")
        self.face_ops = face_ops

    def read(self):
        time.sleep(1.0 / SIM_FPS)
        frame = np.full((480, 640, 3), (8, 6, 5) if SimSource.dark else (70, 62, 56), np.uint8)
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
        self.override = None              # still pose shown outside recordings (liveness)
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
            lm = self.override if self.override is not None else self.idle
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
        self.yaw = 0.0                    # nose offset in eye distances (head turn)
        self.delay = 0.0                  # seconds per model call (slow laptop on battery)
        rng = np.random.RandomState(11)
        self.identity = rng.randn(128).astype(np.float32)
        self.identity /= np.linalg.norm(self.identity)
        self.seen = collections.deque(maxlen=200)   # (frame, yaw) of recent detections

    def detect(self, frame):
        time.sleep(self.delay)
        if not self.face_present:
            return None, None
        self.seen.append((frame, self.yaw))
        return tw.face_row(self.yaw), (270, 50, 110, 140)

    def embed(self, frame, face_row):
        return self.identity + np.random.normal(0, 0.01, 128).astype(np.float32)

    def verify(self, frame, embedding):
        time.sleep(2 * self.delay)        # real verify_face detects again + runs SFace
        score = float(np.dot(self.identity, embedding) / (np.linalg.norm(embedding) + 1e-9))
        # like SFace, a clearly turned face scores far lower than a frontal one
        yaw = next((y for f, y in reversed(self.seen) if f is frame), 0.0)
        if abs(yaw) > 0.15:
            score *= 0.4
        match, conf, reason = classify_face_score(score)
        return match, score, conf, (270, 50, 110, 140), reason


class LivenessSolver(threading.Thread):
    """Plays the user's part of a liveness challenge on the simulated camera:
    shows the requested number of fingers, then turns the head the requested
    way after a frontal moment (so the baseline is taken first)."""

    def __init__(self, camera, tracker, face_ops):
        super().__init__(daemon=True)
        self.camera, self.tracker, self.face_ops = camera, tracker, face_ops
        self.enabled = True
        self.turn_head = True             # False = a flat photo: it cannot turn
        self.stopped = False

    def run(self):
        step_seen = None
        step_started = 0.0
        while not self.stopped:
            st = self.camera.challenge_state() if self.enabled else None
            if st is None or st["state"] != "running":
                if step_seen is not None:
                    self.tracker.override = None
                    self.face_ops.yaw = 0.0
                    step_seen = None
                time.sleep(0.02)
                continue
            step = st["steps"][st["index"]]
            key = (st["index"], step["kind"])
            if key != step_seen:
                step_seen, step_started = key, time.time()
                self.face_ops.yaw = 0.0
            if step["kind"] == "fingers":
                self.tracker.override = tw.finger_pose(step["count"])
            else:
                self.tracker.override = tw.finger_pose(2)           # hand stays up
                if self.turn_head and time.time() - step_started > 0.8:   # after the baseline
                    self.face_ops.yaw = -0.25 if step["direction"] == "left" else 0.25
            time.sleep(0.02)


def make_app(templates_dir, debug=False, liveness=True, auto_liveness=True):
    """Start a real WaveLockServer on a free port with the simulated camera.
    With liveness on, a LivenessSolver answers challenges unless
    auto_liveness is False (tests of failing challenges)."""
    service_mod.RECORDING_DURATION_SEC = FAST_RECORDING_SEC
    face_ops = SimFaceOps()
    tracker = SimTracker()
    def tracker_factory():
        tracker.warm = False              # every camera start warms up again
        return tracker

    camera = CameraService(source_factory=lambda idx: SimSource(idx, face_ops),
                           tracker_factory=tracker_factory, face_ops=face_ops)
    tracker.camera = camera
    service = WaveLockService(camera, templates_dir=templates_dir, debug=debug, liveness=liveness)
    server = WaveLockServer(service, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1},
                     daemon=True).start()
    app = App(server, service, camera, tracker, face_ops)
    app.solver = LivenessSolver(camera, tracker, face_ops)
    app.solver.enabled = auto_liveness
    app.solver.start()
    return app


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
        if getattr(self, "solver", None) is not None:
            self.solver.stopped = True
        self.service.shutdown()
        self.server.shutdown()
        self.server.server_close()


# ------------------------------------------------------------------ flows

def register_via_api(app, username="alice", face=False, pose_method="legacy", overwrite=False,
                     check_level=None):
    app.ok("POST", "/api/camera/start", {})
    body = {"username": username, "overwrite": overwrite, "pose_method": pose_method}
    if check_level is not None:
        body["check_level"] = check_level
    app.ok("POST", "/api/register/begin", body)
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
        st = app.wait(lambda s: s["auth"]["phase"] in ("result", "ready")
                      or s["auth"].get("record_ready") is not None, what="decision")
        if st["auth"].get("record_ready") is not None:     # liveness passed: start when ready
            app.ok("POST", "/api/auth/record")
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
            # refused POSTs with a body still get their response (the body is read first)
            for i in range(30):
                hdr = [{"X-WaveLock-Client": ""}, {"Content-Type": "text/plain"}][i % 2]
                st = app.request("POST", "/api/camera/stop", {"pad": "x" * 60000}, headers=hdr)[0]
                assert st == (403, 415)[i % 2], st
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
            assert summary["check_level"] == "standard"
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
            assert out["state"] == "done" and "ACCEPTANCE RATES PER ATTACK" in out["output"], out
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
            status, payload, _ = app.request("POST", "/api/register/begin", {"username": "Alice"})
            assert status == 409 and payload["code"] == "exists"
            # only the owner may replace a profile: refused until alice signs in
            status, payload, _ = app.request("POST", "/api/register/begin",
                                             {"username": "alice", "overwrite": True})
            assert status == 403 and payload["code"] == "owner_signin_required", payload
            status, res = sign_in(app, "alice", tw.genuine_set(1, seed=31)[0])
            assert res["granted"], res
            app.ok("POST", "/api/auth/end")
            # snapshot after the sign-in (a grant may age a template and write its anchor)
            before = {f: os.path.getmtime(os.path.join(tmp, "alice", f))
                      for f in os.listdir(os.path.join(tmp, "alice"))}
            st = app.ok("POST", "/api/register/begin", {"username": "alice", "overwrite": True})
            assert st["replacing"] is True
            app.ok("POST", "/api/register/face/skip")
            app.wait(lambda s: s["register"]["awaiting_record"], what="awaiting")
            app.ok("POST", "/api/register/cancel")
            app.ok("POST", "/api/register/close")
            assert {f: os.path.getmtime(os.path.join(tmp, "alice", f))
                    for f in os.listdir(os.path.join(tmp, "alice"))} == before
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
            # changing the face needs a recent sign-in by the owner
            status, payload, _ = app.request("POST", "/api/face/enroll", {"username": "alice"})
            assert status == 403 and payload["code"] == "owner_signin_required", payload
            al.record_attempt("alice", True, tmp)                     # owner signed in just now
            app.ok("POST", "/api/face/enroll", {"username": "alice"})
            st = app.wait(lambda s: s["face"]["status"] != "capturing", what="face enrol")
            assert st["face"]["status"] == "done", st["face"]
            app.ok("POST", "/api/face/close")
            stored = load_array(os.path.join(tmp, "alice", "face_embedding.npy"))
            assert float(np.dot(stored, app.face_ops.identity)) > 0.99
            # the verification is used up by the change
            assert not al.verified_recently("alice", tmp)
            status, payload, _ = app.request("POST", "/api/face/enroll", {"username": "alice"})
            assert status == 403, payload

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
    # liveness off: the "show your hand first" refusal belongs to the plain
    # flow (with liveness the challenge itself asks for the hand)
    with tw.TempDir() as tmp:
        app = make_app(tmp, liveness=False)
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


def test_slow_face_models_do_not_starve_the_recording():
    # Regression (2026-10-19): in developer mode with a face in view the camera
    # thread ran YuNet + SFace itself, so on a slow (battery) CPU the preview
    # fell to ~8 fps and a 3 s recording got too few, uneven hand frames.
    import gesture_auth
    real_check = gesture_auth.verify_face_over_frames
    # The decision's face check runs the real models, which find no face in
    # simulated frames; this test is about frame timing, so stand in a match.
    gesture_auth.verify_face_over_frames = lambda frames, emb: {
        "match": True, "score": 0.74, "confidence": 0.86, "reason": None,
        "frames_checked": 6, "faces_detected": 6}
    with tw.TempDir() as tmp:
        app = make_app(tmp, debug=True)
        try:
            register_via_api(app, face=True)
            app.face_ops.delay = 0.3                     # 300 ms detect, 600 ms verify
            app.ok("POST", "/api/auth/begin", {"username": "alice"})
            time.sleep(1.0)                               # identity preview running
            first = app.camera._frame_id
            time.sleep(2.0)
            produced = (app.camera._frame_id - first) / 2.0   # frames per second, really delivered
            assert produced > SIM_FPS * 0.5, f"preview starved by face models: {produced:.1f} frames/s"
            app.tracker.play(tw.genuine_set(1, seed=31)[0])
            app.ok("POST", "/api/auth/record")
            st = app.wait(lambda s: s["auth"]["phase"] == "result"
                          or s["auth"].get("record_ready") is not None, what="liveness")
            if st["auth"].get("record_ready") is not None:     # liveness passed: start when ready
                app.ok("POST", "/api/auth/record")
            st = app.wait(lambda s: s["auth"]["phase"] == "result", what="attempt done")
            frames = st["auth"]["result"]["hand_frames"]
            expected = SIM_FPS * FAST_RECORDING_SEC
            # Nominal rate; a busy machine delivers less even without face work.
            # Before the fix this was a handful of frames (one per face call).
            assert frames > 0.5 * expected, f"recording got {frames} of ~{expected:.0f} frames"
            assert st["auth"]["result"]["granted"], st["auth"]["result"]
        finally:
            app.close()
            gesture_auth.verify_face_over_frames = real_check


def test_check_level_api():
    with tw.TempDir() as tmp:
        app = make_app(tmp, liveness=False)
        try:
            status, payload, _ = app.request("POST", "/api/register/begin",
                                             {"username": "bob", "check_level": "none"})
            assert status == 400, payload
            summary = register_via_api(app, "bob", check_level="relaxed")
            assert summary["check_level"] == "relaxed"
            assert tw.gc._read_config("bob", tmp)["check_level"] == "relaxed"
            register_via_api(app, "carl")
            levels = {u["name"]: u["check_level"] for u in app.ok("GET", "/api/users")}
            assert levels == {"bob": "relaxed", "carl": "standard"}, levels
            status, res = sign_in(app, "bob", tw.genuine_set(1, seed=31)[0])
            assert status == 200 and res["granted"], res
        finally:
            app.close()


def test_liveness_challenge_flow():
    from utils import liveness as lv
    real_timeout = lv.CHALLENGE_TIMEOUT_SEC
    lv.CHALLENGE_TIMEOUT_SEC = 3.0
    try:
        with tw.TempDir() as tmp:
            app = make_app(tmp)                                   # liveness on (default)
            try:
                register_via_api(app, face=True)
                genuine = tw.genuine_set(1, seed=31)[0]

                # 1. owner answers the challenge -> normal decision, verification set
                real = gesture_auth.verify_face_over_frames
                gesture_auth.verify_face_over_frames = lambda f, e: {
                    "match": True, "score": 0.8, "confidence": 0.9, "reason": None,
                    "frames_checked": 6, "faces_detected": 6}
                try:
                    status, res = sign_in(app, "alice", genuine)
                    assert status == 200 and res["granted"], res
                    # after the challenge the user starts the recording when ready
                    app.ok("POST", "/api/auth/reset")
                    app.tracker.play(genuine)
                    app.ok("POST", "/api/auth/record")
                    st = app.wait(lambda s: s["auth"]["record_ready"] is not None, what="ready")
                    assert st["auth"]["phase"] == "challenge", st["auth"]
                    assert 0 < st["auth"]["record_ready"] <= lv.RECORD_READY_WINDOW_SEC, st["auth"]
                    time.sleep(1.0)                                   # nothing starts by itself
                    assert app.ok("GET", "/api/status")["auth"]["phase"] == "challenge"
                    app.ok("POST", "/api/auth/record")
                    st = app.wait(lambda s: s["auth"]["phase"] in ("result", "ready"), what="decision")
                    assert st["auth"]["result"]["granted"], st["auth"]["result"]
                    # not started in time: back to ready, NOT counted as a failed attempt
                    real_window = lv.RECORD_READY_WINDOW_SEC
                    lv.RECORD_READY_WINDOW_SEC = 1.0
                    try:
                        app.ok("POST", "/api/auth/reset")
                        log_before = len(app.ok("GET", "/api/activity")["entries"])
                        app.ok("POST", "/api/auth/record")
                        st = app.wait(lambda s: s["auth"]["phase"] == "ready"
                                      and s["auth"]["notice"] == "ready_timeout", what="ready timeout")
                        assert st["auth"]["result"] is None
                        assert len(app.ok("GET", "/api/activity")["entries"]) == log_before
                    finally:
                        lv.RECORD_READY_WINDOW_SEC = real_window
                finally:
                    gesture_auth.verify_face_over_frames = real

                # 2. a video cannot follow the instruction: nobody shows the fingers
                app.solver.enabled = False
                status, res = sign_in(app, "alice", genuine)
                assert res["liveness_failed"] and res["liveness_reason"] == "liveness_timeout_fingers", res
                assert not res["granted"] and res["attempts_left"] == 4

                # 3. a flat photo: fingers shown, but the head never turns
                app.solver.enabled, app.solver.turn_head = True, False
                status, res = sign_in(app, "alice", genuine)
                assert res["liveness_reason"] == "liveness_timeout_head_turn", res
                app.solver.turn_head = True

                # 4. someone else answers the challenge with their own face
                app.face_ops.identity = -app.face_ops.identity
                status, res = sign_in(app, "alice", genuine)
                assert res["liveness_reason"] == "liveness_face_mismatch", res
                app.face_ops.identity = -app.face_ops.identity
                log = app.ok("GET", "/api/activity")
                assert log["entries"][0]["granted"] is False

                # 5. leaving the screen mid-challenge counts nothing
                app.solver.enabled = False
                before = al.load_attempt_state("alice", tmp)["consecutive_failures"]
                app.ok("POST", "/api/auth/reset")
                app.ok("POST", "/api/auth/record")
                assert app.ok("GET", "/api/status")["auth"]["phase"] == "challenge"
                app.ok("POST", "/api/auth/end")
                time.sleep(0.3)
                assert al.load_attempt_state("alice", tmp)["consecutive_failures"] == before
                assert app.ok("GET", "/api/system")["liveness"] is True
            finally:
                app.close()
    finally:
        lv.CHALLENGE_TIMEOUT_SEC = real_timeout


def test_status_reports_brightness_for_lighting_hint():
    with tw.TempDir() as tmp:
        app = make_app(tmp)
        try:
            app.ok("POST", "/api/camera/start", {})
            app.wait(lambda s: s["camera"]["brightness"] is not None, what="brightness")
            assert app.ok("GET", "/api/status")["camera"]["brightness"] > 40
            SimSource.dark = True
            app.wait(lambda s: s["camera"]["brightness"] < 40, what="dark frames reported")
        finally:
            SimSource.dark = False
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
