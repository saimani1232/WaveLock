"""
Camera service for the WaveLock web UI.

One background thread owns the webcam. For every frame it does what the
desktop programs do (mirror, MediaPipe Hands, throttled YuNet face box),
encodes a JPEG for the browser stream, and feeds any active gesture
recording or face capture. The capture source, the hand tracker and the face
operations are injectable so the web backend can be tested without a camera.
"""

import threading
import time

import cv2
import numpy as np

from utils.landmarks import extract_landmarks
from utils.liveness import count_extended_fingers, yaw_proxy
from utils.face_auth import FACE_SAMPLE_GAP_SEC

FACE_DETECT_INTERVAL_SEC = 0.25      # same throttle as gesture_auth.py's preview
IDENTITY_INTERVAL_SEC = 0.5          # developer identity preview (detect + SFace)
BRIGHTNESS_INTERVAL_SEC = 0.5        # UI lighting hint only; never used by a decision
FACE_CAPTURE_TIMEOUT_SEC = 15.0
JPEG_QUALITY = 72

# YuNet / SFace networks are not thread-safe. The camera thread only ever
# try-acquires this lock (it skips the face box for a frame when busy); the
# sign-in decision holds it while evaluate_attempt runs the face models.
FACE_LOCK = threading.Lock()

# Overlay colours (BGR)
_NEUTRAL = (200, 190, 120)
_GREEN = (110, 200, 60)
_AMBER = (0, 170, 250)
_RED = (80, 80, 240)


class CameraError(RuntimeError):
    """Camera could not be opened or stopped delivering frames."""


# ---------------------------------------------------------------- real parts

class OpenCvSource:
    """Webcam opened exactly like the CLI (utils.camera_utils.open_camera)."""

    def __init__(self, camera_index=None):
        from gesture_capture import setup_camera
        try:
            self.cap = setup_camera(camera_index=camera_index)
        except SystemExit:
            # open_camera() calls sys.exit(1) when the device cannot be opened.
            which = "the auto-detected camera" if camera_index is None else f"camera {camera_index}"
            raise CameraError(
                f"Could not open {which}. Close other apps that use the "
                f"webcam or choose another camera in Settings."
            )

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


class MediaPipeTracker:
    """MediaPipe Hands with the CLI settings (gesture_capture.setup_mediapipe)."""

    def __init__(self):
        from gesture_capture import setup_mediapipe
        self.hands, self.mp_hands, self.drawing, self.styles = setup_mediapipe()

    def process(self, frame):
        """Return the first hand's (21, 3) landmarks or None; draws on frame."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        if not results.multi_hand_landmarks:
            return None
        for hand_lms in results.multi_hand_landmarks:
            self.drawing.draw_landmarks(
                frame, hand_lms, self.mp_hands.HAND_CONNECTIONS,
                self.styles.get_default_hand_landmarks_style(),
                self.styles.get_default_hand_connections_style(),
            )
        return extract_landmarks(results.multi_hand_landmarks[0])

    def close(self):
        self.hands.close()


class FaceOps:
    """YuNet / SFace through utils.face_auth."""

    def detect(self, frame):
        from utils.face_auth import detect_primary_face
        return detect_primary_face(frame)

    def embed(self, frame, face_row):
        from utils.face_auth import extract_face_embedding
        return extract_face_embedding(frame, face_row)

    def verify(self, frame, embedding):
        from utils.face_auth import verify_face
        return verify_face(frame, embedding)


# ---------------------------------------------------------------- jobs

class Recording:
    """A gesture recording in progress (filled by the camera thread)."""

    def __init__(self, duration, keep_clean_frames):
        self.duration = float(duration)
        self.keep_clean_frames = keep_clean_frames
        self.started = None
        self.landmarks = []
        self.clean_frames = []
        self.frames_seen = 0
        self.frame_aspect = None
        self.error = None
        self.done = threading.Event()

    def progress(self, now=None):
        if self.started is None:
            return 0.0
        now = time.time() if now is None else now
        return min(1.0, (now - self.started) / self.duration)


class ChallengeJob:
    """A liveness challenge being answered (utils.liveness.Challenge)."""

    def __init__(self, challenge):
        self.challenge = challenge
        self.face_frame = None            # latest clean frame with a detected face
        self.error = None
        self.done = threading.Event()


class FaceCapture:
    """Collects face embeddings from consecutive frames with a detected face."""

    def __init__(self, required):
        self.required = required
        self.embeddings = []
        self.last_sample = 0.0
        self.started = time.time()
        self.error = None
        self.done = threading.Event()


# ---------------------------------------------------------------- service

class CameraService:
    def __init__(self, source_factory=OpenCvSource, tracker_factory=MediaPipeTracker,
                 face_ops=None):
        self._source_factory = source_factory
        self._tracker_factory = tracker_factory
        self.face_ops = face_ops or FaceOps()

        self._lock = threading.RLock()
        self._frame_cond = threading.Condition()
        self._thread = None
        self._stop = threading.Event()

        self.state = "stopped"            # stopped | starting | running | error
        self.error = None
        self.camera_index = None
        self.fps = 0.0
        # Frame-rate diagnostics (moving averages, milliseconds per frame):
        # time spent waiting for the webcam vs. time spent processing.
        self.read_ms = 0.0
        self.process_ms = 0.0
        self.face_ms = 0.0                # face thread: detection (+ SFace) per check
        self.hand_visible = False
        self.face_bbox = None
        self.face_label = None            # (match, score, reason) in identity preview
        self.frame_aspect = None
        self.brightness = None            # mean grey level 0-255 (lighting hint for the UI)

        self._jpeg = None
        self._frame_id = 0
        self._recording = None
        self._face_capture = None
        self._challenge = None
        self._identity = None             # (embedding, username) for debug preview
        # Latest clean frame for the face thread: the camera thread never waits
        # for the face models (they took 150-350 ms per frame on battery and
        # starved the gesture recording of frames).
        self._face_cond = threading.Condition()
        self._face_frame = None
        self._face_frame_id = 0
        self._face_thread = None

    # ---- lifecycle
    def start(self, camera_index=None):
        with self._lock:
            if (self.state in ("starting", "running")
                    and (camera_index is None or camera_index == self.camera_index)):
                return
        self.stop()
        with self._lock:
            self._stop.clear()
            self.state = "starting"
            self.error = None
            self.camera_index = camera_index
            self.fps = 0.0
            self.read_ms = 0.0
            self.process_ms = 0.0
            self.face_ms = 0.0
            self.brightness = None
        with self._frame_cond:
            self._jpeg = None               # never serve a frame from a previous run
            self._thread = threading.Thread(target=self._run, args=(camera_index,),
                                            name="wavelock-camera", daemon=True)
            self._thread.start()

    def stop(self):
        thread = self._thread
        self._stop.set()
        with self._face_cond:
            self._face_cond.notify_all()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)
        face_thread = self._face_thread
        if face_thread is not None and face_thread is not threading.current_thread():
            face_thread.join(timeout=5.0)
        self._face_thread = None
        with self._lock:
            self._thread = None
            if self.state != "error":
                self.state = "stopped"
            self.hand_visible = False
            self.face_bbox = None
            self.face_label = None
        with self._frame_cond:
            self._frame_cond.notify_all()

    def wait_until_running(self, timeout=20.0):
        end = time.time() + timeout
        while time.time() < end:
            if self.state in ("running", "error", "stopped"):
                return self.state == "running"
            time.sleep(0.02)
        return False

    def set_identity_preview(self, embedding, username=None):
        """Debug mode only: label the live face box as match / impostor."""
        self._identity = None if embedding is None else (embedding, username)
        self.face_label = None

    # ---- jobs
    def start_recording(self, duration, keep_clean_frames=True):
        with self._lock:
            if self.state != "running":
                raise CameraError("The camera is not running.")
            if self._recording is not None or self._face_capture is not None or self._challenge is not None:
                raise RuntimeError("A capture is already in progress.")
            if not self.hand_visible:
                raise RuntimeError("No hand detected. Show your hand to the camera first.")
            rec = Recording(duration, keep_clean_frames)
            self._recording = rec
            return rec

    def start_challenge(self, challenge):
        with self._lock:
            if self.state != "running":
                raise CameraError("The camera is not running.")
            if self._recording is not None or self._face_capture is not None or self._challenge is not None:
                raise RuntimeError("A capture is already in progress.")
            job = ChallengeJob(challenge)
            self._challenge = job
            return job

    def start_face_capture(self, required=5):
        with self._lock:
            if self.state != "running":
                raise CameraError("The camera is not running.")
            if self._recording is not None or self._face_capture is not None or self._challenge is not None:
                raise RuntimeError("A capture is already in progress.")
            cap = FaceCapture(required)
            self._face_capture = cap
            return cap

    def cancel_jobs(self, reason="cancelled"):
        with self._lock:
            for job in (self._recording, self._face_capture, self._challenge):
                if job is not None and not job.done.is_set():
                    job.error = reason
                    if isinstance(job, ChallengeJob):
                        job.challenge.fail("liveness_cancelled")
                    job.done.set()
            self._recording = None
            self._face_capture = None
            self._challenge = None

    def recording_progress(self):
        rec = self._recording
        if rec is None:
            return None
        return {"progress": round(rec.progress(), 3), "hand_frames": len(rec.landmarks),
                "duration": rec.duration}

    def status(self):
        return {
            "state": self.state,
            "error": self.error,
            "camera_index": self.camera_index,
            "fps": round(self.fps, 1),
            "read_ms": round(self.read_ms, 1),        # waiting for the webcam
            "process_ms": round(self.process_ms, 1),  # hands + JPEG (camera thread)
            "face_ms": round(self.face_ms, 1),        # face thread, per check
            "hand_visible": bool(self.hand_visible),
            "face_in_view": self.face_bbox is not None,
            "frame_aspect": self.frame_aspect,
            "brightness": None if self.brightness is None else round(self.brightness),
        }

    def wait_jpeg(self, last_id, timeout=2.0):
        """
        Block until a frame newer than last_id exists, at most `timeout` s.
        Returns (frame_id, jpeg_bytes) or (last_id, None).

        Before the first frame of a run there is no JPEG yet; this waits for
        it instead of returning at once (the old early return made the
        browser stream give up while a real camera was still warming up).
        """
        end = time.time() + timeout
        with self._frame_cond:
            while self._jpeg is None or self._frame_id == last_id:
                remaining = end - time.time()
                if remaining <= 0:
                    return last_id, None
                self._frame_cond.wait(remaining)
            return self._frame_id, self._jpeg

    # ---- camera thread
    def _run(self, camera_index):
        source = tracker = None
        try:
            tracker = self._tracker_factory()
            source = self._source_factory(camera_index)
        except Exception as e:                       # CameraError, cv2.error, OSError...
            self._fail(str(e) or type(e).__name__)
            for part in (source, tracker):
                self._close(part)
            return

        # The state becomes "running" only when the first frame has been
        # processed and encoded (MediaPipe warms up on that frame).
        self._face_thread = threading.Thread(target=self._face_loop, name="wavelock-face",
                                             daemon=True)
        self._face_thread.start()
        last_light = 0.0
        prev = time.time()
        try:
            while not self._stop.is_set():
                t_read = time.perf_counter()
                ok, frame = source.read()
                t_proc = time.perf_counter()
                if not ok or frame is None:
                    self._fail("The camera stopped delivering frames.")
                    break
                frame = cv2.flip(frame, 1)
                clean = frame.copy()
                h, w = frame.shape[:2]
                landmarks = tracker.process(frame)
                now = time.time()

                rec = self._recording
                if rec is None:                       # face thread sleeps through recordings
                    with self._face_cond:             # hand the frame over; never wait
                        self._face_frame = clean
                        self._face_frame_id += 1
                        self._face_cond.notify_all()

                self.hand_visible = landmarks is not None
                self.frame_aspect = w / h
                if now - last_light >= BRIGHTNESS_INTERVAL_SEC:
                    # 1/8-size grey copy: about 0.1 ms, so it costs no frame rate
                    small = cv2.resize(clean, (w // 8, h // 8), interpolation=cv2.INTER_AREA)
                    self.brightness = float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())
                    last_light = now
                if rec is not None:
                    self._feed_recording(rec, now, landmarks, clean, w / h)
                job = self._challenge
                if job is not None:
                    job.challenge.feed_hand(count_extended_fingers(landmarks), now)
                    job.challenge.tick(now)
                    self._finish_challenge_if_done(job)

                self._draw_face_box(frame)
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if ok:
                    with self._frame_cond:
                        self._jpeg = buf.tobytes()
                        self._frame_id += 1
                        self._frame_cond.notify_all()
                    if self.state == "starting":
                        with self._lock:
                            if self.state == "starting":
                                self.state = "running"
                dt = now - prev
                prev = now
                if dt > 0:
                    self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps else 1.0 / dt
                read_ms = 1000.0 * (t_proc - t_read)
                proc_ms = 1000.0 * (time.perf_counter() - t_proc)
                self.read_ms = 0.9 * self.read_ms + 0.1 * read_ms if self.read_ms else read_ms
                self.process_ms = 0.9 * self.process_ms + 0.1 * proc_ms if self.process_ms else proc_ms
        except Exception as e:
            self._fail(f"Camera processing failed: {e}")
        finally:
            self.cancel_jobs("camera stopped")
            self._close(source)
            self._close(tracker)
            with self._lock:
                if self.state != "error":
                    self.state = "stopped"
                self.hand_visible = False
            with self._frame_cond:
                self._frame_cond.notify_all()

    def _face_loop(self):
        """
        Face preview and face capture, off the camera thread.

        While a gesture is being recorded nothing runs here (like the desktop
        gesture_auth.py, which pauses its face preview while recording): the
        recording gets every frame the camera delivers. The face decision of
        a sign-in is not made here; it runs afterwards on the frames saved in
        the recording (gesture_auth.evaluate_attempt).
        """
        seen_id = 0                       # newest frame this loop has looked at
        last_face = last_ident = 0.0
        while not self._stop.is_set():
            with self._face_cond:
                # Always sleep until a NEW frame arrives (or 0.25 s), so this
                # thread never spins and takes CPU from the camera thread.
                if self._face_frame_id == seen_id and not self._stop.is_set():
                    self._face_cond.wait(0.25)
                frame, frame_id = self._face_frame, self._face_frame_id
            new_frame = frame_id != seen_id
            seen_id = frame_id
            fcap = self._face_capture
            now = time.time()
            if fcap is not None:
                self._check_face_timeout(fcap, now)
            if self._stop.is_set() or frame is None or not new_frame:
                continue
            if self._recording is not None:           # gesture recording: hands only
                continue
            job = self._challenge
            if fcap is None and job is None and now - last_face < FACE_DETECT_INTERVAL_SEC:
                continue
            if not FACE_LOCK.acquire(blocking=False):  # a sign-in decision is using the models
                time.sleep(0.05)
                continue
            t0 = time.perf_counter()
            try:
                face_row, bbox = self.face_ops.detect(frame)
                last_face = now
                self.face_bbox = bbox
                ident = self._identity
                if ident is None or bbox is None:
                    self.face_label = None
                elif now - last_ident >= IDENTITY_INTERVAL_SEC:
                    match, score, _, _, reason = self.face_ops.verify(frame, ident[0])
                    self.face_label = (bool(match), float(score), reason)
                    last_ident = now
                fcap = self._face_capture
                if fcap is not None:
                    self._feed_face_capture(fcap, frame, face_row, now)
                job = self._challenge
                if job is not None:
                    if face_row is not None:
                        job.face_frame = frame
                    job.challenge.feed_face(yaw_proxy(face_row), now)
                    self._finish_challenge_if_done(job)
            except Exception:
                self.face_bbox = None             # never let a face error stop the camera
                self.face_label = None
            finally:
                FACE_LOCK.release()
            ms = 1000.0 * (time.perf_counter() - t0)
            self.face_ms = 0.8 * self.face_ms + 0.2 * ms if self.face_ms else ms

    def _feed_recording(self, rec, now, landmarks, clean, aspect):
        if rec.started is None:
            rec.started = now
        rec.frames_seen += 1
        rec.frame_aspect = aspect
        if rec.keep_clean_frames:
            rec.clean_frames.append(clean)
        if landmarks is not None:
            rec.landmarks.append(np.asarray(landmarks, dtype=np.float64))
        if now - rec.started >= rec.duration:
            with self._lock:
                if self._recording is rec:
                    self._recording = None
            rec.done.set()

    def _feed_face_capture(self, fcap, clean, face_row, now):
        # frames at least FACE_SAMPLE_GAP_SEC apart: small head movements are covered
        if face_row is not None and now - fcap.last_sample >= FACE_SAMPLE_GAP_SEC:
            fcap.last_sample = now
            fcap.embeddings.append(np.asarray(self.face_ops.embed(clean, face_row)).flatten())
        if len(fcap.embeddings) >= fcap.required:
            self._finish_face_capture(fcap)
        else:
            self._check_face_timeout(fcap, now)

    def _check_face_timeout(self, fcap, now):
        if now - fcap.started > FACE_CAPTURE_TIMEOUT_SEC:
            fcap.error = "No face was detected long enough. Face the camera in good light."
            self._finish_face_capture(fcap)

    def _finish_challenge_if_done(self, job):
        if job.challenge.state == "running":
            return
        with self._lock:
            if self._challenge is job:
                self._challenge = None
        job.done.set()

    def challenge_state(self):
        job = self._challenge
        return None if job is None else job.challenge.public()

    def _finish_face_capture(self, fcap):
        with self._lock:
            if self._face_capture is fcap:
                self._face_capture = None
        fcap.done.set()

    def _draw_face_box(self, frame):
        bbox = self.face_bbox
        if bbox is None:
            return
        x, y, w, h = bbox
        label = self.face_label
        if label is None:
            # Protected mode: box only. The page shows "Face in view" as a
            # readable chip; small burned-in text was unreadable on phones.
            color, text = _NEUTRAL, None
        else:
            match, score, reason = label
            if match:
                color, text = _GREEN, f"[MATCH] cosine {score:.2f}"
            elif reason == "lookalike_sibling_detected":
                color, text = _AMBER, f"[ALERT] look-alike {score:.2f}"
            else:
                color, text = _RED, f"[ALERT] impostor {score:.2f}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        if text:                          # developer identity preview
            cv2.putText(frame, text, (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, color, 1, cv2.LINE_AA)

    def _fail(self, message):
        with self._lock:
            self.state = "error"
            self.error = message

    @staticmethod
    def _close(part):
        if part is None:
            return
        for name in ("release", "close"):
            fn = getattr(part, name, None)
            if fn is not None:
                try:
                    fn()
                except Exception:
                    pass
