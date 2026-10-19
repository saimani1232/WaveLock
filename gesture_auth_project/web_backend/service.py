"""
WaveLock workflows for the web UI (no HTTP code here).

Every decision is made by the existing modules; this file only sequences
them for a browser instead of an OpenCV window:

- sign-in        -> gesture_auth.evaluate_attempt, utils.attempt_limit,
                    gesture_compare.update_templates_if_high_confidence
- registration   -> gesture_capture.collect_gesture_samples (quality gate,
                    retry/restart policy), threshold details, cohort
                    uniqueness, gesture_compare.save_registration
- face enrolment -> same 5-frame mean embedding as the CLI,
                    utils.face_auth.save_face_embedding

Results returned in normal mode contain only the decision, attempts left and
lockout time; scores are added only when the server runs with --debug.
"""

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime

import numpy as np

from gesture_compare import (
    TEMPLATES_DIR,
    NUM_REGISTRATION_SAMPLES,
    NEW_PROFILE_TEMPORAL_TRIM,
    matching_recording,
    list_registered_users,
    normalize_username,
    load_user_profile,
    save_registration,
    set_face_enrolled_flag,
    update_templates_if_high_confidence,
    compute_threshold_from_samples,
    compute_threshold_details,
    compute_finger_state_threshold_details,
    compute_transition_threshold_details,
    compute_segment_threshold_details,
    extract_finger_transitions,
    compute_dtw_distance,
    _read_config,
)
from gesture_capture import (
    RECORDING_DURATION_SEC,
    TARGET_FRAMES,
    MIN_RECORDED_FRAMES,
    MAX_RETRIES_PER_SAMPLE,
    MAX_REGISTRATION_RESTARTS,
    collect_gesture_samples,
    validate_sample_quality,
)
from gesture_auth import evaluate_attempt
from cohort_library import generate_cohort_library, validate_gesture_uniqueness
from utils.normalize import (
    normalize_gesture,
    aspect_for_method,
    POSE_METHOD_LEGACY,
    POSE_METHOD_ASPECT,
)
from utils.face_auth import (
    combine_face_embeddings,
    save_face_embedding,
    SFACE_COSINE_THRESHOLD,
    SFACE_LOOKALIKE_FLOOR,
    YUNET_PATH,
    SFACE_PATH,
)
from utils import attempt_limit as al
from web_backend.camera import CameraError, FACE_LOCK
from utils import liveness as lv

HAND_WAIT_AFTER_CHALLENGE_SEC = lv.HAND_WAIT_AFTER_CHALLENGE_SEC

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACE_REQUIRED_FRAMES = 5          # same burst as gesture_capture / enroll_face
BENCHMARKS = {
    "gesture": "evaluate_biometric_security.py",
    "multimodal": "evaluate_multimodal_security.py",
}


class ApiError(Exception):
    def __init__(self, status, message, code=None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.code = code


def jsonable(value):
    """Convert numpy scalars/arrays (and tuples) into JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if np.isfinite(v) else None
    return value


def _mean_face_embedding(embeddings):
    """Enrolled embedding from the captured frames (utils.face_auth.combine_face_embeddings):
    None when the frames are invalid or do not all show the same person."""
    master, _ = combine_face_embeddings(embeddings)
    return master


def _console_never_fails():
    """The reused CLI functions print symbols such as ✓. On a console or pipe
    that cannot encode them (cp1252) print() would raise inside a worker
    thread and abort a registration, so unencodable characters are replaced."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


class WaveLockService:
    def __init__(self, camera, templates_dir=TEMPLATES_DIR, debug=False, default_camera=None,
                 liveness=True):
        _console_never_fails()
        self.camera = camera
        self.templates_dir = templates_dir
        self.debug = bool(debug)
        self.default_camera = default_camera      # --camera N; None = auto-detect
        self.liveness = bool(liveness)            # challenge before every sign-in (--no-liveness)
        self._lock = threading.RLock()
        self.activity = None          # None | "auth" | "register" | "face"
        self.auth = None
        self.reg = None
        self.face = None
        self.bench = None

    # ------------------------------------------------------------ helpers
    def _claim(self, name):
        with self._lock:
            # A sign-in screen that was left open (e.g. the tab was closed)
            # must not block everything else: an idle sign-in session is
            # ended automatically. A running attempt is never interrupted.
            if (self.activity == "auth" and name != "auth" and self.auth is not None
                    and self.auth["phase"] in ("ready", "result")):
                self.camera.set_identity_preview(None)
                self.auth = None
                self.activity = None
            if self.activity not in (None, name):
                where = {"auth": "A sign-in attempt is running. Wait a few seconds.",
                         "register": "A registration is in progress. Open Register to continue or cancel it.",
                         "face": "A face enrolment is in progress. Wait for it to finish."}
                raise ApiError(409, where[self.activity], "busy")
            self.activity = name

    def _release(self, name):
        with self._lock:
            if self.activity == name:
                self.activity = None

    def _username(self, raw, must_exist=True):
        try:
            name = normalize_username(raw)
        except ValueError as e:
            raise ApiError(400, str(e), "invalid_username")
        if must_exist and name not in list_registered_users(self.templates_dir):
            raise ApiError(404, f"User '{name}' is not registered.", "not_registered")
        return name

    def _start_recording(self, keep_clean_frames):
        try:
            return self.camera.start_recording(RECORDING_DURATION_SEC, keep_clean_frames)
        except CameraError as e:
            raise ApiError(503, str(e), "camera")
        except RuntimeError as e:
            code = "no_hand" if "hand" in str(e).lower() else "busy"
            raise ApiError(409, str(e), code)

    def _start_face_capture(self):
        try:
            return self.camera.start_face_capture(FACE_REQUIRED_FRAMES)
        except CameraError as e:
            raise ApiError(503, str(e), "camera")
        except RuntimeError as e:
            raise ApiError(409, str(e), "busy")

    # ------------------------------------------------------------ overview
    def system(self):
        info = {
            "debug": self.debug,
            "python": sys.version.split()[0],
            "models": {"yunet": os.path.exists(YUNET_PATH), "sface": os.path.exists(SFACE_PATH)},
            "camera": self.camera.status(),
            "activity": self.activity,
            "recording_sec": RECORDING_DURATION_SEC,
            "liveness": self.liveness,
            "samples_required": NUM_REGISTRATION_SAMPLES,
            "min_hand_frames": MIN_RECORDED_FRAMES,
            "max_retries_per_sample": MAX_RETRIES_PER_SAMPLE,
            "max_registration_restarts": MAX_REGISTRATION_RESTARTS,
            "limits": {
                "max_failures": al.MAX_CONSECUTIVE_FAILURES,
                "lockout_base_sec": al.LOCKOUT_BASE_SEC,
                "lockout_max_sec": al.LOCKOUT_MAX_SEC,
            },
            "pose_methods": {"legacy": POSE_METHOD_LEGACY, "aspect": POSE_METHOD_ASPECT},
        }
        if self.debug:
            info["face_thresholds"] = {"match": SFACE_COSINE_THRESHOLD,
                                       "lookalike_floor": SFACE_LOOKALIKE_FLOOR}
            info["templates_dir"] = self.templates_dir
        return info

    def users(self):
        out = []
        for name in list_registered_users(self.templates_dir):
            user_dir = os.path.join(self.templates_dir, name)
            entry = {"name": name, "locked_for": round(al.seconds_locked(name, self.templates_dir), 1)}
            try:
                config = _read_config(name, self.templates_dir)
            except ValueError:
                entry.update({"error": "Profile config is corrupted; re-register this user."})
                out.append(entry)
                continue
            samples = config.get("num_samples")
            if samples is None:
                samples = sum(1 for f in os.listdir(user_dir)
                              if f.startswith("gesture_") and f.endswith(".npy")
                              and not f.endswith("_raw.npy"))
            entry.update({
                "samples": samples,
                "face_enrolled": bool(config.get("face_enrolled"))
                or os.path.exists(os.path.join(user_dir, "face_embedding.npy")),
                "pose_method": config.get("pose_method", POSE_METHOD_LEGACY),
                "raw_recordings": os.path.exists(os.path.join(user_dir, "gesture_1_raw.npy")),
                "path_gate": config.get("trajectory_threshold") is not None,
                "last_attempt": self._last_attempt(name),
            })
            if self.debug:
                entry["thresholds"] = jsonable({
                    "dtw": config.get("threshold"),
                    "finger_state": config.get("finger_state_threshold"),
                    "transition": config.get("transition_threshold"),
                    "segment": config.get("segment_threshold"),
                    "trajectory": config.get("trajectory_threshold"),
                    "anthro_tolerance": (config.get("anthropometric_profile") or {}).get("tolerance"),
                    "consistency": config.get("consistency_score"),
                })
            out.append(entry)
        return out

    def _attempt_log(self, name):
        path = os.path.join(self.templates_dir, al.ATTEMPTS_DIRNAME, f"{name}.log")
        rows = []
        if not os.path.exists(path):
            return rows
        with open(path, "r") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
        return rows

    def _last_attempt(self, name):
        rows = self._attempt_log(name)
        if not rows:
            return None
        last = rows[-1]
        return {"time": last.get("time"), "granted": bool(last.get("granted"))}

    def activity_log(self, limit=50, username=None):
        names = [username] if username else list_registered_users(self.templates_dir)
        rows = []
        for name in names:
            for r in self._attempt_log(name):
                row = {"user": name, "time": r.get("time"), "granted": bool(r.get("granted"))}
                if self.debug:
                    row["reason"] = r.get("reason")
                rows.append(row)
        rows.sort(key=lambda r: r["time"] or "", reverse=True)
        today = datetime.now().date().isoformat()
        todays = [r for r in rows if (r["time"] or "").startswith(today)]
        locked = []
        for name in list_registered_users(self.templates_dir):
            left = al.seconds_locked(name, self.templates_dir)
            if left > 0:
                locked.append({"user": name, "locked_for": round(left, 1)})
        return {
            "entries": rows[:max(1, int(limit))],
            "today": {"total": len(todays),
                      "granted": sum(r["granted"] for r in todays),
                      "denied": sum(not r["granted"] for r in todays)},
            "locked": locked,
        }

    def status(self):
        with self._lock:
            return {
                "camera": self.camera.status(),
                "activity": self.activity,
                "auth": self._auth_state(),
                "register": self._reg_state(),
                "face": self._face_state(),
                "bench": None if self.bench is None else
                {"state": self.bench["state"], "kind": self.bench["kind"], "user": self.bench["user"]},
            }

    # ------------------------------------------------------------ camera
    def camera_start(self, camera_index=None):
        """Start the camera. An explicit index also becomes the default for
        this server session (Settings > Camera); None uses the default."""
        if camera_index is None:
            camera_index = self.default_camera
        else:
            try:
                camera_index = int(camera_index)
            except (TypeError, ValueError):
                raise ApiError(400, "Camera index must be a whole number.")
            if not 0 <= camera_index <= 9:
                raise ApiError(400, "Camera index must be between 0 and 9.")
            self.default_camera = camera_index
        self.camera.start(camera_index)
        self.camera.wait_until_running(timeout=20.0)
        return self.camera.status()

    def camera_stop(self):
        if self.activity is not None:
            raise ApiError(409, "Finish or cancel the current activity before stopping the camera.", "busy")
        self.camera.stop()
        return self.camera.status()

    def cameras(self):
        if self.camera.state in ("starting", "running"):
            raise ApiError(409, "Stop the camera before scanning for devices.", "camera_running")
        from utils.camera_utils import list_available_cameras
        return jsonable(list_available_cameras())

    # ------------------------------------------------------------ sign-in
    def auth_begin(self, raw_username):
        name = self._username(raw_username)
        try:
            profile = load_user_profile(name, self.templates_dir)
        except (FileNotFoundError, ValueError) as e:
            raise ApiError(409, str(e), "profile")
        self._claim("auth")
        with self._lock:
            self.auth = {"username": name, "profile": profile, "phase": "ready",
                         "result": None, "error": None, "attempt": 0}
        self.camera.set_identity_preview(
            profile["face_embedding"] if self.debug else None, name)
        return self._auth_state()

    def auth_record(self):
        a = self.auth
        if a is None or self.activity != "auth":
            raise ApiError(409, "Enter a username to start signing in.", "no_session")
        if a["phase"] == "challenge" and a.get("ready_until"):
            a["go"].set()                     # liveness passed: the user starts the recording
            return self._auth_state()
        if a["phase"] in ("challenge", "recording", "processing"):
            raise ApiError(409, "An attempt is already running.", "busy")
        a["notice"] = None
        left = al.seconds_locked(a["username"], self.templates_dir)
        if left > 0:
            raise ApiError(423, f"Too many failed attempts. Try again in {left:.0f} s.", "locked")
        if self.liveness:
            return self._auth_challenge(a)
        rec = self._start_recording(keep_clean_frames=True)
        with self._lock:
            a["phase"] = "recording"
            a["result"] = None
            a["error"] = None
            a["attempt"] += 1
            a["rec"] = rec
        threading.Thread(target=self._auth_worker, args=(a, rec), daemon=True).start()
        return self._auth_state()

    # ------------------------------------------------------------ liveness
    def _auth_challenge(self, a):
        challenge = lv.new_challenge(a["profile"]["face_embedding"] is not None)
        try:
            job = self.camera.start_challenge(challenge)
        except CameraError as e:
            raise ApiError(503, str(e), "camera")
        except RuntimeError as e:
            raise ApiError(409, str(e), "busy")
        with self._lock:
            a["phase"] = "challenge"
            a["result"] = None
            a["error"] = None
            a["attempt"] += 1
            a["job"] = job
            a["ready_until"] = None
            a["go"] = threading.Event()
            a["notice"] = None
        threading.Thread(target=self._challenge_worker, args=(a, job), daemon=True).start()
        return self._auth_state()

    def _challenge_worker(self, a, job):
        job.done.wait()
        ch = job.challenge
        if job.error and ch.reason == "liveness_cancelled":
            with self._lock:                         # sign-in screen left: nothing to count
                a["phase"] = "ready"
            return
        if ch.state == "passed" and a["profile"]["face_embedding"] is not None:
            # Bind the challenge to the enrolled face: the person who turned
            # their head must be the account owner, not a helper.
            if job.face_frame is None:
                ch.state, ch.reason = "failed", "liveness_no_face"
            else:
                with FACE_LOCK:
                    match, _, _, _, reason = self.camera.face_ops.verify(
                        job.face_frame, a["profile"]["face_embedding"])
                if not match:
                    ch.state, ch.reason = "failed", "liveness_face_mismatch"
        if ch.state != "passed":
            self._liveness_failed(a, ch.reason or "liveness_failed")
            return
        # Passed: the user starts the recording when ready (R / Space / button,
        # i.e. auth_record), as at registration. Not started in time: cancelled,
        # not counted as a failed attempt.
        with self._lock:
            a["ready_until"] = time.time() + lv.RECORD_READY_WINDOW_SEC
        while self.auth is a and not a["go"].wait(0.05):
            if time.time() >= a["ready_until"]:
                with self._lock:
                    if self.auth is a:
                        a["phase"] = "ready"
                        a["ready_until"] = None
                        a["notice"] = "ready_timeout"
                return
        if self.auth is not a:
            return
        end = time.time() + HAND_WAIT_AFTER_CHALLENGE_SEC
        while not self.camera.hand_visible and time.time() < end and self.auth is a:
            time.sleep(0.05)
        if self.auth is not a:
            return
        try:
            rec = self.camera.start_recording(RECORDING_DURATION_SEC, keep_clean_frames=True)
        except (CameraError, RuntimeError):
            self._liveness_failed(a, "liveness_no_hand_for_gesture")
            return
        with self._lock:
            a["phase"] = "recording"
            a["rec"] = rec
            a["ready_until"] = None
        self._auth_worker(a, rec)

    def _liveness_failed(self, a, reason):
        state = al.record_attempt(a["username"], False, self.templates_dir, reason=reason)
        payload = {"granted": False, "too_few_frames": False, "liveness_failed": True,
                   "liveness_reason": reason, "hand_frames": 0,
                   "attempts_left": state["attempts_left"],
                   "locked_for": round(state["locked_for"], 1),
                   "time": datetime.now().isoformat(timespec="seconds")}
        if self.debug:
            payload["details"] = {"failure_reason": reason}
        with self._lock:
            a["result"] = jsonable(payload)
            a["phase"] = "result"

    def _auth_worker(self, a, rec):
        rec.done.wait()
        if rec.error:
            with self._lock:
                a["phase"] = "ready"
                a["error"] = f"Recording stopped ({rec.error})."
            return
        with self._lock:
            a["phase"] = "processing"
        try:
            payload = self._evaluate(a, rec)
        except Exception as e:                       # never leave the UI waiting
            with self._lock:
                a["phase"] = "ready"
                a["error"] = f"Evaluation failed: {e}"
            return
        with self._lock:
            a["result"] = payload
            a["phase"] = "result"

    def _evaluate(self, a, rec):
        name, profile = a["username"], a["profile"]
        # The face models are shared with the camera thread (see FACE_LOCK).
        with FACE_LOCK:
            result = evaluate_attempt(rec.landmarks, rec.clean_frames,
                                      rec.frame_aspect or (4 / 3), profile)
        granted = bool(result["granted"])
        too_few = result["reason"] == "too_few_frames"
        payload = {"granted": granted, "too_few_frames": too_few,
                   "hand_frames": result["frames"], "attempts_left": None,
                   "locked_for": 0.0, "time": datetime.now().isoformat(timespec="seconds")}

        # Same accounting as gesture_auth.py: an attempt with too few hand
        # frames was never evaluated, so it is not counted.
        if not too_few:
            state = al.record_attempt(name, granted, self.templates_dir, reason=result["reason"])
            payload["attempts_left"] = state["attempts_left"]
            payload["locked_for"] = round(state["locked_for"], 1)

        if granted:
            d = result["details"]
            aging = update_templates_if_high_confidence(
                name, result["pose"], result["distance"], profile["threshold"],
                profile["templates"], templates_dir=self.templates_dir,
                anthropometric_match=d["anthropometric_match"],
                anthropometric_confidence=d["anthropometric_confidence"],
                kinematic_match=d["kinematic_match"],
                kinematic_confidence=d["kinematic_confidence"],
                face_match=result["face_match"],
                face_confidence=result["face_confidence"],
                live_raw=result["raw"],
            )
            if aging:
                a["profile"] = load_user_profile(name, self.templates_dir)
                if self.debug:
                    payload["template_updated"] = aging["replaced_template"]

        if self.debug:
            payload["details"] = self._details(result, profile)
        return jsonable(payload)

    @staticmethod
    def _details(result, profile):
        d = result["details"] or {}
        if result["reason"] == "too_few_frames" or "distance" not in d:
            return {"failure_reason": result["reason"]}
        traj_active = bool(d.get("trajectory_active"))
        gates = [
            {"key": "dtw", "label": "G1 Pose DTW", "value": d["distance"],
             "limit": d["threshold"], "pass": d["passes_distance"]},
            {"key": "finger", "label": "G2 Finger states", "value": d["finger_state_mismatch"],
             "limit": d["finger_state_threshold"], "pass": d["passes_finger_state"]},
            {"key": "order", "label": "G3 Finger order", "value": d["transition_dissimilarity"],
             "limit": d["transition_threshold"], "pass": d["passes_transition"]},
            {"key": "segment", "label": "G4 Worst segment", "value": d["segment_max_mismatch"],
             "limit": d["segment_threshold"], "pass": d["passes_segment"]},
            {"key": "path", "label": "Wrist path", "value": d.get("trajectory_distance"),
             "limit": d.get("trajectory_threshold"), "pass": d["passes_trajectory"],
             "active": traj_active},
        ]
        return {
            "failure_reason": d.get("failure_reason"),
            "face": {"enrolled": profile["face_embedding"] is not None,
                     "match": result["face_match"], "score": result["face_score"],
                     "confidence": result["face_confidence"], "reason": result["face_reason"],
                     "threshold": SFACE_COSINE_THRESHOLD, "lookalike_floor": SFACE_LOOKALIKE_FLOOR},
            "gates": gates,
            "macro": {"passed": d["passes_macro"], "fused_score": d["fused_score"],
                      "threshold": d["fusion_acceptance_threshold"],
                      "weights": d["fusion_weights"],
                      "best_template": (result["best_index"] or 0) + 1},
            "hand": {"anthro_match": d["anthropometric_match"],
                     "anthro_confidence": d["anthropometric_confidence"],
                     "anthro_details": d["anthropometric_details"],
                     "kin_match": d["kinematic_match"],
                     "kin_confidence": d["kinematic_confidence"],
                     "kin_details": d["kinematic_details"],
                     "soft_consensus_used": d["soft_consensus_used"],
                     "passed": d["passes_biometric"]},
            "multimodal_fused_score": d["multimodal_fused_score"],
            "hand_frames": result["frames"],
        }

    def auth_reset(self):
        a = self.auth
        if a is not None and a["phase"] in ("result", "ready"):
            with self._lock:
                a["phase"] = "ready"
                a["result"] = None
                a["error"] = None
        return self._auth_state()

    def auth_end(self):
        if self.auth is not None and self.auth["phase"] in ("challenge", "recording"):
            self.camera.cancel_jobs("cancelled")
        self.camera.set_identity_preview(None)
        with self._lock:
            self.auth = None
        self._release("auth")
        return {"ok": True}

    def _auth_state(self):
        a = self.auth
        if a is None:
            return None
        state = {
            "username": a["username"],
            "phase": a["phase"],
            "face_enrolled": a["profile"]["face_embedding"] is not None,
            "templates": len(a["profile"]["templates"]),
            "attempt": a["attempt"],
            "result": a["result"],
            "error": a["error"],
            "locked_for": round(al.seconds_locked(a["username"], self.templates_dir), 1),
            "recording": self.camera.recording_progress() if a["phase"] == "recording" else None,
            "challenge": self.camera.challenge_state() if a["phase"] == "challenge" else None,
            "record_ready": (round(max(0.0, a["ready_until"] - time.time()), 1)
                             if a["phase"] == "challenge" and a.get("ready_until") else None),
            "notice": a.get("notice"),
            "liveness": self.liveness,
        }
        if self.debug:
            state["pose_method"] = a["profile"].get("pose_method")
        return state

    # ------------------------------------------------------------ registration
    def register_begin(self, raw_username, overwrite=False, pose_method="legacy"):
        name = self._username(raw_username, must_exist=False)
        exists = name in list_registered_users(self.templates_dir)
        if exists and not overwrite:
            raise ApiError(409, f"User '{name}' already exists. Confirm to re-register "
                                f"(the old profile is replaced when the new one is saved).",
                           "exists")
        if exists and not al.verified_recently(name, self.templates_dir):
            # Only the owner may replace a profile (code review 2026-10-13).
            raise ApiError(403, al.OWNER_REQUIRED_MESSAGE.format(user=name), "owner_signin_required")
        if pose_method not in ("legacy", "aspect"):
            raise ApiError(400, "pose_method must be 'legacy' or 'aspect'.")
        if self.reg is not None and self.reg["status"] == "active":
            # e.g. a second browser tab: never orphan the running worker
            raise ApiError(409, f"A registration for '{self.reg['username']}' is already in progress. "
                                f"Open Register to continue or cancel it.", "busy")
        self._claim("register")
        with self._lock:
            self.reg = {
                "username": name, "replacing": exists,
                "pose_method": POSE_METHOD_ASPECT if pose_method == "aspect" else POSE_METHOD_LEGACY,
                "step": "face", "status": "active",
                "face_status": "pending", "face_embedding": None, "face_progress": 0,
                "current_sample": 1, "accepted": 0, "awaiting_record": False,
                "phase": "idle", "rejections": 0, "restarts": 0, "events": [],
                "summary": None, "error": None, "message": None,
                "record_event": threading.Event(), "cancel": threading.Event(),
                "rec": None, "fcap": None,
            }
        return self._reg_state()

    def _require_reg(self, step=None):
        r = self.reg
        if r is None or self.activity != "register" or r["status"] != "active":
            raise ApiError(409, "No registration is in progress.", "no_session")
        if step is not None and r["step"] != step:
            raise ApiError(409, f"This action belongs to the '{step}' step.", "wrong_step")
        return r

    def register_face_capture(self):
        r = self._require_reg("face")
        if r["face_status"] == "capturing":
            raise ApiError(409, "Face capture is already running.", "busy")
        fcap = self._start_face_capture()
        with self._lock:
            r["face_status"] = "capturing"
            r["fcap"] = fcap
            r["message"] = None
        threading.Thread(target=self._reg_face_worker, args=(r, fcap), daemon=True).start()
        return self._reg_state()

    def _reg_face_worker(self, r, fcap):
        fcap.done.wait()
        with self._lock:
            r["fcap"] = None
            if r["status"] != "active":
                return
            if fcap.error:
                r["face_status"] = "failed"
                r["message"] = fcap.error
                return
            master = _mean_face_embedding(fcap.embeddings)
            if master is None:
                r["face_status"] = "failed"
                r["message"] = "The face frames were invalid or did not all show the same person. Keep only your face in view and try again."
                return
            r["face_embedding"] = master
            r["face_status"] = "captured"
            r["step"] = "gesture"
        self._start_gesture_worker(r)

    def register_face_skip(self):
        r = self._require_reg("face")
        if r["face_status"] == "capturing":
            raise ApiError(409, "Face capture is running.", "busy")
        with self._lock:
            r["face_status"] = "skipped"
            r["step"] = "gesture"
        self._start_gesture_worker(r)
        return self._reg_state()

    def _start_gesture_worker(self, r):
        threading.Thread(target=self._reg_gesture_worker, args=(r,), daemon=True).start()

    def _reg_gesture_worker(self, r):
        last = {"n": 0}

        def event(kind, **kw):
            with self._lock:
                r["events"].append(dict(kw, type=kind, sample=r["current_sample"],
                                        time=time.time()))
                r["events"] = r["events"][-40:]

        def record_fn(n):
            # collect_gesture_samples restarted the set: tell the UI.
            if n < last["n"]:
                with self._lock:
                    r["restarts"] += 1
                    r["accepted"] = 0
                event("restart", restart=r["restarts"], max_restarts=MAX_REGISTRATION_RESTARTS)
            if n != last["n"]:
                with self._lock:
                    r["rejections"] = 0
            last["n"] = n
            while True:
                with self._lock:
                    r["current_sample"] = n
                    r["awaiting_record"] = True
                    r["phase"] = "idle"
                while not r["record_event"].wait(0.2):
                    if r["cancel"].is_set():
                        return None
                r["record_event"].clear()
                rec = r["rec"]
                rec.done.wait()
                if r["cancel"].is_set():
                    return None
                if rec.error:
                    event("error", message=f"Recording stopped ({rec.error}).")
                    continue
                if len(rec.landmarks) < MIN_RECORDED_FRAMES:
                    event("too_few_frames", frames=len(rec.landmarks),
                          message=f"Only {len(rec.landmarks)} hand frames. Keep your hand in view and record again.")
                    continue
                raw = np.asarray(rec.landmarks, dtype=np.float64)
                aspect = rec.frame_aspect or (4 / 3)
                pose = normalize_gesture(matching_recording(raw, NEW_PROFILE_TEMPORAL_TRIM),
                                         TARGET_FRAMES,
                                         frame_aspect=aspect_for_method(r["pose_method"], aspect))
                with self._lock:
                    r["phase"] = "checking"
                return pose, raw, aspect

        def validate_fn(sample, accepted):
            ok, msg = validate_sample_quality(sample, accepted)
            with self._lock:
                if ok:
                    r["accepted"] = len(accepted) + 1
                else:
                    r["rejections"] += 1
            event("accepted" if ok else "rejected", message=msg,
                  rejections=r["rejections"], max_retries=MAX_RETRIES_PER_SAMPLE)
            return ok, msg

        try:
            collected = collect_gesture_samples(record_fn, validate_fn)
        except Exception as e:
            self._reg_finish(r, "failed", f"Registration failed: {e}")
            return
        if collected is None:
            if r["cancel"].is_set():
                return                                   # register_cancel() finished it
            self._reg_finish(r, "failed",
                             "The recordings stayed inconsistent after "
                             f"{MAX_REGISTRATION_RESTARTS} restarts. Practise the gesture "
                             "until it feels the same each time, then try again. "
                             "Any existing profile is unchanged.")
            return
        with self._lock:
            r["phase"] = "saving"
            r["step"] = "review"
        try:
            summary = self._save_registration(r, *collected)
        except (OSError, ValueError) as e:
            self._reg_finish(r, "failed", f"Could not save the registration: {e}. "
                                          f"Any existing profile is unchanged.")
            return
        with self._lock:
            r["summary"] = summary
        self._reg_finish(r, "done", None)

    def _save_registration(self, r, samples, raw_samples, frame_aspect):
        """Same calibration report and save as gesture_capture.main()."""
        name = r["username"]
        threshold, pairwise = compute_threshold_from_samples(samples)
        details = compute_threshold_details(pairwise)
        finger = compute_finger_state_threshold_details(samples)
        transition = compute_transition_threshold_details(samples)
        segment = compute_segment_threshold_details(samples)
        no_transitions = all(not extract_finger_transitions(s) for s in samples)
        uniq = validate_gesture_uniqueness(samples, generate_cohort_library(),
                                           intra_user_max=max(pairwise),
                                           compute_dtw_fn=compute_dtw_distance)
        user_dir = save_registration(
            name, samples, threshold, pairwise, templates_dir=self.templates_dir,
            raw_samples=raw_samples, frame_aspect=frame_aspect,
            face_embedding=r["face_embedding"], pose_method=r["pose_method"],
            temporal_trim=NEW_PROFILE_TEMPORAL_TRIM)
        if r["replacing"]:
            al.consume_verification(name, self.templates_dir)
        config = _read_config(name, self.templates_dir)
        if r["face_embedding"] is not None:
            face = "enrolled"
        elif config.get("face_enrolled"):
            face = "kept"
        else:
            face = "skipped"
        warnings = []
        if no_transitions:
            warnings.append("Your gesture never raises or folds a finger, so the finger-order "
                            "gate cannot add security. Consider a finger sequence such as 1-2-4-3.")
        if not uniq["is_unique"]:
            warnings.append(f"{uniq['warning']} (impostor ratio {uniq['ratio']:.2f}x, minimum 1.50x)")
        return jsonable({
            "username": name,
            "samples": len(samples),
            "face": face,
            "pose_method": r["pose_method"],
            "pairwise_distances": [round(d, 4) for d in pairwise],
            "max_variation": max(pairwise),
            "consistency": details["consistency_score"],
            "thresholds": {
                "dtw": config["threshold"],
                "finger_state": finger["threshold"],
                "transition": transition["threshold"],
                "segment": segment["threshold"],
                "trajectory": config.get("trajectory_threshold"),
                "anthro_tolerance": config["anthropometric_profile"]["tolerance"],
            },
            "uniqueness_ratio": uniq["ratio"] if np.isfinite(uniq["ratio"]) else None,
            "unique": uniq["is_unique"],
            "warnings": warnings,
            "location": user_dir,
        })

    def _reg_finish(self, r, status, message):
        with self._lock:
            r["status"] = status
            r["phase"] = "idle"
            r["awaiting_record"] = False
            if message:
                r["message"] = message
                r["error"] = message
            if status == "done":
                r["step"] = "review"
        self._release("register")

    def register_record(self):
        r = self._require_reg("gesture")
        if not r["awaiting_record"]:
            raise ApiError(409, "Wait until the current sample has been checked.", "busy")
        rec = self._start_recording(keep_clean_frames=False)
        with self._lock:
            r["rec"] = rec
            r["awaiting_record"] = False
            r["phase"] = "recording"
            r["message"] = None
        r["record_event"].set()
        return self._reg_state()

    def register_cancel(self):
        r = self.reg
        if r is None:
            return {"ok": True}
        r["cancel"].set()
        self.camera.cancel_jobs("cancelled")
        with self._lock:
            if r["status"] == "active":
                r["status"] = "cancelled"
                r["message"] = "Registration cancelled. Any existing profile is unchanged."
        self._release("register")
        return self._reg_state()

    def register_close(self):
        r = self.reg
        if r is not None and r["status"] == "active":
            self.register_cancel()
        with self._lock:
            self.reg = None
        return {"ok": True}

    def _reg_state(self):
        r = self.reg
        if r is None:
            return None
        rec = r.get("rec")
        fcap = r.get("fcap")
        return jsonable({
            "username": r["username"], "replacing": r["replacing"],
            "pose_method": r["pose_method"], "step": r["step"], "status": r["status"],
            "face_status": r["face_status"],
            "face_progress": len(fcap.embeddings) if fcap is not None else
            (FACE_REQUIRED_FRAMES if r["face_status"] == "captured" else 0),
            "face_required": FACE_REQUIRED_FRAMES,
            "samples_required": NUM_REGISTRATION_SAMPLES,
            "current_sample": r["current_sample"], "accepted": r["accepted"],
            "awaiting_record": r["awaiting_record"], "phase": r["phase"],
            "rejections": r["rejections"], "max_retries": MAX_RETRIES_PER_SAMPLE,
            "restarts": r["restarts"], "max_restarts": MAX_REGISTRATION_RESTARTS,
            "events": [{k: v for k, v in e.items()} for e in r["events"][-12:]],
            "recording": self.camera.recording_progress()
            if r["phase"] == "recording" and rec is not None else None,
            "summary": r["summary"], "message": r["message"],
        })

    # ------------------------------------------------------------ face enrolment
    def face_enroll_begin(self, raw_username):
        name = self._username(raw_username)
        if not al.verified_recently(name, self.templates_dir):
            # Adding or replacing the face of a profile is a profile change.
            raise ApiError(403, al.OWNER_REQUIRED_MESSAGE.format(user=name), "owner_signin_required")
        self._claim("face")
        try:
            fcap = self._start_face_capture()
        except ApiError:
            self._release("face")
            raise
        with self._lock:
            self.face = {"username": name, "status": "capturing", "fcap": fcap, "message": None}
        threading.Thread(target=self._face_worker, args=(self.face, fcap), daemon=True).start()
        return self._face_state()

    def _face_worker(self, f, fcap):
        fcap.done.wait()
        try:
            if fcap.error:
                status, message = "failed", fcap.error
            else:
                master = _mean_face_embedding(fcap.embeddings)
                if master is None:
                    status, message = "failed", "The face frames were invalid or did not all show the same person. Keep only your face in view and try again."
                else:
                    save_face_embedding(f["username"], master, self.templates_dir)
                    set_face_enrolled_flag(f["username"], self.templates_dir)
                    al.consume_verification(f["username"], self.templates_dir)
                    status, message = "done", "Face enrolled. It is checked at every sign-in."
        except (OSError, ValueError) as e:
            status, message = "failed", f"Could not save the face profile: {e}"
        with self._lock:
            f["status"] = status
            f["message"] = message
            f["fcap"] = None
        self._release("face")

    def face_close(self):
        f = self.face
        if f is not None and f["status"] == "capturing":
            self.camera.cancel_jobs("cancelled")
        with self._lock:
            self.face = None
        self._release("face")
        return {"ok": True}

    def _face_state(self):
        f = self.face
        if f is None:
            return None
        fcap = f.get("fcap")
        return {"username": f["username"], "status": f["status"], "message": f["message"],
                "progress": len(fcap.embeddings) if fcap is not None else
                (FACE_REQUIRED_FRAMES if f["status"] == "done" else 0),
                "required": FACE_REQUIRED_FRAMES}

    # ------------------------------------------------------------ benchmarks
    def benchmark_run(self, raw_username, kind):
        if not self.debug:
            raise ApiError(403, "Benchmarks are available in developer mode (start with --debug).",
                           "debug_only")
        if kind not in BENCHMARKS:
            raise ApiError(400, "kind must be 'gesture' or 'multimodal'.")
        name = self._username(raw_username)
        with self._lock:
            if self.bench is not None and self.bench["state"] == "running":
                raise ApiError(409, "A benchmark is already running.", "busy")
            self.bench = {"state": "running", "kind": kind, "user": name, "output": "",
                          "returncode": None, "started": time.time()}
        bench = self.bench
        cmd = [sys.executable, os.path.join(PROJECT_DIR, BENCHMARKS[kind]),
               "--user", name, "--templates-dir", self.templates_dir]

        def run():
            env = dict(os.environ, PYTHONIOENCODING="utf-8")
            try:
                out = subprocess.run(cmd, cwd=PROJECT_DIR, capture_output=True, text=True,
                                     encoding="utf-8", errors="replace", timeout=1200, env=env)
                text = out.stdout + (("\n" + out.stderr) if out.returncode else "")
                code = out.returncode
            except (OSError, subprocess.TimeoutExpired) as e:
                text, code = f"Benchmark could not run: {e}", -1
            with self._lock:
                bench.update(output=text, returncode=code,
                             state="done" if code == 0 else "failed",
                             seconds=round(time.time() - bench["started"], 1))

        threading.Thread(target=run, daemon=True).start()
        return self.benchmark_status()

    def benchmark_status(self):
        if not self.debug:
            raise ApiError(403, "Benchmarks are available in developer mode (start with --debug).",
                           "debug_only")
        b = self.bench
        if b is None:
            return None
        return {k: b.get(k) for k in ("state", "kind", "user", "output", "returncode", "seconds")}

    # ------------------------------------------------------------ shutdown
    def shutdown(self):
        if self.reg is not None:
            self.reg["cancel"].set()
        self.camera.cancel_jobs("server stopped")
        self.camera.stop()
