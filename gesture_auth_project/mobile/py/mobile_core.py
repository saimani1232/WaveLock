"""
WaveLock mobile core: the desktop Python modules, driven by the mobile app.

The mobile app (mobile/) runs this module in the browser with Pyodide. The
camera, MediaPipe Hands (web build), YuNet and SFace run in JavaScript; every
decision is made here by the SAME code as the desktop programs:
gesture_compare (gates, calibration, aging), gesture_auth.evaluate_attempt,
gesture_capture.validate_sample_quality, utils.liveness, utils.attempt_limit,
utils.face_auth and the result formatting of web_backend.service. The
JavaScript side only orchestrates (screens, timers, recordings).

Every public function takes and returns JSON-safe values; errors that the
UI shows are returned as {"__error__": {"status", "message", "code"}}.
Profiles live in TEMPLATES_DIR, which the app keeps in the browser's own
storage on the device (IndexedDB); nothing is sent anywhere.
"""

import os
import types

import numpy as np

import gesture_compare as gc
import gesture_auth as ga
import gesture_capture as gcap
from utils import attempt_limit as al
from utils import face_auth as fa
from utils import liveness as lv
from utils.normalize import normalize_gesture, aspect_for_method
from web_backend import service as service_mod
from web_backend.service import ApiError, WaveLockService, jsonable


class _NoCamera:
    """The service methods used here never touch the camera; the app's
    camera state is reported by JavaScript."""

    def status(self):
        return {"state": "stopped"}

    def set_identity_preview(self, *args, **kwargs):
        pass

    def cancel_jobs(self, *args, **kwargs):
        pass


class Core:
    def __init__(self, templates_dir, debug=False, liveness=True):
        os.makedirs(templates_dir, exist_ok=True)
        self.templates_dir = templates_dir
        self.svc = WaveLockService(_NoCamera(), templates_dir=templates_dir,
                                   debug=debug, liveness=liveness)
        self.profiles = {}          # username -> load_user_profile() of an open sign-in
        self.challenge = None       # utils.liveness.Challenge being answered
        self.reg = None             # registration being recorded

    # ------------------------------------------------------------ settings / overview
    def set_options(self, debug=None, liveness=None):
        if debug is not None:
            self.svc.debug = bool(debug)
        if liveness is not None:
            self.svc.liveness = bool(liveness)
        return {"debug": self.svc.debug, "liveness": self.svc.liveness}

    def constants(self):
        return {
            "recording_sec": gcap.RECORDING_DURATION_SEC,
            "min_hand_frames": gcap.MIN_RECORDED_FRAMES,
            "samples_required": gc.NUM_REGISTRATION_SAMPLES,
            "max_retries_per_sample": gcap.MAX_RETRIES_PER_SAMPLE,
            "max_registration_restarts": gcap.MAX_REGISTRATION_RESTARTS,
            "face_required_frames": service_mod.FACE_REQUIRED_FRAMES,
            "face_sample_gap_sec": fa.FACE_SAMPLE_GAP_SEC,
            "record_ready_window_sec": lv.RECORD_READY_WINDOW_SEC,
            "hand_wait_after_challenge_sec": lv.HAND_WAIT_AFTER_CHALLENGE_SEC,
            "challenge_timeout_sec": lv.CHALLENGE_TIMEOUT_SEC,
        }

    def system(self):
        return jsonable(self.svc.system())

    def users(self):
        return jsonable(self.svc.users())

    def activity(self, limit=50, user=None):
        return jsonable(self.svc.activity_log(limit, user or None))

    def locked_for(self, username):
        return round(al.seconds_locked(username, self.templates_dir), 1)

    # ------------------------------------------------------------ sign-in
    def auth_begin(self, raw_username):
        name = self.svc._username(raw_username)
        try:
            profile = gc.load_user_profile(name, self.templates_dir)
        except (FileNotFoundError, ValueError) as e:
            raise ApiError(409, str(e), "profile")
        self.profiles[name] = profile
        return {"username": name,
                "face_enrolled": profile["face_embedding"] is not None,
                "templates": len(profile["templates"]),
                "pose_method": profile.get("pose_method"),
                "check_level": profile.get("check_level")}

    def auth_end(self, username):
        self.profiles.pop(username, None)
        self.challenge = None
        return {"ok": True}

    def _profile(self, username):
        if username not in self.profiles:
            raise ApiError(409, "Enter a username to start signing in.", "no_session")
        return self.profiles[username]

    def challenge_new(self, username, now):
        face = self._profile(username)["face_embedding"] is not None
        self.challenge = lv.new_challenge(face, now=now)
        return self.challenge.public(now)

    def challenge_feed(self, hand=None, frame_aspect=4 / 3, face_row=None,
                       face_seen=False, now=None):
        """One camera frame: hand landmarks (21 x 3, mirrored like the desktop
        frames) and, when the face detector ran on this frame, its YuNet row
        (or None for no face). Returns the state and the frame's |yaw|."""
        ch = self.challenge
        if ch is None:
            return None
        if ch.state == "running":
            lm = None if hand is None else np.asarray(hand, dtype=np.float64)
            ch.feed_hand(lv.count_shown_fingers(lm, frame_aspect), now)
            yaw = None
            if face_seen:
                yaw = lv.yaw_proxy(None if face_row is None else np.asarray(face_row, np.float64))
                ch.feed_face(yaw, now)
            ch.tick(now)
        else:
            yaw = None
            if face_seen and face_row is not None:
                yaw = lv.yaw_proxy(np.asarray(face_row, np.float64))
        return {"public": ch.public(now), "abs_yaw": None if yaw is None else abs(yaw)}

    def challenge_cancel(self):
        if self.challenge is not None and self.challenge.state == "running":
            self.challenge.fail("liveness_cancelled")
        self.challenge = None
        return {"ok": True}

    def challenge_bind(self, username, embedding=None):
        """After a passed challenge: the most frontal face seen during it must
        be the enrolled face (web_backend.service._challenge_worker)."""
        profile = self._profile(username)
        if profile["face_embedding"] is None:
            return {"passed": True, "reason": None}
        if embedding is None:
            return {"passed": False, "reason": "liveness_no_face"}
        score = fa.cosine_score(embedding, profile["face_embedding"])
        match, _, _ = fa.classify_face_score(score)
        return {"passed": bool(match), "reason": None if match else "liveness_face_mismatch"}

    def liveness_failed(self, username, reason):
        a = {"username": username, "profile": self._profile(username)}
        self.svc._liveness_failed(a, reason)
        return a["result"]

    def face_sample_indices(self, n_frames):
        return [int(i) for i in fa.face_sample_indices(int(n_frames))] if n_frames else []

    def evaluate(self, username, landmarks, frame_aspect, face_samples=None):
        """One recorded attempt. face_samples: the SFace embeddings (or None
        for no face) of the frames face_sample_indices() chose, computed in
        the browser; the decision is utils.face_auth's."""
        profile = self._profile(username)
        enrolled = profile["face_embedding"]

        def verify_over_samples(frames, enrolled_embedding):
            samples = []
            for emb in frames:
                if emb is None:
                    samples.append(None)
                    continue
                score = fa.cosine_score(emb, enrolled_embedding)
                match, _, _ = fa.classify_face_score(score)
                samples.append((match, score))
            return fa.decide_face_over_samples(samples)

        rec = types.SimpleNamespace(
            landmarks=[np.asarray(f, dtype=np.float64) for f in landmarks],
            clean_frames=list(face_samples or []) if enrolled is not None else [],
            frame_aspect=frame_aspect,
        )
        a = {"username": username, "profile": profile}
        original = ga.verify_face_over_frames
        ga.verify_face_over_frames = verify_over_samples
        try:
            payload = self.svc._evaluate(a, rec)
        finally:
            ga.verify_face_over_frames = original
        self.profiles[username] = a["profile"]          # reloaded after template aging
        return payload

    # ------------------------------------------------------------ registration
    def reg_begin(self, raw_username, overwrite=False, pose_method="legacy",
                  check_level=gc.CHECK_LEVEL_STANDARD):
        """Same checks and messages as the web app (owner sign-in for an
        existing profile, valid options)."""
        try:
            state = self.svc.register_begin(raw_username, overwrite, pose_method, check_level)
            r = self.svc.reg
        finally:
            self.svc.reg = None
            self.svc._release("register")
        self.reg = {"username": r["username"], "replacing": r["replacing"],
                    "pose_method": r["pose_method"], "check_level": r["check_level"],
                    "face_embedding": None, "poses": [], "raws": [], "aspect": None}
        return {"username": state["username"], "replacing": state["replacing"],
                "pose_method": state["pose_method"]}

    def _reg(self):
        if self.reg is None:
            raise ApiError(409, "No registration is in progress.", "no_session")
        return self.reg

    def reg_face(self, embeddings):
        r = self._reg()
        master, _ = fa.combine_face_embeddings(embeddings)
        if master is None:
            return {"ok": False, "message": "The face frames were invalid or did not all show "
                                            "the same person. Keep only your face in view and try again."}
        r["face_embedding"] = master
        return {"ok": True}

    def reg_sample(self, landmarks, frame_aspect):
        """One recorded sample: the quality gate of gesture_capture."""
        r = self._reg()
        if len(landmarks) < gcap.MIN_RECORDED_FRAMES:
            return {"too_few_frames": True, "frames": len(landmarks)}
        raw = np.asarray(landmarks, dtype=np.float64)
        aspect = frame_aspect or (4 / 3)
        pose = normalize_gesture(gc.matching_recording(raw, gc.NEW_PROFILE_TEMPORAL_TRIM),
                                 gcap.TARGET_FRAMES,
                                 frame_aspect=aspect_for_method(r["pose_method"], aspect))
        ok, message = gcap.validate_sample_quality(pose, r["poses"])
        if ok:
            r["poses"].append(pose)
            r["raws"].append(raw)
            if r["aspect"] is None:
                r["aspect"] = aspect
        return {"ok": bool(ok), "message": message, "accepted": len(r["poses"])}

    def reg_restart(self):
        r = self._reg()
        r["poses"], r["raws"], r["aspect"] = [], [], None
        return {"ok": True}

    def reg_save(self):
        r = self._reg()
        summary = self.svc._save_registration(r, r["poses"], r["raws"], r["aspect"])
        self.reg = None
        summary.pop("location", None)
        return summary

    def reg_cancel(self):
        self.reg = None
        return {"ok": True}

    # ------------------------------------------------------------ face enrolment (Users page)
    def face_enroll_check(self, raw_username):
        name = self.svc._username(raw_username)
        if not al.verified_recently(name, self.templates_dir):
            raise ApiError(403, al.OWNER_REQUIRED_MESSAGE.format(user=name), "owner_signin_required")
        return {"username": name}

    def face_enroll_save(self, username, embeddings):
        master, _ = fa.combine_face_embeddings(embeddings)
        if master is None:
            return {"status": "failed", "message": "The face frames were invalid or did not all "
                    "show the same person. Keep only your face in view and try again."}
        fa.save_face_embedding(username, master, self.templates_dir)
        gc.set_face_enrolled_flag(username, self.templates_dir)
        al.consume_verification(username, self.templates_dir)
        return {"status": "done", "message": "Face enrolled. It is checked at every sign-in."}


_core = None


def init(templates_dir, debug=False, liveness=True):
    global _core
    _core = Core(templates_dir, debug=debug, liveness=liveness)
    return {"ok": True}


def call(method, kwargs):
    """Entry point for JavaScript: call(name, {...}) -> result or __error__."""
    if _core is None:
        raise RuntimeError("mobile_core.init() was not called")
    if method.startswith("_") or not hasattr(Core, method):
        raise ValueError(f"unknown method {method}")
    try:
        return jsonable(getattr(_core, method)(**(kwargs or {})))
    except ApiError as e:
        return {"__error__": {"status": e.status, "message": e.message, "code": e.code}}
