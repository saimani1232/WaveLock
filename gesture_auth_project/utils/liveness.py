"""
Liveness challenge (presentation-attack countermeasure) for WaveLock sign-in.

A printed photo or a pre-recorded video cannot react to an instruction that
is chosen at random at the moment of sign-in. Before the gesture is recorded
the user completes two short steps:

  1. "Show N fingers"  - N drawn at random from 1..4; all five fingers count,
     so five fingers do not pass a request for four. Index to pinky use the
     matcher's straight-finger rule; the thumb counts when its tip reaches
     sideways beyond the index knuckle by more than THUMB_OUT_MIN palm widths
     (on the 2710 real enrolment frames this measure is bimodal: folded about
     -0.9..0.1, spread 0.6..1.3, 2 % in between). Held for HOLD_SEC.
  2. With a face enrolled: "Turn your head toward the arrow" (left or right at
     random). The cue is 3-D: when a real head turns, the nose tip moves
     sideways relative to the eye midpoint (yaw_proxy). A flat photo tilted
     in any direction keeps that ratio almost constant.
     Without a face enrolled: a second finger count (different from the first).

Both steps must be done within CHALLENGE_TIMEOUT_SEC. Chance of a random
video passing: 1/4 (count) x 1/2 (direction) = 1/8 with a face enrolled,
1/4 x 1/3 = 1/12 without, before the gesture, hand and face checks.

Limits: a live attacker who copies the gesture AND shows their own face and
hand passes the challenge itself (the identity checks stop them); a
real-time deep-fake video could follow the instructions. The thresholds are
geometric estimates (see HEAD_TURN_MIN) and have not been measured on many
people (semester-2 limitation: no new user recordings).

This module is pure logic (no camera, no threads), driven by feed_hand /
feed_face / tick, so it can be tested exactly.
"""

import random
import time

import numpy as np

from utils.normalize import normalize_spatial

CHALLENGE_TIMEOUT_SEC = 12.0
# After a passed challenge the user starts the gesture recording (R / Space /
# button) when ready, as at registration, so the gesture starts at the same
# moment of the recording window as the enrolment samples. Not started within
# RECORD_READY_WINDOW_SEC: the attempt is cancelled (not counted as a failure).
RECORD_READY_WINDOW_SEC = 20.0
HAND_WAIT_AFTER_CHALLENGE_SEC = 3.0   # after the start: time to raise the hand
THUMB_OUT_MIN = 0.40          # thumb tip beyond the index knuckle, in palm widths
HOLD_SEC = 0.6                # finger count held continuously this long
# Yaw proxy change needed for a head turn. Geometry: the nose tip sits about
# 0.55 inter-eye distances in front of the eye plane, so a ~15 degree turn
# moves it about 0.14 eye-distances sideways; 0.10 accepts a modest turn and
# stays well above the jitter of the 5-point detector on a frontal face.
HEAD_TURN_MIN = 0.10
HEAD_HOLD_SAMPLES = 2         # consecutive face samples beyond the threshold
BASELINE_SAMPLES = 2          # frontal samples taken at the start of the head step

FINGER_JOINTS = {             # MCP, PIP, TIP (MediaPipe indices), thumb excluded
    "index": (5, 6, 8),
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20),
}
EXTENDED_ANGLE_DEG = 150.0    # same rule as gesture_compare (straight finger)

_rng = random.SystemRandom()


def _angle(a, b, c):
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-9:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))))


def count_extended_fingers(landmarks):
    """Number of extended fingers among index, middle, ring and pinky (0-4)."""
    if landmarks is None:
        return None
    frame = normalize_spatial(np.asarray(landmarks, dtype=np.float64))
    wrist = frame[0]
    count = 0
    for mcp, pip, tip in FINGER_JOINTS.values():
        straight = _angle(frame[mcp], frame[pip], frame[tip]) >= EXTENDED_ANGLE_DEG
        reaches_out = np.linalg.norm(frame[tip] - wrist) > np.linalg.norm(frame[pip] - wrist) * 1.02
        count += int(straight and reaches_out)
    return count


def thumb_extended(landmarks, frame_aspect=4 / 3):
    """True when the thumb tip reaches sideways beyond the index knuckle by
    more than THUMB_OUT_MIN palm widths (axis: pinky knuckle -> index knuckle,
    so it works for either hand and any in-plane rotation)."""
    if landmarks is None:
        return None
    p = np.asarray(landmarks, dtype=np.float64)[:, :2] * [float(frame_aspect), 1.0]
    across = p[5] - p[17]
    width = float(np.linalg.norm(across))
    if width < 1e-6:
        return False
    return float(np.dot(p[4] - p[5], across / width) / width) > THUMB_OUT_MIN


def count_shown_fingers(landmarks, frame_aspect=4 / 3):
    """Fingers shown for the liveness challenge: index..pinky plus the thumb (0-5)."""
    n = count_extended_fingers(landmarks)
    if n is None:
        return None
    return n + int(thumb_extended(landmarks, frame_aspect))


def yaw_proxy(face_row):
    """
    Horizontal nose offset from the eye midpoint, in inter-eye distances.
    face_row is a YuNet detection: x, y, w, h, right eye (4,5), left eye
    (6,7), nose tip (8,9), mouth corners, score. Negative = nose toward the
    left of the (mirrored) picture.
    """
    if face_row is None:
        return None
    r = np.asarray(face_row, dtype=np.float64).ravel()
    if r.size < 10:
        return None
    eye_r, eye_l, nose = r[4:6], r[6:8], r[8:10]
    dist = float(np.linalg.norm(eye_l - eye_r))
    if dist < 1e-6:
        return None
    return float((nose[0] - (eye_r[0] + eye_l[0]) / 2.0) / dist)


class Challenge:
    """One liveness challenge: a list of steps completed in order."""

    def __init__(self, steps, now=None, timeout=CHALLENGE_TIMEOUT_SEC):
        self.steps = steps
        self.index = 0
        self.state = "running"            # running | passed | failed
        self.reason = None
        self.started = time.time() if now is None else now
        self.timeout = timeout
        self._hold_since = None
        self._yaw_base = []
        self._turn_hits = 0
        self.last_count = None
        self.last_yaw_delta = None

    # ---------------------------------------------------------------- input
    @property
    def step(self):
        return self.steps[self.index] if self.state == "running" else None

    def feed_hand(self, count, now=None):
        now = time.time() if now is None else now
        st = self.step
        self.last_count = count
        if st is None or st["kind"] != "fingers":
            return
        if count == st["count"]:
            if self._hold_since is None:
                self._hold_since = now
            if now - self._hold_since >= HOLD_SEC:
                self._advance(now)
        else:
            self._hold_since = None

    def feed_face(self, yaw, now=None):
        now = time.time() if now is None else now
        st = self.step
        if st is None or st["kind"] != "head":
            return
        if yaw is None:                   # face lost: start counting again
            self._turn_hits = 0
            return
        if len(self._yaw_base) < BASELINE_SAMPLES:
            self._yaw_base.append(yaw)
            return
        delta = yaw - float(np.median(self._yaw_base))
        self.last_yaw_delta = delta
        wanted = -1.0 if st["direction"] == "left" else 1.0
        if delta * wanted >= HEAD_TURN_MIN:
            self._turn_hits += 1
            if self._turn_hits >= HEAD_HOLD_SAMPLES:
                self._advance(now)
        else:
            self._turn_hits = 0

    def tick(self, now=None):
        now = time.time() if now is None else now
        if self.state == "running" and now - self.started > self.timeout:
            self.state = "failed"
            st = self.steps[self.index]
            self.reason = "liveness_timeout_" + ("fingers" if st["kind"] == "fingers" else "head_turn")

    def _advance(self, now):
        self.index += 1
        self._hold_since = None
        self._turn_hits = 0
        self._yaw_base = []
        if self.index >= len(self.steps):
            self.state = "passed"

    def fail(self, reason):
        if self.state == "running":
            self.state = "failed"
            self.reason = reason

    # ---------------------------------------------------------------- output
    def public(self, now=None):
        """JSON-safe description for the UI (no thresholds or measurements)."""
        now = time.time() if now is None else now
        return {
            "state": self.state,
            "reason": self.reason,
            "index": self.index,
            "steps": [dict(s) for s in self.steps],
            "remaining": round(max(0.0, self.timeout - (now - self.started)), 1),
        }


def new_challenge(face_enrolled, now=None, rng=None):
    """A random two-step challenge (see module docstring)."""
    rng = rng or _rng
    first = rng.randint(1, 4)
    steps = [{"kind": "fingers", "count": first}]
    if face_enrolled:
        steps.append({"kind": "head", "direction": rng.choice(["left", "right"])})
    else:
        steps.append({"kind": "fingers", "count": rng.choice([n for n in range(1, 5) if n != first])})
    return Challenge(steps, now=now, timeout=CHALLENGE_TIMEOUT_SEC)   # read at call time
