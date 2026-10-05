"""
Gesture Comparison Module — DTW-Based Gesture Matching

Compares gesture templates using Dynamic Time Warping (DTW).
Supports multi-sample registration where each user has multiple
gesture recordings for improved accuracy.

Distance interpretation:
    - Lower score  = more similar gestures (same person, same gesture)
    - Higher score = less similar gestures (different gesture or person)

Uses `dtaidistance` for fast, C-optimized multi-dimensional DTW.

Can be used as:
    1. An imported module by gesture_auth.py
    2. A standalone script to compare saved templates:
       python gesture_compare.py
"""

import os
import re
import json
import shutil
import tempfile

import numpy as np
from dtaidistance import dtw_ndim

from utils.anthropometrics import (
    extract_gesture_anthropometric_signature,
    calibrate_anthropometric_profile,
    compare_anthropometric_signatures,
    DEFAULT_ANTHRO_TOLERANCE,
)
from utils.kinematics import (
    extract_kinematic_signature,
    calibrate_kinematic_profile,
    compare_kinematic_signatures,
    prepare_kinematic_sequence,
    DEFAULT_KINEMATIC_TOLERANCE,
    KINEMATIC_METHOD_SMOOTHED,
)
from utils.face_auth import load_face_embedding
from utils.normalize import extract_wrist_trajectory


# ============================================================================
# CONFIGURATION
# ============================================================================

# Resolve templates directory relative to THIS script's location
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(_SCRIPT_DIR, "templates")

# Number of gesture samples recorded during registration.
# Five samples gives 10 intra-user comparisons, which makes threshold
# calibration more stable than the previous 3-sample setup.
NUM_REGISTRATION_SAMPLES = 5

# Fallback threshold used ONLY for old single-sample registrations
# that don't have a computed threshold. New registrations auto-compute
# their own threshold from the recorded samples.
DEFAULT_THRESHOLD = 5.0

# Minimum threshold floor — prevents unreasonably strict thresholds
# when a user is very consistent during registration.
MIN_THRESHOLD = 2.0

# Legacy multiplier retained for old configs and comparison metadata.
THRESHOLD_MULTIPLIER = 1.5

# Statistical threshold calibration settings.
# The final threshold is based on the user's genuine registration-distance
# distribution, then capped so one inconsistent sample cannot make the
# account overly easy to unlock
THRESHOLD_STD_FACTOR = 2.0
THRESHOLD_PERCENTILE = 90
THRESHOLD_SAFETY_MARGIN = 1.15
THRESHOLD_MAX_MARGIN = 1.25

THRESHOLD_METHOD = "statistical_v2"

# Finger-state security gate. DTW checks whether the motion is similar;
# this gate checks whether the same fingers are extended over time.
DEFAULT_FINGER_STATE_THRESHOLD = 0.28
MIN_FINGER_STATE_THRESHOLD = 0.12
MAX_FINGER_STATE_THRESHOLD = 0.30
FINGER_STATE_MARGIN = 0.08
FINGER_EXTENDED_ANGLE_DEG = 150.0
FINGER_TIP_DISTANCE_MARGIN = 1.02
FINGER_STATE_METHOD = "finger_state_sequence_v1"

FINGER_JOINTS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}

# Finger transition sequence gate — captures the ORDER in which fingers
# change state. Critical for sequential gestures like "raise index, then
# middle, then ring, then pinky" where the motion path is similar but
# the finger order defines the password.
TRANSITION_SMOOTHING_WINDOW = 5
DEFAULT_TRANSITION_THRESHOLD = 0.30
MIN_TRANSITION_THRESHOLD = 0.10
MAX_TRANSITION_THRESHOLD = 0.40
TRANSITION_MARGIN = 0.08

# Segment-based finger state gate — divides the gesture into temporal
# segments and uses the MAXIMUM per-segment mismatch instead of a global
# average. Catches cases where the overall average finger mismatch is
# low but one critical section of the gesture uses wrong fingers.
SEGMENT_COUNT = 6
DEFAULT_SEGMENT_THRESHOLD = 0.35
MIN_SEGMENT_THRESHOLD = 0.12
MAX_SEGMENT_THRESHOLD = 0.50
SEGMENT_MARGIN = 0.10

# Robust statistics — Median Absolute Deviation (MAD) replaces volatile
# mean/std on small sample sizes. The constant 1.4826 converts MAD to a
# consistent estimator of the standard deviation for normal distributions.
MAD_CONSISTENCY_CONSTANT = 1.4826
ROBUST_STD_FACTOR = 2.5
THRESHOLD_METHOD_ROBUST = "robust_statistical_v3"

# Interactive quality gate during registration — rejects outlier samples
# that deviate too far from the growing cluster of accepted recordings.
OUTLIER_REJECTION_FACTOR = 1.4

# Cohort-based calibration — minimum DTW distance between user gesture
# and cohort gestures below which a "too simple" warning is issued.
MIN_IMPOSTOR_DISTANCE_RATIO = 1.5   # user-to-cohort must be > 1.5 × intra-user max

# Score fusion — weighted combination of numeric gate scores.
# Gate 3 (transition order) remains a hard boolean gate.
FUSION_WEIGHTS = (0.45, 0.30, 0.25)       # DTW, Finger, Segment
FUSION_ACCEPTANCE_THRESHOLD = 0.55         # Minimum fused score to grant access

# Adaptive template aging — only replace templates when the live
# gesture matches with very high confidence (well within threshold).
AGING_CONFIDENCE_RATIO = 0.70              # DTW must be <= 70% of threshold
AGING_MIN_FACE_CONFIDENCE = 0.70           # face_confidence (mapped scale, ~cosine 0.59)
AGING_MIN_ANTHRO_CONFIDENCE = 0.85
AGING_MIN_KINEMATIC_CONFIDENCE = 0.65

# Multimodal soft consensus: a verified face with RAW cosine >= this value
# may admit borderline hand anatomy (see authenticate_with_details).
SOFT_CONSENSUS_MIN_FACE_COSINE = 0.70
SOFT_CONSENSUS_MIN_ANTHRO_CONFIDENCE = 0.35
SOFT_CONSENSUS_MIN_BIOMETRIC_SCORE = 0.50

# Global hand-trajectory gate (wrist path in palm lengths). Only available
# for users registered with raw landmarks (gesture_N_raw.npy).
# ponytail: floor is a calibration knob; a still hand vs a hand that moves
# ~1 palm length scores several units, so 3.0 still separates them while
# tolerating small drift. Retune with real registrations.
MIN_TRAJECTORY_THRESHOLD = 3.0
TRAJECTORY_METHOD = "wrist_path_dtw_v1"

# Face failure reasons produced by utils/face_auth.py that are passed
# through verbatim; anything else is reported as an impostor face.
FACE_FAILURE_REASONS = {
    "impostor_face_identity",
    "lookalike_sibling_detected",
    "no_face_detected",
    "unstable_face_match",
}

# Usernames become directory names, so they are restricted to a safe
# character set (no path separators, no leading dot, no "..").
USERNAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")


# ============================================================================
# USERNAMES & SAFE FILE I/O
# ============================================================================

def is_valid_username(name):
    """True if name is already a normalised, filesystem-safe username."""
    return bool(USERNAME_PATTERN.match(name or "")) and ".." not in name


def normalize_username(raw):
    """
    Normalise user input to the canonical username used on disk.

    Registration always stored lower-case names with spaces replaced by
    underscores, so authentication and face enrollment must apply the same
    rule. Raises ValueError for names that are empty or unsafe as a
    directory name (e.g. containing '/', '\\' or '..').
    """
    name = (raw or "").strip().lower().replace(" ", "_")
    if not is_valid_username(name):
        raise ValueError(
            f"Invalid username '{raw}'. Use 1-64 letters, digits, '_', '-' "
            f"or '.', starting with a letter or digit."
        )
    return name


def _atomic_write_json(path, data):
    """Write JSON so a crash never leaves a half-written file."""
    fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(path), prefix=".tmp_", suffix=".json"
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _atomic_save_npy(path, array):
    """np.save that never leaves a half-written file."""
    fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(path), prefix=".tmp_", suffix=".npy"
    )
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, array)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _read_config(username, templates_dir=TEMPLATES_DIR):
    """
    Read templates/<user>/config.json.

    Returns {} when the file does not exist (legacy single-sample users).
    Raises ValueError for a corrupted file instead of silently treating it
    as empty — the old behaviour could overwrite the corrupted config with
    a near-empty one and lose every calibrated threshold.
    """
    config_path = os.path.join(templates_dir, username, "config.json")
    if not os.path.exists(config_path):
        return {}
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(
            f"Profile config is corrupted: {config_path} ({e}). "
            f"Re-register the user to rebuild it."
        ) from e
    if not isinstance(config, dict):
        raise ValueError(f"Profile config is not a JSON object: {config_path}")
    return config


# ============================================================================
# CORE DTW FUNCTIONS
# ============================================================================

def compute_dtw_distance(gesture_a, gesture_b):
    """
    Compute the DTW distance between two normalized gesture sequences.

    Each gesture has shape (N, 21, 3). Landmarks are flattened to 63
    dimensions per frame for multi-dimensional DTW comparison.

    Args:
        gesture_a: numpy array of shape (N, 21, 3).
        gesture_b: numpy array of shape (M, 21, 3).

    Returns:
        float: DTW distance score. Lower = more similar. 0.0 = identical.
    """
    flat_a = gesture_a.reshape(gesture_a.shape[0], -1).astype(np.float64)
    flat_b = gesture_b.reshape(gesture_b.shape[0], -1).astype(np.float64)

    distance = dtw_ndim.distance(flat_a, flat_b)
    return distance


def _joint_angle_degrees(point_a, point_b, point_c):
    """
    Return the angle at point_b formed by point_a -> point_b -> point_c.
    """
    vector_a = point_a - point_b
    vector_c = point_c - point_b

    norm_a = np.linalg.norm(vector_a)
    norm_c = np.linalg.norm(vector_c)

    if norm_a < 1e-8 or norm_c < 1e-8:
        return 0.0

    cosine = np.dot(vector_a, vector_c) / (norm_a * norm_c)
    cosine = np.clip(cosine, -1.0, 1.0)

    return float(np.degrees(np.arccos(cosine)))


def _is_finger_extended(frame, finger_name, joints):
    """
    Estimate whether one finger is extended in a normalized landmark frame.

    The check combines finger straightness with fingertip distance from the
    wrist. That makes it more stable than a simple y-coordinate comparison
    when the hand rotates in the camera view.
    """
    wrist = frame[0]
    base_idx, lower_idx, upper_idx, tip_idx = joints
    base = frame[base_idx]
    lower = frame[lower_idx]
    upper = frame[upper_idx]
    tip = frame[tip_idx]

    if finger_name == "thumb":
        angle = _joint_angle_degrees(base, upper, tip)
        reference_distance = np.linalg.norm(upper - wrist)
    else:
        angle = _joint_angle_degrees(base, lower, tip)
        reference_distance = np.linalg.norm(lower - wrist)

    tip_distance = np.linalg.norm(tip - wrist)

    return (
        angle >= FINGER_EXTENDED_ANGLE_DEG
        and tip_distance >= reference_distance * FINGER_TIP_DISTANCE_MARGIN
    )


def extract_finger_state_sequence(gesture):
    """
    Convert a normalized gesture into per-frame finger extension states.

    Returns:
        numpy array of shape (N, 5), ordered as
        thumb, index, middle, ring, pinky. Values are 0.0 or 1.0.
    """
    states = []

    for frame in gesture:
        frame_state = [
            1.0 if _is_finger_extended(frame, name, joints) else 0.0
            for name, joints in FINGER_JOINTS.items()
        ]
        states.append(frame_state)

    return np.array(states, dtype=np.float64)


def compute_finger_state_mismatch(gesture_a, gesture_b):
    """
    Compute how often two gestures use different extended fingers.

    Returns:
        float in [0.0, 1.0], where 0.0 means the finger-state sequence
        matched exactly and 1.0 means every finger state differed.
    """
    states_a = extract_finger_state_sequence(gesture_a)
    states_b = extract_finger_state_sequence(gesture_b)

    n_frames = min(states_a.shape[0], states_b.shape[0])
    if n_frames == 0:
        return 1.0

    mismatches = np.abs(states_a[:n_frames] - states_b[:n_frames])
    return float(np.mean(mismatches))


def compute_pairwise_distances(samples):
    """
    Compute DTW distances between every pair of registration samples.

    Args:
        samples: list of numpy arrays, each of shape (N, 21, 3).

    Returns:
        list of float: All pairwise DTW distances.
    """
    pairwise_distances = []

    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            d = compute_dtw_distance(samples[i], samples[j])
            pairwise_distances.append(d)

    return pairwise_distances


def compute_threshold_details(pairwise_distances, min_threshold=MIN_THRESHOLD):
    """
    Compute a robust per-user threshold and diagnostic metadata.

    Uses Median Absolute Deviation (MAD) instead of volatile mean/std
    for threshold estimation on small sample sizes (n=10).  MAD is
    resistant to outliers — a single anomalous distance cannot skew the
    threshold the way it does with standard deviation.

    The threshold is computed as:
        robust_threshold = median + ROBUST_STD_FACTOR * (1.4826 * MAD)

    Then capped by max_margin and floored by min_threshold.

    Args:
        pairwise_distances: list of genuine intra-user DTW distances.
        min_threshold: floor for the final threshold (MIN_THRESHOLD for the
            pose DTW gate, MIN_TRAJECTORY_THRESHOLD for the wrist path).

    Returns:
        dict with threshold, summary statistics, and method metadata.
    """
    if not pairwise_distances:
        return {
            "threshold": DEFAULT_THRESHOLD,
            "method": "default_single_sample",
            "mean_distance": 0.0,
            "std_distance": 0.0,
            "median_distance": 0.0,
            "mad_distance": 0.0,
            "robust_std": 0.0,
            "percentile_distance": 0.0,
            "max_pairwise_distance": 0.0,
            "statistical_threshold": DEFAULT_THRESHOLD,
            "robust_threshold": DEFAULT_THRESHOLD,
            "percentile_threshold": DEFAULT_THRESHOLD,
            "max_margin_threshold": DEFAULT_THRESHOLD,
            "legacy_threshold": DEFAULT_THRESHOLD,
            "consistency_score": 0.0,
        }

    distances = np.array(pairwise_distances, dtype=np.float64)

    mean_distance = float(np.mean(distances))
    std_distance = float(np.std(distances))
    median_distance = float(np.median(distances))
    percentile_distance = float(
        np.percentile(distances, THRESHOLD_PERCENTILE)
    )
    max_distance = float(np.max(distances))

    # ── Robust statistics (MAD-based) ─────────────────────────────
    mad_distance = float(np.median(np.abs(distances - median_distance)))
    robust_std = MAD_CONSISTENCY_CONSTANT * mad_distance
    robust_threshold = median_distance + (ROBUST_STD_FACTOR * robust_std)

    # ── Legacy statistics (kept for backward-compat metadata) ─────
    statistical_threshold = mean_distance + (
        THRESHOLD_STD_FACTOR * std_distance
    )
    percentile_threshold = percentile_distance * THRESHOLD_SAFETY_MARGIN
    max_margin_threshold = max_distance * THRESHOLD_MAX_MARGIN
    legacy_threshold = max_distance * THRESHOLD_MULTIPLIER

    # Primary candidate: the robust (MAD-based) threshold, but at least
    # as high as the percentile-based estimate so genuine variation is
    # always covered.  Then cap with max_margin to prevent runaway.
    candidate_threshold = min(
        max(robust_threshold, percentile_threshold),
        max_margin_threshold,
        legacy_threshold,
    )
    threshold = max(min_threshold, candidate_threshold)

    if median_distance > 1e-6:
        consistency_score = max(
            0.0,
            100.0 * (1.0 - (robust_std / median_distance))
        )
    else:
        consistency_score = 100.0

    return {
        "threshold": float(threshold),
        "method": THRESHOLD_METHOD_ROBUST,
        "mean_distance": mean_distance,
        "std_distance": std_distance,
        "median_distance": median_distance,
        "mad_distance": float(mad_distance),
        "robust_std": float(robust_std),
        "percentile_distance": percentile_distance,
        "max_pairwise_distance": max_distance,
        "statistical_threshold": float(statistical_threshold),
        "robust_threshold": float(robust_threshold),
        "percentile_threshold": float(percentile_threshold),
        "max_margin_threshold": float(max_margin_threshold),
        "legacy_threshold": float(max(min_threshold, legacy_threshold)),
        "consistency_score": float(consistency_score),
    }


def compute_finger_state_pairwise_mismatches(samples):
    """
    Compute finger-state mismatches between every pair of samples.

    Args:
        samples: list of numpy arrays, each of shape (N, 21, 3).

    Returns:
        list of float: Pairwise finger-state mismatch rates.
    """
    pairwise_mismatches = []

    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            mismatch = compute_finger_state_mismatch(samples[i], samples[j])
            pairwise_mismatches.append(mismatch)

    return pairwise_mismatches


def compute_finger_state_threshold_details(samples):
    """
    Compute a per-user threshold for the finger-state security gate.

    The threshold is calibrated from genuine registration samples, then
    capped so different-finger attempts cannot be accepted simply because
    the motion trajectory is close.
    """
    pairwise_mismatches = compute_finger_state_pairwise_mismatches(samples)

    if not pairwise_mismatches:
        return {
            "threshold": DEFAULT_FINGER_STATE_THRESHOLD,
            "method": "default_single_sample",
            "pairwise_mismatches": [],
            "mean_mismatch": 0.0,
            "max_mismatch": 0.0,
        }

    mismatches = np.array(pairwise_mismatches, dtype=np.float64)
    mean_mismatch = float(np.mean(mismatches))
    max_mismatch = float(np.max(mismatches))

    threshold = max(
        MIN_FINGER_STATE_THRESHOLD,
        max_mismatch + FINGER_STATE_MARGIN
    )
    threshold = min(MAX_FINGER_STATE_THRESHOLD, threshold)

    return {
        "threshold": float(threshold),
        "method": FINGER_STATE_METHOD,
        "pairwise_mismatches": pairwise_mismatches,
        "mean_mismatch": mean_mismatch,
        "max_mismatch": max_mismatch,
    }


# ============================================================================
# TRANSITION SEQUENCE & SEGMENT ANALYSIS
# ============================================================================

def smooth_finger_states(states, window=TRANSITION_SMOOTHING_WINDOW):
    """
    Apply sliding-window majority vote to denoise per-frame finger states.

    Finger detection can flicker between extended/not-extended for a frame
    or two due to landmark jitter. Smoothing ensures that only sustained
    state changes are counted as real transitions.

    Args:
        states: numpy array of shape (N, 5) with 0.0/1.0 values.
        window: int, size of the smoothing window.

    Returns:
        numpy array of shape (N, 5) with smoothed 0.0/1.0 values.
    """
    n_frames, n_fingers = states.shape
    smoothed = np.zeros_like(states)
    half = window // 2

    for i in range(n_frames):
        start = max(0, i - half)
        end = min(n_frames, i + half + 1)
        for f in range(n_fingers):
            smoothed[i, f] = (
                1.0 if np.mean(states[start:end, f]) >= 0.5 else 0.0
            )

    return smoothed


def extract_finger_transitions(gesture):
    """
    Extract the ordered sequence of finger state transitions from a gesture.

    A transition is a sustained change from extended to not-extended (or
    vice versa). The sequence captures WHAT changed and in WHAT ORDER,
    which is the core of sequential finger-pattern passwords like "1-2-4-3".

    The raw per-frame finger states are smoothed first to eliminate
    momentary flicker, so only deliberate, sustained state changes are
    recorded.

    Args:
        gesture: numpy array of shape (N, 21, 3).

    Returns:
        list of tuples: [(finger_index, direction), ...] where
        finger_index is 0–4 (thumb through pinky) and direction
        is 'up' (extended) or 'down' (folded).
    """
    raw_states = extract_finger_state_sequence(gesture)
    states = smooth_finger_states(raw_states)

    # Skip the thumb (index 0) — its extended/folded detection is
    # inherently noisy across hand orientations and adds spurious
    # transitions that pollute the edit-distance comparison.  Gesture
    # passwords are defined by fingers 1-4 (index through pinky).
    finger_range = range(1, 5)

    transitions = []
    for frame_idx in range(1, states.shape[0]):
        for finger_idx in finger_range:
            prev = states[frame_idx - 1, finger_idx]
            curr = states[frame_idx, finger_idx]
            if prev == 0.0 and curr == 1.0:
                transitions.append((finger_idx, 'up'))
            elif prev == 1.0 and curr == 0.0:
                transitions.append((finger_idx, 'down'))

    # Deduplicate consecutive identical transitions that may appear if
    # smoothing creates a brief plateau followed by the same change.
    deduped = []
    for t in transitions:
        if not deduped or deduped[-1] != t:
            deduped.append(t)

    return deduped


def _levenshtein_distance(seq_a, seq_b):
    """
    Compute the Levenshtein (edit) distance between two sequences.

    Each element is compared for equality using ==.  The edit distance is
    the minimum number of insertions, deletions, and substitutions needed
    to transform seq_a into seq_b.

    Args:
        seq_a, seq_b: lists of comparable elements.

    Returns:
        int: edit distance (0 means identical sequences).
    """
    n = len(seq_a)
    m = len(seq_b)

    # dp[i][j] = edit distance between seq_a[:i] and seq_b[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1],  # substitution
                )

    return dp[n][m]


def compute_transition_dissimilarity(gesture_a, gesture_b):
    """
    Compare finger transition sequences between two gestures.

    Returns a dissimilarity score in [0.0, 1.0]:
        0.0 = identical transition sequences
        1.0 = completely different sequences

    The score is the edit distance normalized by the length of the
    longer sequence.  This penalizes missing, extra, or reordered
    finger state changes — which is exactly the information that
    DTW and average-finger-mismatch fail to capture.

    Args:
        gesture_a, gesture_b: numpy arrays of shape (N, 21, 3).

    Returns:
        float: normalized transition dissimilarity.
    """
    trans_a = extract_finger_transitions(gesture_a)
    trans_b = extract_finger_transitions(gesture_b)

    # Both static (no transitions) = identical.
    if not trans_a and not trans_b:
        return 0.0

    max_len = max(len(trans_a), len(trans_b))
    if max_len == 0:
        return 0.0

    edit_dist = _levenshtein_distance(trans_a, trans_b)
    return min(1.0, edit_dist / max_len)


def compute_segment_max_mismatch(gesture_a, gesture_b,
                                  n_segments=SEGMENT_COUNT):
    """
    Divide gestures into temporal segments and return the MAXIMUM
    per-segment finger state mismatch.

    Unlike compute_finger_state_mismatch() which averages over ALL
    frames, this function catches cases where most of the gesture
    matches but one critical section uses different fingers.  A
    single bad segment is enough to fail authentication.

    Args:
        gesture_a, gesture_b: numpy arrays of shape (N, 21, 3).
        n_segments: number of temporal segments to divide into.

    Returns:
        float in [0.0, 1.0]: maximum per-segment mismatch rate.
    """
    states_a = extract_finger_state_sequence(gesture_a)
    states_b = extract_finger_state_sequence(gesture_b)

    n_frames = min(states_a.shape[0], states_b.shape[0])
    if n_frames == 0:
        return 1.0

    segment_size = max(1, n_frames // n_segments)
    max_mismatch = 0.0

    for seg in range(n_segments):
        start = seg * segment_size
        if seg == n_segments - 1:
            end = n_frames      # last segment takes all remaining frames
        else:
            end = min(n_frames, start + segment_size)

        if start >= end:
            continue

        seg_a = states_a[start:end]
        seg_b = states_b[start:end]
        seg_mismatch = float(np.mean(np.abs(seg_a - seg_b)))
        max_mismatch = max(max_mismatch, seg_mismatch)

    return max_mismatch


def compute_transition_pairwise_dissimilarities(samples):
    """Compute transition dissimilarity between every pair of samples."""
    pairwise = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            d = compute_transition_dissimilarity(samples[i], samples[j])
            pairwise.append(d)
    return pairwise


def compute_transition_threshold_details(samples):
    """
    Compute a per-user threshold for the transition sequence gate.

    The threshold is calibrated from genuine registration samples so that
    natural timing variation between performances of the same gesture is
    tolerated, but a missing or reordered finger raise is rejected.
    """
    pairwise = compute_transition_pairwise_dissimilarities(samples)

    if not pairwise:
        return {
            "threshold": DEFAULT_TRANSITION_THRESHOLD,
            "method": "default_single_sample",
            "pairwise_dissimilarities": [],
            "mean_dissimilarity": 0.0,
            "max_dissimilarity": 0.0,
        }

    arr = np.array(pairwise, dtype=np.float64)
    mean_val = float(np.mean(arr))
    max_val = float(np.max(arr))

    threshold = max(MIN_TRANSITION_THRESHOLD, max_val + TRANSITION_MARGIN)
    threshold = min(MAX_TRANSITION_THRESHOLD, threshold)

    return {
        "threshold": float(threshold),
        "method": "transition_edit_distance_v1",
        "pairwise_dissimilarities": pairwise,
        "mean_dissimilarity": mean_val,
        "max_dissimilarity": max_val,
    }


def compute_segment_pairwise_mismatches(samples, n_segments=SEGMENT_COUNT):
    """Compute segment max mismatch between every pair of samples."""
    pairwise = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            m = compute_segment_max_mismatch(
                samples[i], samples[j], n_segments
            )
            pairwise.append(m)
    return pairwise


def compute_segment_threshold_details(samples):
    """
    Compute a per-user threshold for the segment-based finger gate.

    Uses the maximum segment mismatch seen across all registration pairs,
    plus a safety margin.
    """
    pairwise = compute_segment_pairwise_mismatches(samples)

    if not pairwise:
        return {
            "threshold": DEFAULT_SEGMENT_THRESHOLD,
            "method": "default_single_sample",
            "pairwise_max_mismatches": [],
            "mean_max_mismatch": 0.0,
            "max_max_mismatch": 0.0,
        }

    arr = np.array(pairwise, dtype=np.float64)
    mean_val = float(np.mean(arr))
    max_val = float(np.max(arr))

    threshold = max(MIN_SEGMENT_THRESHOLD, max_val + SEGMENT_MARGIN)
    threshold = min(MAX_SEGMENT_THRESHOLD, threshold)

    return {
        "threshold": float(threshold),
        "method": "segment_max_mismatch_v1",
        "pairwise_max_mismatches": pairwise,
        "mean_max_mismatch": mean_val,
        "max_max_mismatch": max_val,
    }


def compute_threshold_from_samples(samples):
    """
    Compute an optimal authentication threshold from registration samples.

    Strategy:
        1. Compute DTW distance between every pair of samples
        2. Estimate genuine variation using mean + standard deviation
        3. Add a percentile-based safety margin
        4. Cap the result so one noisy sample cannot make access too loose
        5. Enforce MIN_THRESHOLD floor to prevent being too strict

    Args:
        samples: list of numpy arrays, each of shape (N, 21, 3).

    Returns:
        tuple of (float, list):
            - float: The computed threshold
            - list: All pairwise distances (for display/debugging)
    """
    pairwise_distances = compute_pairwise_distances(samples)
    details = compute_threshold_details(pairwise_distances)

    return details["threshold"], pairwise_distances


# ============================================================================
# TEMPLATE LOADING FUNCTIONS
# ============================================================================

def load_gesture_template(filepath):
    """
    Load a gesture template from a .npy file.

    Args:
        filepath: str, path to the .npy file.

    Returns:
        numpy array of shape (N, 21, 3).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the array shape is not (N, 21, 3).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Gesture template not found: {filepath}")

    template = np.load(filepath)

    if template.ndim != 3 or template.shape[1:] != (21, 3):
        raise ValueError(
            f"Invalid gesture shape: expected (N, 21, 3), "
            f"got {template.shape}. File may be corrupted."
        )

    return template


def load_all_user_templates(username, templates_dir=TEMPLATES_DIR):
    """
    Load ALL gesture templates for a user.

    Supports both formats:
        - New: gesture_1.npy, gesture_2.npy, gesture_3.npy (multi-sample)
        - Old: gesture.npy (single-sample, backward compatible)

    Args:
        username: str, the user's identifier.
        templates_dir: str, path to templates root directory.

    Returns:
        list of numpy arrays, each of shape (N, 21, 3).

    Raises:
        FileNotFoundError: If no templates exist for the user.
    """
    user_dir = os.path.join(templates_dir, username)

    if not os.path.exists(user_dir):
        raise FileNotFoundError(
            f"No registration found for user '{username}'"
        )

    templates = []

    # Try new format first: gesture_1.npy, gesture_2.npy, ...
    i = 1
    while True:
        filepath = os.path.join(user_dir, f"gesture_{i}.npy")
        if os.path.exists(filepath):
            templates.append(load_gesture_template(filepath))
            i += 1
        else:
            break

    # Fallback to old format: gesture.npy
    if not templates:
        old_path = os.path.join(user_dir, "gesture.npy")
        if os.path.exists(old_path):
            templates.append(load_gesture_template(old_path))

    if not templates:
        raise FileNotFoundError(
            f"No gesture templates found for user '{username}'"
        )

    return templates


def load_user_threshold(username, templates_dir=TEMPLATES_DIR):
    """
    Load the per-user authentication threshold from config.json.

    Falls back to DEFAULT_THRESHOLD if no config file exists
    (e.g., for old single-sample registrations).

    Args:
        username: str, the user's identifier.
        templates_dir: str, path to templates root directory.

    Returns:
        float: The authentication threshold for this user.
    """
    return _read_config(username, templates_dir).get(
        "threshold", DEFAULT_THRESHOLD
    )


def load_user_finger_state_threshold(username, templates_dir=TEMPLATES_DIR):
    """
    Load the per-user finger-state mismatch threshold from config.json.

    If a user was registered before this security gate existed, derive a
    threshold from their saved samples when multiple templates are present.
    Single-sample legacy users fall back to DEFAULT_FINGER_STATE_THRESHOLD.
    """
    config = _read_config(username, templates_dir)
    if "finger_state_threshold" in config:
        return config["finger_state_threshold"]

    try:
        templates = load_all_user_templates(username, templates_dir)
    except FileNotFoundError:
        return DEFAULT_FINGER_STATE_THRESHOLD

    if len(templates) < 2:
        return DEFAULT_FINGER_STATE_THRESHOLD

    details = compute_finger_state_threshold_details(templates)
    return details["threshold"]


def load_user_transition_threshold(username, templates_dir=TEMPLATES_DIR):
    """
    Load the per-user transition sequence threshold from config.json.

    For users registered before this gate existed, the threshold is
    computed on-the-fly from saved templates.  Single-sample legacy
    users fall back to DEFAULT_TRANSITION_THRESHOLD.
    """
    config = _read_config(username, templates_dir)
    if "transition_threshold" in config:
        return config["transition_threshold"]

    try:
        templates = load_all_user_templates(username, templates_dir)
    except FileNotFoundError:
        return DEFAULT_TRANSITION_THRESHOLD

    if len(templates) < 2:
        return DEFAULT_TRANSITION_THRESHOLD

    details = compute_transition_threshold_details(templates)
    return details["threshold"]


def load_user_segment_threshold(username, templates_dir=TEMPLATES_DIR):
    """
    Load the per-user segment mismatch threshold from config.json.

    For users registered before this gate existed, the threshold is
    computed on-the-fly from saved templates.  Single-sample legacy
    users fall back to DEFAULT_SEGMENT_THRESHOLD.
    """
    config = _read_config(username, templates_dir)
    if "segment_threshold" in config:
        return config["segment_threshold"]

    try:
        templates = load_all_user_templates(username, templates_dir)
    except FileNotFoundError:
        return DEFAULT_SEGMENT_THRESHOLD

    if len(templates) < 2:
        return DEFAULT_SEGMENT_THRESHOLD

    details = compute_segment_threshold_details(templates)
    return details["threshold"]


def load_user_anthropometric_profile(username, stored_templates=None,
                                     templates_dir=TEMPLATES_DIR):
    """
    Load the user's anthropometric hand geometry profile.

    If present in config.json with current method 'metacarpal_invariant_v2', loads it.
    If missing or older method, re-calibrates from stored templates and updates config.json.
    """
    config_path = os.path.join(templates_dir, username, "config.json")
    config = _read_config(username, templates_dir)
    profile = config.get("anthropometric_profile")
    if isinstance(profile, dict) and profile.get("method") == "metacarpal_invariant_v2":
        return profile

    if stored_templates is None:
        try:
            stored_templates = load_all_user_templates(username, templates_dir)
        except FileNotFoundError:
            stored_templates = []

    if stored_templates:
        new_profile = calibrate_anthropometric_profile(stored_templates)
        if os.path.exists(config_path):
            # Upgrade the cached profile in place. A failed write only means
            # recalibrating again next time, so it must not block login.
            config["anthropometric_profile"] = new_profile
            try:
                _atomic_write_json(config_path, config)
            except OSError as e:
                print(f"  Warning: could not update {config_path}: {e}")
        return new_profile

    # Fallback to neutral default
    return {
        "baseline": [0.55, 1.02, 0.80, 0.75, 0.50],
        "std": [0.03, 0.03, 0.03, 0.03, 0.03],
        "tolerance": DEFAULT_ANTHRO_TOLERANCE,
        "method": "metacarpal_invariant_v2",
    }


def load_user_kinematic_profile(username, stored_templates=None,
                                templates_dir=TEMPLATES_DIR,
                                raw_templates=None):
    """
    Load the user's kinematic rhythm profile.

    If present in config.json, loads the cached baseline (its optional
    "method" key tells authentication how to compute the live signature).
    Otherwise calibrates on the fly: from raw landmarks when available
    (smoothed method), else from the stored 60-frame templates (legacy).
    """
    config = _read_config(username, templates_dir)
    if isinstance(config.get("kinematic_profile"), dict):
        return config["kinematic_profile"]

    if raw_templates:
        return calibrate_kinematic_profile(
            [prepare_kinematic_sequence(r) for r in raw_templates],
            method=KINEMATIC_METHOD_SMOOTHED,
        )

    if stored_templates is None:
        try:
            stored_templates = load_all_user_templates(username, templates_dir)
        except FileNotFoundError:
            stored_templates = []

    if stored_templates:
        return calibrate_kinematic_profile(stored_templates)

    return {
        "baseline_binned_velocity": [0.1] * 10,
        "baseline_jerk": 0.001,
        "baseline_peak_phase": 0.5,
        "tolerance": DEFAULT_KINEMATIC_TOLERANCE,
    }


def load_user_face_profile(username, templates_dir=TEMPLATES_DIR):
    """
    Load the enrolled 128-D face embedding for the user.
    Returns numpy array of shape (128,) or None if user has not enrolled face (legacy mode).
    """
    return load_face_embedding(username, templates_dir)


def load_user_raw_templates(username, n_templates, templates_dir=TEMPLATES_DIR):
    """
    Load the raw (un-normalised) landmark recordings gesture_N_raw.npy.

    Returns a list aligned with gesture_1..gesture_N, or None when the user
    was registered before raw recordings were stored (or any file is
    missing/invalid) — callers then fall back to the legacy pose-only path.
    Mixing raw and non-raw templates is never allowed.
    """
    user_dir = os.path.join(templates_dir, username)
    raws = []
    for i in range(1, n_templates + 1):
        path = os.path.join(user_dir, f"gesture_{i}_raw.npy")
        if not os.path.exists(path):
            return None
        raw = np.load(path)
        if raw.ndim != 3 or raw.shape[1:] != (21, 3) or raw.shape[0] < 2:
            return None
        raws.append(raw)
    return raws if raws else None


def load_user_profile(username, templates_dir=TEMPLATES_DIR):
    """
    Load everything needed to authenticate a user, in one place.

    Used at startup and again after adaptive template aging, so the
    in-memory profile can never drift from what is on disk.

    Raises:
        FileNotFoundError: user or templates missing.
        ValueError: invalid username or corrupted config.json.
    """
    username = normalize_username(username)
    templates = load_all_user_templates(username, templates_dir)
    config = _read_config(username, templates_dir)
    raw_templates = load_user_raw_templates(username, len(templates), templates_dir)

    trajectory_threshold = None
    trajectory_templates = None
    frame_aspect = config.get("frame_aspect")
    if raw_templates is not None and config.get("trajectory_threshold") is not None:
        trajectory_threshold = float(config["trajectory_threshold"])
        aspect = float(frame_aspect) if frame_aspect else 1.0
        trajectory_templates = [
            extract_wrist_trajectory(r, aspect) for r in raw_templates
        ]

    return {
        "username": username,
        "templates": templates,
        "raw_templates": raw_templates,
        "threshold": load_user_threshold(username, templates_dir),
        "finger_state_threshold": load_user_finger_state_threshold(username, templates_dir),
        "transition_threshold": load_user_transition_threshold(username, templates_dir),
        "segment_threshold": load_user_segment_threshold(username, templates_dir),
        "anthropometric_profile": load_user_anthropometric_profile(
            username, templates, templates_dir
        ),
        "kinematic_profile": load_user_kinematic_profile(
            username, templates, templates_dir, raw_templates
        ),
        "trajectory_threshold": trajectory_threshold,
        "trajectory_templates": trajectory_templates,
        "frame_aspect": frame_aspect,
        "face_embedding": load_face_embedding(username, templates_dir),
    }


# ============================================================================
# AUTHENTICATION FUNCTIONS
# ============================================================================

def _face_failure_reason(face_details):
    """Pass through known face failure reasons; default to impostor."""
    reason = (face_details or {}).get("reason")
    return reason if reason in FACE_FAILURE_REASONS else "impostor_face_identity"


def authenticate_with_details(live_gesture, stored_templates, threshold,
                              finger_state_threshold=None,
                              transition_threshold=None,
                              segment_threshold=None,
                              anthropometric_profile=None,
                              kinematic_profile=None,
                              face_match=None,
                              face_confidence=0.0,
                              face_details=None,
                              live_trajectory=None,
                              stored_trajectories=None,
                              trajectory_threshold=None,
                              live_kinematic_sequence=None):
    """
    Authenticate a live gesture against stored templates.

    Macro decision for one template (all must hold):
        - Gate 3 transition order passes (hard gate),
        - Gate 1 DTW distance <= calibrated threshold (hard gate),
        - Gate 2 finger-state OR Gate 4 segment mismatch within threshold
          (they measure the same finger states, so one may be borderline),
        - wrist trajectory within threshold, when the user has one,
        - fused score S = w1*(1 - DTW/θ) + w2*(1 - M_avg) + w3*(1 - M_seg)
          >= FUSION_ACCEPTANCE_THRESHOLD.

    Previously only the transition gate and the fused score decided, so a
    template could fail Gate 1 outright and still pass on finger scores
    alone (0.30 + 0.25 = 0.55), and the calibrated per-gate thresholds were
    never enforced.

    On top of the macro gates, hand anthropometry, kinematics and (when
    enrolled) face identity must all verify.

    Args:
        live_gesture: numpy array of shape (N, 21, 3), pose-normalised.
        stored_templates: list of numpy arrays (one per registration sample).
        threshold: float, DTW threshold (gate + score normalisation).
        finger_state_threshold, transition_threshold, segment_threshold:
            per-gate thresholds (defaults used when None).
        anthropometric_profile, kinematic_profile: calibrated baselines.
        face_match: True/False, or None when face is not enrolled (legacy).
        face_confidence: mapped face confidence in [0, 1].
        face_details: dict with "score" (raw cosine) and "reason".
        live_trajectory, stored_trajectories, trajectory_threshold: wrist
            path gate; skipped unless all three are given.
        live_kinematic_sequence: prepare_kinematic_sequence(raw) output;
            required when kinematic_profile["method"] is the smoothed method.

    Returns:
        tuple (granted, best_distance, best_index, details).
    """
    if not stored_templates:
        raise ValueError("No stored templates to authenticate against.")
    # Normalise once: callers may pass numpy.bool_, which breaks `is True`.
    face_match = None if face_match is None else bool(face_match)
    if finger_state_threshold is None:
        finger_state_threshold = DEFAULT_FINGER_STATE_THRESHOLD
    if transition_threshold is None:
        transition_threshold = DEFAULT_TRANSITION_THRESHOLD
    if segment_threshold is None:
        segment_threshold = DEFAULT_SEGMENT_THRESHOLD

    # ── Biometric Identity Layer (Anti-Shoulder-Surfing) ─────────
    if anthropometric_profile is None:
        anthropometric_profile = calibrate_anthropometric_profile(stored_templates)
    if kinematic_profile is None:
        kinematic_profile = calibrate_kinematic_profile(stored_templates)

    live_anthro_sig = extract_gesture_anthropometric_signature(live_gesture)
    anthro_tol = anthropometric_profile.get("tolerance", DEFAULT_ANTHRO_TOLERANCE)
    anthro_match, anthro_conf, anthro_details = compare_anthropometric_signatures(
        live_anthro_sig, anthropometric_profile["baseline"], anthro_tol
    )

    # The live signature must be computed the same way as the baseline.
    if kinematic_profile.get("method") == KINEMATIC_METHOD_SMOOTHED:
        if live_kinematic_sequence is None:
            raise ValueError(
                "This kinematic profile was calibrated from raw landmarks; "
                "pass live_kinematic_sequence=prepare_kinematic_sequence(raw)."
            )
        kinematic_input = live_kinematic_sequence
    else:
        kinematic_input = live_gesture
    live_kinematic_sig = extract_kinematic_signature(kinematic_input)
    kin_tol = kinematic_profile.get("tolerance", DEFAULT_KINEMATIC_TOLERANCE)
    kin_match, kin_conf, kin_details = compare_kinematic_signatures(
        live_kinematic_sig, kinematic_profile, kin_tol
    )

    trajectory_active = (
        live_trajectory is not None
        and stored_trajectories is not None
        and trajectory_threshold is not None
        and len(stored_trajectories) == len(stored_templates)
    )

    w_dtw, w_finger, w_segment = FUSION_WEIGHTS
    comparisons = []

    for i, template in enumerate(stored_templates):
        distance = compute_dtw_distance(live_gesture, template)
        finger_mismatch = compute_finger_state_mismatch(live_gesture, template)
        transition_dissim = compute_transition_dissimilarity(live_gesture, template)
        segment_max = compute_segment_max_mismatch(live_gesture, template)

        passes_distance = distance <= threshold
        passes_finger = finger_mismatch <= finger_state_threshold
        passes_transition = transition_dissim <= transition_threshold
        passes_segment = segment_max <= segment_threshold

        if trajectory_active:
            trajectory_distance = compute_dtw_distance(
                live_trajectory, stored_trajectories[i]
            )
            passes_trajectory = trajectory_distance <= trajectory_threshold
        else:
            trajectory_distance = None
            passes_trajectory = True

        # Individual component scores for fusion (clamped to [0, 1])
        score_dtw = max(0.0, 1.0 - distance / threshold) if threshold > 0 else 0.0
        score_finger = max(0.0, 1.0 - finger_mismatch)
        score_segment = max(0.0, 1.0 - segment_max)
        fused_score = (
            w_dtw * score_dtw + w_finger * score_finger + w_segment * score_segment
        )

        passes_gates = bool(
            passes_transition
            and passes_distance
            and (passes_finger or passes_segment)
            and passes_trajectory
            and fused_score >= FUSION_ACCEPTANCE_THRESHOLD
        )

        comparisons.append({
            "sample_index": i,
            "distance": distance,
            "finger_state_mismatch": finger_mismatch,
            "transition_dissimilarity": transition_dissim,
            "segment_max_mismatch": segment_max,
            "trajectory_distance": trajectory_distance,
            "passes_distance": passes_distance,
            "passes_finger_state": passes_finger,
            "passes_transition": passes_transition,
            "passes_segment": passes_segment,
            "passes_trajectory": passes_trajectory,
            "passes_gates": passes_gates,
            "score_dtw": score_dtw,
            "score_finger": score_finger,
            "score_segment": score_segment,
            "fused_score": fused_score,
        })

    passing = [item for item in comparisons if item["passes_gates"]]

    # Combined Biometric Identity Verification:
    passes_biometric = anthro_match and kin_match
    biometric_fused_score = 0.60 * anthro_conf + 0.40 * kin_conf

    # Multimodal Soft Consensus: a strongly verified face (RAW cosine, not
    # the mapped confidence) with fluent kinematics may admit borderline
    # hand anatomy, e.g. in-air signatures where posture inflates one ratio.
    face_cosine = (face_details or {}).get("score")
    soft_consensus_used = False
    if (not passes_biometric and kin_match
            and anthro_conf >= SOFT_CONSENSUS_MIN_ANTHRO_CONFIDENCE
            and face_match is True
            and face_cosine is not None
            and face_cosine >= SOFT_CONSENSUS_MIN_FACE_COSINE
            and biometric_fused_score >= SOFT_CONSENSUS_MIN_BIOMETRIC_SCORE):
        passes_biometric = True
        soft_consensus_used = True

    # Multimodal Identity Consensus:
    face_ok = True if face_match is None else bool(face_match)
    passes_multimodal = face_ok and passes_biometric

    identity_failures = []
    if face_match is False:
        identity_failures.append(_face_failure_reason(face_details))
    if not anthro_match and not soft_consensus_used:
        identity_failures.append("impostor_hand_morphology")
    if not kin_match:
        identity_failures.append("impostor_kinematic_dynamics")

    if passing:
        best = max(passing, key=lambda item: item["fused_score"])
        granted = bool(passes_multimodal)
        failure_reason = None if granted else "_and_".join(identity_failures)
    else:
        best = min(comparisons, key=lambda item: item["distance"])
        granted = False

        # Build failure reason listing every failing condition.
        failed_gates = []
        if not best["passes_distance"]:
            failed_gates.append("distance")
        if not best["passes_finger_state"] and not best["passes_segment"]:
            failed_gates.append("finger_state")
            failed_gates.append("segment_mismatch")
        if not best["passes_transition"]:
            failed_gates.append("transition_order")
        if not best["passes_trajectory"]:
            failed_gates.append("trajectory")
        if best["fused_score"] < FUSION_ACCEPTANCE_THRESHOLD:
            failed_gates.append("low_confidence")
        failed_gates.extend(identity_failures)
        failure_reason = "_and_".join(failed_gates) if failed_gates else "unknown"

    if face_match is None:
        multimodal_fused_score = (
            0.60 * best["fused_score"] + 0.40 * biometric_fused_score
        )
    else:
        multimodal_fused_score = (
            0.40 * face_confidence
            + 0.35 * best["fused_score"]
            + 0.25 * biometric_fused_score
        )

    macro_gates_passed = sum([
        bool(best["passes_distance"]),
        bool(best["passes_finger_state"]),
        bool(best["passes_transition"]),
        bool(best["passes_segment"]),
    ])

    details = {
        "distance": best["distance"],
        "threshold": threshold,
        "finger_state_mismatch": best["finger_state_mismatch"],
        "finger_state_threshold": finger_state_threshold,
        "transition_dissimilarity": best["transition_dissimilarity"],
        "transition_threshold": transition_threshold,
        "segment_max_mismatch": best["segment_max_mismatch"],
        "segment_threshold": segment_threshold,
        "trajectory_active": trajectory_active,
        "trajectory_distance": best["trajectory_distance"],
        "trajectory_threshold": trajectory_threshold if trajectory_active else None,
        "passes_distance": best["passes_distance"],
        "passes_finger_state": best["passes_finger_state"],
        "passes_transition": best["passes_transition"],
        "passes_segment": best["passes_segment"],
        "passes_trajectory": best["passes_trajectory"],
        "passes_macro": bool(passing),
        "macro_gates_passed": macro_gates_passed,
        "score_dtw": best["score_dtw"],
        "score_finger": best["score_finger"],
        "score_segment": best["score_segment"],
        "fused_score": best["fused_score"],
        "fusion_weights": FUSION_WEIGHTS,
        "fusion_acceptance_threshold": FUSION_ACCEPTANCE_THRESHOLD,
        "anthropometric_match": anthro_match,
        "anthropometric_confidence": anthro_conf,
        "anthropometric_details": anthro_details,
        "kinematic_match": kin_match,
        "kinematic_confidence": kin_conf,
        "kinematic_details": kin_details,
        "kinematic_method": kinematic_profile.get("method", "legacy_template_v1"),
        "face_match": face_match,
        "face_confidence": face_confidence,
        "face_details": face_details,
        "soft_consensus_used": soft_consensus_used,
        "passes_biometric": passes_biometric,
        "passes_multimodal": passes_multimodal,
        "biometric_fused_score": round(biometric_fused_score, 4),
        "multimodal_fused_score": round(multimodal_fused_score, 4),
        "failure_reason": failure_reason,
        "comparisons": comparisons,
    }

    return granted, best["distance"], best["sample_index"], details


def authenticate(live_gesture, stored_templates, threshold,
                 finger_state_threshold=None,
                 transition_threshold=None,
                 segment_threshold=None):
    """
    Backward-compatible authentication wrapper.

    Returns the original 3-tuple while internally applying all four
    security gates (DTW, finger state, transition order, segment max).
    """
    granted, distance, best_idx, _ = authenticate_with_details(
        live_gesture,
        stored_templates,
        threshold,
        finger_state_threshold,
        transition_threshold,
        segment_threshold,
    )

    return granted, distance, best_idx


def list_registered_users(templates_dir=TEMPLATES_DIR):
    """
    List all users who have saved gesture templates.

    Directories whose names are not valid usernames (including the hidden
    ".<user>.partial" / ".<user>.old" staging folders used by
    save_registration) are ignored.

    Args:
        templates_dir: str, path to the templates root directory.

    Returns:
        list of str: Sorted list of registered usernames.
    """
    if not os.path.exists(templates_dir):
        return []

    users = []
    for entry in sorted(os.listdir(templates_dir)):
        user_dir = os.path.join(templates_dir, entry)
        if not os.path.isdir(user_dir) or not is_valid_username(entry):
            continue

        # Check for new format (gesture_1.npy) or old format (gesture.npy)
        has_new = os.path.exists(os.path.join(user_dir, "gesture_1.npy"))
        has_old = os.path.exists(os.path.join(user_dir, "gesture.npy"))

        if has_new or has_old:
            users.append(entry)

    return users


# ============================================================================
# REGISTRATION SAVE FUNCTIONS
# ============================================================================

def build_profile_config(samples, raw_samples=None, frame_aspect=None):
    """
    Calibrate every threshold and biometric baseline from a template set.

    Shared by registration and adaptive template aging so both always
    produce the same config structure.

    Args:
        samples: list of pose-normalised templates, each (60, 21, 3).
        raw_samples: optional list of raw landmark recordings (N, 21, 3),
            aligned with samples. Enables the wrist-trajectory gate and the
            smoothed kinematic profile.
        frame_aspect: width / height of the camera frames used to record.

    Returns:
        dict ready to be written to config.json (face_enrolled excluded).
    """
    if raw_samples is not None and len(raw_samples) != len(samples):
        raise ValueError("raw_samples must align one-to-one with samples.")

    pairwise_distances = compute_pairwise_distances(samples)
    threshold_details = compute_threshold_details(pairwise_distances)
    finger_state_details = compute_finger_state_threshold_details(samples)
    transition_details = compute_transition_threshold_details(samples)
    segment_details = compute_segment_threshold_details(samples)

    config = {
        "threshold": round(threshold_details["threshold"], 4),
        "num_samples": len(samples),
        "threshold_method": threshold_details["method"],
        "finger_state_threshold": round(finger_state_details["threshold"], 4),
        "finger_state_method": finger_state_details["method"],
        "finger_state_pairwise_mismatches": [
            round(m, 4) for m in finger_state_details["pairwise_mismatches"]
        ],
        "finger_state_mean_mismatch": round(finger_state_details["mean_mismatch"], 4),
        "finger_state_max_mismatch": round(finger_state_details["max_mismatch"], 4),
        "finger_state_margin": FINGER_STATE_MARGIN,
        "finger_state_min_threshold": MIN_FINGER_STATE_THRESHOLD,
        "finger_state_max_threshold": MAX_FINGER_STATE_THRESHOLD,
        "pairwise_distances": [round(d, 4) for d in pairwise_distances],
        "mean_pairwise_distance": round(threshold_details["mean_distance"], 4),
        "std_pairwise_distance": round(threshold_details["std_distance"], 4),
        "median_pairwise_distance": round(threshold_details["median_distance"], 4),
        "percentile_pairwise_distance": round(threshold_details["percentile_distance"], 4),
        "max_pairwise_distance": round(threshold_details["max_pairwise_distance"], 4),
        "statistical_threshold": round(threshold_details["statistical_threshold"], 4),
        "percentile_threshold": round(threshold_details["percentile_threshold"], 4),
        "max_margin_threshold": round(threshold_details["max_margin_threshold"], 4),
        "legacy_threshold": round(threshold_details["legacy_threshold"], 4),
        "mad_pairwise_distance": round(threshold_details["mad_distance"], 4),
        "robust_std": round(threshold_details["robust_std"], 4),
        "robust_threshold": round(threshold_details["robust_threshold"], 4),
        "consistency_score": round(threshold_details["consistency_score"], 2),
        "threshold_std_factor": THRESHOLD_STD_FACTOR,
        "robust_std_factor": ROBUST_STD_FACTOR,
        "threshold_percentile": THRESHOLD_PERCENTILE,
        "threshold_safety_margin": THRESHOLD_SAFETY_MARGIN,
        "threshold_max_margin": THRESHOLD_MAX_MARGIN,
        "threshold_multiplier": THRESHOLD_MULTIPLIER,
        "transition_threshold": round(transition_details["threshold"], 4),
        "transition_method": transition_details["method"],
        "transition_pairwise_dissimilarities": [
            round(d, 4) for d in transition_details["pairwise_dissimilarities"]
        ],
        "transition_mean_dissimilarity": round(transition_details["mean_dissimilarity"], 4),
        "transition_max_dissimilarity": round(transition_details["max_dissimilarity"], 4),
        "transition_margin": TRANSITION_MARGIN,
        "segment_threshold": round(segment_details["threshold"], 4),
        "segment_method": segment_details["method"],
        "segment_pairwise_max_mismatches": [
            round(m, 4) for m in segment_details["pairwise_max_mismatches"]
        ],
        "segment_mean_max_mismatch": round(segment_details["mean_max_mismatch"], 4),
        "segment_max_max_mismatch": round(segment_details["max_max_mismatch"], 4),
        "segment_margin": SEGMENT_MARGIN,
        "segment_count": SEGMENT_COUNT,
        "anthropometric_profile": calibrate_anthropometric_profile(samples),
    }

    if raw_samples is not None:
        aspect = float(frame_aspect) if frame_aspect else 1.0
        config["kinematic_profile"] = calibrate_kinematic_profile(
            [prepare_kinematic_sequence(r) for r in raw_samples],
            method=KINEMATIC_METHOD_SMOOTHED,
        )
        config["frame_aspect"] = round(aspect, 6)
        if len(raw_samples) >= 2:
            trajectories = [extract_wrist_trajectory(r, aspect) for r in raw_samples]
            traj_pairwise = compute_pairwise_distances(trajectories)
            traj_details = compute_threshold_details(
                traj_pairwise, min_threshold=MIN_TRAJECTORY_THRESHOLD
            )
            config["trajectory_threshold"] = round(traj_details["threshold"], 4)
            config["trajectory_method"] = TRAJECTORY_METHOD
            config["trajectory_pairwise_distances"] = [
                round(d, 4) for d in traj_pairwise
            ]
    else:
        config["kinematic_profile"] = calibrate_kinematic_profile(samples)

    return config


def save_registration(username, samples, threshold=None, pairwise_distances=None,
                      templates_dir=TEMPLATES_DIR, raw_samples=None,
                      frame_aspect=None, face_embedding=None):
    """
    Atomically save a complete multi-sample registration to disk.

    Everything is written to a hidden staging folder first and swapped in
    only when complete, so a crash or error can never leave a half-written
    profile (the old code deleted the previous files before writing).

    Face handling:
        - face_embedding given: it becomes the user's enrolled face.
        - face_embedding None: an existing face_embedding.npy is kept.

    Args:
        username: str, the user's identifier (normalised here).
        samples: list of pose-normalised templates, each (60, 21, 3).
        threshold, pairwise_distances: accepted for backward compatibility;
            all thresholds are recomputed from samples by
            build_profile_config() so the saved config is self-consistent.
        templates_dir: str, path to templates root directory.
        raw_samples: optional raw landmark recordings aligned with samples.
        frame_aspect: camera frame width / height used during recording.
        face_embedding: optional (128,) face embedding.

    Returns:
        str: Path to the user's template directory.
    """
    username = normalize_username(username)
    if not samples:
        raise ValueError("Cannot save a registration without samples.")

    config = build_profile_config(samples, raw_samples, frame_aspect)

    os.makedirs(templates_dir, exist_ok=True)
    user_dir = os.path.join(templates_dir, username)
    staging_dir = os.path.join(templates_dir, f".{username}.partial")
    backup_dir = os.path.join(templates_dir, f".{username}.old")

    shutil.rmtree(staging_dir, ignore_errors=True)
    os.makedirs(staging_dir)
    try:
        for i, sample in enumerate(samples, start=1):
            np.save(os.path.join(staging_dir, f"gesture_{i}.npy"), sample)
            if raw_samples is not None:
                np.save(
                    os.path.join(staging_dir, f"gesture_{i}_raw.npy"),
                    np.asarray(raw_samples[i - 1], dtype=np.float64),
                )

        staged_face = os.path.join(staging_dir, "face_embedding.npy")
        old_face = os.path.join(user_dir, "face_embedding.npy")
        if face_embedding is not None:
            np.save(staged_face,
                    np.asarray(face_embedding, dtype=np.float32).reshape(128))
        elif os.path.exists(old_face):
            shutil.copy2(old_face, staged_face)

        config["face_enrolled"] = os.path.exists(staged_face)
        with open(os.path.join(staging_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # Swap: current -> backup, staging -> current, then drop backup.
        shutil.rmtree(backup_dir, ignore_errors=True)
        if os.path.exists(user_dir):
            os.rename(user_dir, backup_dir)
        try:
            os.rename(staging_dir, user_dir)
        except OSError:
            if os.path.exists(backup_dir) and not os.path.exists(user_dir):
                os.rename(backup_dir, user_dir)
            raise
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    shutil.rmtree(backup_dir, ignore_errors=True)
    return user_dir


def set_face_enrolled_flag(username, templates_dir=TEMPLATES_DIR):
    """Refresh config.json["face_enrolled"] after a face-only enrollment."""
    username = normalize_username(username)
    config_path = os.path.join(templates_dir, username, "config.json")
    if not os.path.exists(config_path):
        return
    config = _read_config(username, templates_dir)
    config["face_enrolled"] = os.path.exists(
        os.path.join(templates_dir, username, "face_embedding.npy")
    )
    _atomic_write_json(config_path, config)


# ============================================================================
# ADAPTIVE TEMPLATE AGING
# ============================================================================

def update_templates_if_high_confidence(
    username, live_gesture, best_distance, threshold,
    stored_templates, templates_dir=TEMPLATES_DIR,
    anthropometric_match=True, anthropometric_confidence=1.0,
    kinematic_match=True, kinematic_confidence=1.0,
    face_match=True, face_confidence=1.0,
    live_raw=None,
):
    """
    Replace the most distant stored template when the live gesture matches
    with very high confidence, solving temporal drift in biomechanics.

    Only triggers when best_distance <= AGING_CONFIDENCE_RATIO * threshold,
    meaning the live gesture is well within the genuine acceptance zone.
    Replaces the single template that is most distant from the current
    cluster (the weakest/oldest/noisiest template).

    Multimodal Biometric Guard:
    Template aging is strictly inhibited unless the face anchor (when
    enrolled), hand anthropometry and kinematics all confirm identity with
    high confidence (face_confidence >= 0.70, anthro >= 0.85,
    kinematic >= 0.65). This prevents impostor template poisoning attacks.

    For users registered with raw landmarks, live_raw is required so the
    raw recording set stays aligned with the pose templates; otherwise
    aging is skipped rather than leaving a mixed template set.

    Returns:
        dict with update details if a template was replaced, else None.
    """
    username = normalize_username(username)

    # Safeguard 1: Impostor template poisoning guard (Multimodal Face + Hand Biometrics)
    face_ok = True if face_match is None else (
        bool(face_match) and face_confidence >= AGING_MIN_FACE_CONFIDENCE
    )
    if not face_ok:
        return None
    if not anthropometric_match or anthropometric_confidence < AGING_MIN_ANTHRO_CONFIDENCE:
        return None
    if not kinematic_match or kinematic_confidence < AGING_MIN_KINEMATIC_CONFIDENCE:
        return None

    # Safeguard 2: Only update on high-confidence macro matches
    if best_distance > AGING_CONFIDENCE_RATIO * threshold:
        return None

    if len(stored_templates) < 2:
        return None

    raw_templates = load_user_raw_templates(
        username, len(stored_templates), templates_dir
    )
    if raw_templates is not None and live_raw is None:
        return None

    # Find the template most distant from the cluster center
    # (average distance to all other templates)
    avg_distances = []
    for i, tmpl in enumerate(stored_templates):
        total = 0.0
        for j, other in enumerate(stored_templates):
            if i != j:
                total += compute_dtw_distance(tmpl, other)
        avg_distances.append(total / (len(stored_templates) - 1))

    worst_idx = int(np.argmax(avg_distances))

    # Check that the live gesture would actually be a better fit
    live_avg = 0.0
    for j, other in enumerate(stored_templates):
        if j != worst_idx:
            live_avg += compute_dtw_distance(live_gesture, other)
    live_avg /= (len(stored_templates) - 1)

    if live_avg >= avg_distances[worst_idx]:
        # The live gesture is actually worse than the current worst — skip
        return None

    updated_templates = list(stored_templates)
    updated_templates[worst_idx] = live_gesture
    updated_raw = None
    if raw_templates is not None:
        updated_raw = list(raw_templates)
        updated_raw[worst_idx] = np.asarray(live_raw, dtype=np.float64)

    user_dir = os.path.join(templates_dir, username)
    config_path = os.path.join(user_dir, "config.json")
    old_config = _read_config(username, templates_dir)
    new_config = build_profile_config(
        updated_templates, updated_raw, old_config.get("frame_aspect")
    )

    # Merge so unrelated keys survive, then write data before config.
    config = dict(old_config)
    config.update(new_config)
    config["last_template_update"] = worst_idx + 1
    config["face_enrolled"] = os.path.exists(os.path.join(user_dir, "face_embedding.npy"))

    if updated_raw is not None:
        _atomic_save_npy(
            os.path.join(user_dir, f"gesture_{worst_idx + 1}_raw.npy"),
            updated_raw[worst_idx],
        )
    _atomic_save_npy(
        os.path.join(user_dir, f"gesture_{worst_idx + 1}.npy"), live_gesture
    )
    _atomic_write_json(config_path, config)

    return {
        "replaced_template": worst_idx + 1,
        "old_avg_distance": avg_distances[worst_idx],
        "new_avg_distance": live_avg,
        "new_threshold": config["threshold"],
        "new_consistency": config["consistency_score"],
    }


# ============================================================================
# STANDALONE TEST MODE
# ============================================================================

def main():
    """
    Standalone mode: compare saved gesture templates and show distances.
    """
    print()
    print("=" * 58)
    print("   GESTURE COMPARE — DTW Distance Calculator")
    print("=" * 58)
    print()

    users = list_registered_users()

    if len(users) == 0:
        print("  No registered users found.")
        print("  Run gesture_capture.py first to register gestures.")
        return

    print(f"  Registered users: {', '.join(users)}")
    print()

    # Show per-user info
    for user in users:
        templates = load_all_user_templates(user)
        threshold = load_user_threshold(user)
        finger_state_threshold = load_user_finger_state_threshold(user)
        config = _read_config(user)
        method = config.get("threshold_method", "legacy")
        consistency = config.get("consistency_score")

        line = (f"  {user}: {len(templates)} sample(s), "
                f"threshold = {threshold:.2f}, "
                f"finger limit = {finger_state_threshold:.2f}, "
                f"method = {method}")
        if consistency is not None:
            line += f", consistency = {consistency:.1f}/100"
        print(line)

    print()
    print("  Cross-user comparison (using first sample from each):")
    print("  " + "-" * 50)
    print(f"  {'User A':<12} {'User B':<12} {'Distance':>10}  {'Notes'}")
    print("  " + "-" * 50)

    for i, user_a in enumerate(users):
        templates_a = load_all_user_templates(user_a)
        for j, user_b in enumerate(users):
            if j < i:
                continue
            templates_b = load_all_user_templates(user_b)
            distance = compute_dtw_distance(templates_a[0], templates_b[0])
            marker = " <-- same user" if i == j else ""
            print(f"  {user_a:<12} {user_b:<12} {distance:>10.4f}  "
                  f"{marker}")

    print("  " + "-" * 50)
    print()


if __name__ == "__main__":
    main()
