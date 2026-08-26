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
import sys
import json

import numpy as np
from dtaidistance import dtw_ndim


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


def compute_threshold_details(pairwise_distances):
    """
    Compute a robust per-user threshold and diagnostic metadata.

    The old project used max(pairwise_distances) * 1.5. That works, but
    it can become too permissive when one registration sample is noisy.
    This statistical method combines:
        - mean + standard deviation for normal user variation
        - percentile + margin for robust tolerance
        - max + smaller margin as a hard cap

    Args:
        pairwise_distances: list of genuine intra-user DTW distances.

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
            "percentile_distance": 0.0,
            "max_pairwise_distance": 0.0,
            "statistical_threshold": DEFAULT_THRESHOLD,
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

    statistical_threshold = mean_distance + (
        THRESHOLD_STD_FACTOR * std_distance
    )
    percentile_threshold = percentile_distance * THRESHOLD_SAFETY_MARGIN
    max_margin_threshold = max_distance * THRESHOLD_MAX_MARGIN
    legacy_threshold = max_distance * THRESHOLD_MULTIPLIER

    # Use the stricter of the statistically reasonable candidates. This
    # keeps genuine variation covered while reducing outlier-driven looseness.
    candidate_threshold = min(
        max(statistical_threshold, percentile_threshold),
        max_margin_threshold,
        legacy_threshold,
    )
    threshold = max(MIN_THRESHOLD, candidate_threshold)

    if mean_distance > 1e-6:
        consistency_score = max(
            0.0,
            100.0 * (1.0 - (std_distance / mean_distance))
        )
    else:
        consistency_score = 100.0

    return {
        "threshold": float(threshold),
        "method": THRESHOLD_METHOD,
        "mean_distance": mean_distance,
        "std_distance": std_distance,
        "median_distance": median_distance,
        "percentile_distance": percentile_distance,
        "max_pairwise_distance": max_distance,
        "statistical_threshold": float(statistical_threshold),
        "percentile_threshold": float(percentile_threshold),
        "max_margin_threshold": float(max_margin_threshold),
        "legacy_threshold": float(max(MIN_THRESHOLD, legacy_threshold)),
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
    config_path = os.path.join(templates_dir, username, "config.json")

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config.get("threshold", DEFAULT_THRESHOLD)

    return DEFAULT_THRESHOLD


def load_user_finger_state_threshold(username, templates_dir=TEMPLATES_DIR):
    """
    Load the per-user finger-state mismatch threshold from config.json.

    If a user was registered before this security gate existed, derive a
    threshold from their saved samples when multiple templates are present.
    Single-sample legacy users fall back to DEFAULT_FINGER_STATE_THRESHOLD.
    """
    config_path = os.path.join(templates_dir, username, "config.json")

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
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
    config_path = os.path.join(templates_dir, username, "config.json")

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
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
    config_path = os.path.join(templates_dir, username, "config.json")

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
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


# ============================================================================
# AUTHENTICATION FUNCTIONS
# ============================================================================

def authenticate_with_details(live_gesture, stored_templates, threshold,
                              finger_state_threshold=None,
                              transition_threshold=None,
                              segment_threshold=None):
    """
    Authenticate a live gesture against stored templates.

    Compares the live gesture against ALL stored templates and uses
    the best template that passes ALL FOUR security gates:
        1. DTW movement distance <= threshold
        2. Finger-state average mismatch <= finger_state_threshold
        3. Finger transition order dissimilarity <= transition_threshold
        4. Segment max finger mismatch <= segment_threshold

    Gates 3 and 4 were added to catch gestures where the overall motion
    looks similar but the ORDER of finger raises is wrong (e.g., doing
    "1-2-3" instead of "1-2-4-3").

    Args:
        live_gesture: numpy array of shape (N, 21, 3).
        stored_templates: list of numpy arrays (one per registration sample).
        threshold: float, maximum DTW distance to accept.
        finger_state_threshold: float, maximum finger-state mismatch rate.
        transition_threshold: float, maximum transition edit distance.
        segment_threshold: float, maximum per-segment finger mismatch.

    Returns:
        tuple of (bool, float, int, dict):
            - bool: True if access granted, False if denied.
            - float: The minimum DTW distance found (best match).
            - int: Index of the best-matching template.
            - dict: Diagnostic details about all security gates.
    """
    if finger_state_threshold is None:
        finger_state_threshold = DEFAULT_FINGER_STATE_THRESHOLD
    if transition_threshold is None:
        transition_threshold = DEFAULT_TRANSITION_THRESHOLD
    if segment_threshold is None:
        segment_threshold = DEFAULT_SEGMENT_THRESHOLD

    comparisons = []

    for i, template in enumerate(stored_templates):
        distance = compute_dtw_distance(live_gesture, template)
        finger_mismatch = compute_finger_state_mismatch(
            live_gesture, template
        )
        transition_dissim = compute_transition_dissimilarity(
            live_gesture, template
        )
        segment_max = compute_segment_max_mismatch(
            live_gesture, template
        )

        comparisons.append({
            "sample_index": i,
            "distance": distance,
            "finger_state_mismatch": finger_mismatch,
            "transition_dissimilarity": transition_dissim,
            "segment_max_mismatch": segment_max,
            "passes_distance": distance <= threshold,
            "passes_finger_state": (
                finger_mismatch <= finger_state_threshold
            ),
            "passes_transition": (
                transition_dissim <= transition_threshold
            ),
            "passes_segment": (
                segment_max <= segment_threshold
            ),
        })

    # A template passes only if ALL FOUR gates are satisfied.
    passing = [
        item for item in comparisons
        if (item["passes_distance"]
            and item["passes_finger_state"]
            and item["passes_transition"]
            and item["passes_segment"])
    ]

    if passing:
        best = min(passing, key=lambda item: item["distance"])
        granted = True
        failure_reason = None
    else:
        best = min(comparisons, key=lambda item: item["distance"])
        granted = False

        # Build failure reason listing every gate that failed.
        failed_gates = []
        if not best["passes_distance"]:
            failed_gates.append("distance")
        if not best["passes_finger_state"]:
            failed_gates.append("finger_state")
        if not best["passes_transition"]:
            failed_gates.append("transition_order")
        if not best["passes_segment"]:
            failed_gates.append("segment_mismatch")
        failure_reason = "_and_".join(failed_gates) if failed_gates else "unknown"

    details = {
        "distance": best["distance"],
        "threshold": threshold,
        "finger_state_mismatch": best["finger_state_mismatch"],
        "finger_state_threshold": finger_state_threshold,
        "transition_dissimilarity": best["transition_dissimilarity"],
        "transition_threshold": transition_threshold,
        "segment_max_mismatch": best["segment_max_mismatch"],
        "segment_threshold": segment_threshold,
        "passes_distance": best["passes_distance"],
        "passes_finger_state": best["passes_finger_state"],
        "passes_transition": best["passes_transition"],
        "passes_segment": best["passes_segment"],
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
        if not os.path.isdir(user_dir):
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

def save_registration(username, samples, threshold, pairwise_distances,
                      templates_dir=TEMPLATES_DIR):
    """
    Save a complete multi-sample registration to disk.

    Saves each gesture sample as gesture_1.npy, gesture_2.npy, etc.,
    and a config.json file containing the computed threshold.

    Args:
        username: str, the user's identifier.
        samples: list of numpy arrays, each of shape (N, 21, 3).
        threshold: float, the computed authentication threshold.
        pairwise_distances: list of float, distances between samples.
        templates_dir: str, path to templates root directory.

    Returns:
        str: Path to the user's template directory.
    """
    user_dir = os.path.join(templates_dir, username)
    os.makedirs(user_dir, exist_ok=True)

    # Remove old files if re-registering
    for old_file in os.listdir(user_dir):
        os.remove(os.path.join(user_dir, old_file))

    # Save each sample
    for i, sample in enumerate(samples, start=1):
        filepath = os.path.join(user_dir, f"gesture_{i}.npy")
        np.save(filepath, sample)

    threshold_details = compute_threshold_details(pairwise_distances)
    finger_state_details = compute_finger_state_threshold_details(samples)
    transition_details = compute_transition_threshold_details(samples)
    segment_details = compute_segment_threshold_details(samples)

    # Save config with threshold and metadata
    config = {
        "threshold": round(threshold, 4),
        "num_samples": len(samples),
        "threshold_method": threshold_details["method"],
        "finger_state_threshold": round(
            finger_state_details["threshold"], 4
        ),
        "finger_state_method": finger_state_details["method"],
        "finger_state_pairwise_mismatches": [
            round(m, 4)
            for m in finger_state_details["pairwise_mismatches"]
        ],
        "finger_state_mean_mismatch": round(
            finger_state_details["mean_mismatch"], 4
        ),
        "finger_state_max_mismatch": round(
            finger_state_details["max_mismatch"], 4
        ),
        "finger_state_margin": FINGER_STATE_MARGIN,
        "finger_state_min_threshold": MIN_FINGER_STATE_THRESHOLD,
        "finger_state_max_threshold": MAX_FINGER_STATE_THRESHOLD,
        "pairwise_distances": [round(d, 4) for d in pairwise_distances],
        "mean_pairwise_distance": round(
            threshold_details["mean_distance"], 4
        ),
        "std_pairwise_distance": round(
            threshold_details["std_distance"], 4
        ),
        "median_pairwise_distance": round(
            threshold_details["median_distance"], 4
        ),
        "percentile_pairwise_distance": round(
            threshold_details["percentile_distance"], 4
        ),
        "max_pairwise_distance": round(
            threshold_details["max_pairwise_distance"], 4
        ),
        "statistical_threshold": round(
            threshold_details["statistical_threshold"], 4
        ),
        "percentile_threshold": round(
            threshold_details["percentile_threshold"], 4
        ),
        "max_margin_threshold": round(
            threshold_details["max_margin_threshold"], 4
        ),
        "legacy_threshold": round(threshold_details["legacy_threshold"], 4),
        "consistency_score": round(
            threshold_details["consistency_score"], 2
        ),
        "threshold_std_factor": THRESHOLD_STD_FACTOR,
        "threshold_percentile": THRESHOLD_PERCENTILE,
        "threshold_safety_margin": THRESHOLD_SAFETY_MARGIN,
        "threshold_max_margin": THRESHOLD_MAX_MARGIN,
        "threshold_multiplier": THRESHOLD_MULTIPLIER,
        "transition_threshold": round(
            transition_details["threshold"], 4
        ),
        "transition_method": transition_details["method"],
        "transition_pairwise_dissimilarities": [
            round(d, 4)
            for d in transition_details["pairwise_dissimilarities"]
        ],
        "transition_mean_dissimilarity": round(
            transition_details["mean_dissimilarity"], 4
        ),
        "transition_max_dissimilarity": round(
            transition_details["max_dissimilarity"], 4
        ),
        "transition_margin": TRANSITION_MARGIN,
        "segment_threshold": round(
            segment_details["threshold"], 4
        ),
        "segment_method": segment_details["method"],
        "segment_pairwise_max_mismatches": [
            round(m, 4)
            for m in segment_details["pairwise_max_mismatches"]
        ],
        "segment_mean_max_mismatch": round(
            segment_details["mean_max_mismatch"], 4
        ),
        "segment_max_max_mismatch": round(
            segment_details["max_max_mismatch"], 4
        ),
        "segment_margin": SEGMENT_MARGIN,
        "segment_count": SEGMENT_COUNT,
    }

    config_path = os.path.join(user_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    return user_dir


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
        config_path = os.path.join(TEMPLATES_DIR, user, "config.json")
        method = "legacy"
        consistency = None

        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
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
