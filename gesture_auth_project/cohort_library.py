"""
Cohort Library — Synthetic Impostor Gestures for Negative Sampling

Generates a set of standard synthetic gestures that serve as a
"cohort" for calibration during registration.  By comparing the user's
genuine gesture against known impostor patterns, the system can detect
when a chosen gesture is too simple or too similar to common hand poses.

The cohort gestures are generated programmatically using idealized
landmark positions, so no pre-recorded .npy files need to be shipped.

Usage (called automatically during registration):
    from cohort_library import generate_cohort_library, validate_gesture_uniqueness
"""

import numpy as np


# ============================================================================
# IDEALIZED HAND LANDMARKS
# ============================================================================

# Approximate 3D positions for 21 hand landmarks in a "rest" pose.
# These are in the normalized coordinate space (wrist at origin, unit sphere).
# The positions are approximate but structurally correct for generating
# meaningful synthetic gestures.

# Finger base (MCP) positions relative to wrist
_MCP_POSITIONS = {
    "thumb":  np.array([0.25, -0.35, -0.05]),
    "index":  np.array([0.10, -0.55, -0.02]),
    "middle": np.array([0.00, -0.58, -0.01]),
    "ring":   np.array([-0.10, -0.55, -0.02]),
    "pinky":  np.array([-0.20, -0.48, -0.04]),
}

# Joint offsets for extended vs curled fingers
_EXTENDED_OFFSETS = [
    np.array([0.00, -0.12, 0.00]),   # MCP → PIP
    np.array([0.00, -0.11, 0.00]),   # PIP → DIP
    np.array([0.00, -0.10, 0.00]),   # DIP → TIP
]

_CURLED_OFFSETS = [
    np.array([0.00, -0.04, 0.06]),   # MCP → PIP (curled inward)
    np.array([0.00, 0.02, 0.08]),    # PIP → DIP (folded back)
    np.array([0.00, 0.04, 0.06]),    # DIP → TIP (tucked in)
]


def _build_hand_frame(finger_states):
    """
    Build a single (21, 3) landmark frame from finger extension states.

    Args:
        finger_states: dict mapping finger names to bool (True = extended).

    Returns:
        numpy array of shape (21, 3).
    """
    landmarks = np.zeros((21, 3), dtype=np.float64)

    # Landmark 0: wrist at origin
    landmarks[0] = [0.0, 0.0, 0.0]

    finger_indices = {
        "thumb":  (1, 2, 3, 4),
        "index":  (5, 6, 7, 8),
        "middle": (9, 10, 11, 12),
        "ring":   (13, 14, 15, 16),
        "pinky":  (17, 18, 19, 20),
    }

    for finger_name, (mcp_idx, pip_idx, dip_idx, tip_idx) in finger_indices.items():
        mcp = _MCP_POSITIONS[finger_name]
        landmarks[mcp_idx] = mcp

        offsets = _EXTENDED_OFFSETS if finger_states.get(finger_name, False) else _CURLED_OFFSETS

        landmarks[pip_idx] = mcp + offsets[0]
        landmarks[dip_idx] = mcp + offsets[0] + offsets[1]
        landmarks[tip_idx] = mcp + offsets[0] + offsets[1] + offsets[2]

    # Normalize to unit sphere
    distances = np.linalg.norm(landmarks, axis=1)
    max_dist = np.max(distances)
    if max_dist > 1e-6:
        landmarks /= max_dist

    return landmarks


def _make_static_gesture(finger_states, n_frames=60):
    """Create a gesture where the hand holds a static pose for all frames."""
    frame = _build_hand_frame(finger_states)
    # Add tiny noise to avoid DTW distance of exactly 0
    gesture = np.tile(frame, (n_frames, 1, 1))
    noise = np.random.RandomState(42).normal(0, 0.002, gesture.shape)
    return gesture + noise


def _make_sequential_gesture(finger_order, n_frames=60):
    """
    Create a gesture where fingers are raised one at a time in order.

    Args:
        finger_order: list of finger names in the order they should be raised.
        n_frames: total frames.
    """
    all_fingers = ["thumb", "index", "middle", "ring", "pinky"]
    segment_len = n_frames // (len(finger_order) + 1)

    frames = []
    active = {f: False for f in all_fingers}

    # Initial segment: all curled
    for _ in range(segment_len):
        frames.append(_build_hand_frame(active))

    # Raise each finger in order
    for finger in finger_order:
        active[finger] = True
        for _ in range(segment_len):
            frames.append(_build_hand_frame(active))

    # Pad to n_frames
    while len(frames) < n_frames:
        frames.append(_build_hand_frame(active))

    gesture = np.array(frames[:n_frames], dtype=np.float64)
    noise = np.random.RandomState(123).normal(0, 0.002, gesture.shape)
    return gesture + noise


def _make_wave_gesture(n_frames=60):
    """Create a gesture where all fingers open and close together."""
    frames = []
    all_fingers = ["thumb", "index", "middle", "ring", "pinky"]

    for i in range(n_frames):
        # Oscillate between open and closed every 15 frames
        is_open = (i // 15) % 2 == 0
        state = {f: is_open for f in all_fingers}
        frames.append(_build_hand_frame(state))

    gesture = np.array(frames, dtype=np.float64)
    noise = np.random.RandomState(77).normal(0, 0.002, gesture.shape)
    return gesture + noise


def _make_flutter_gesture(n_frames=60, seed=999):
    """Create a gesture with random finger state changes."""
    rng = np.random.RandomState(seed)
    all_fingers = ["thumb", "index", "middle", "ring", "pinky"]
    frames = []

    for i in range(n_frames):
        state = {f: bool(rng.randint(0, 2)) for f in all_fingers}
        frames.append(_build_hand_frame(state))

    return np.array(frames, dtype=np.float64)


# ============================================================================
# PUBLIC API
# ============================================================================

def generate_cohort_library():
    """
    Generate the standard cohort library of 6 synthetic impostor gestures.

    Returns:
        list of numpy arrays, each of shape (60, 21, 3).
    """
    return [
        _make_static_gesture(  # 1. Static open hand
            {"thumb": True, "index": True, "middle": True,
             "ring": True, "pinky": True}
        ),
        _make_static_gesture(  # 2. Static fist
            {"thumb": False, "index": False, "middle": False,
             "ring": False, "pinky": False}
        ),
        _make_wave_gesture(),  # 3. All fingers wave
        _make_sequential_gesture(  # 4. Sequential forward
            ["index", "middle", "ring", "pinky"]
        ),
        _make_sequential_gesture(  # 5. Sequential backward
            ["pinky", "ring", "middle", "index"]
        ),
        _make_flutter_gesture(),  # 6. Random flutter
    ]


def compute_impostor_distances(user_samples, cohort, compute_dtw_fn):
    """
    Compute DTW distances from user samples to cohort gestures.

    Args:
        user_samples: list of numpy arrays (user's registration samples).
        cohort: list of numpy arrays (synthetic cohort gestures).
        compute_dtw_fn: callable, the DTW distance function.

    Returns:
        list of float: minimum distance from each cohort gesture to any
        user sample (i.e., how close the closest impostor is).
    """
    min_distances = []

    for cohort_gesture in cohort:
        min_d = float('inf')
        for sample in user_samples:
            d = compute_dtw_fn(sample, cohort_gesture)
            min_d = min(min_d, d)
        min_distances.append(min_d)

    return min_distances


def validate_gesture_uniqueness(user_samples, cohort, intra_user_max,
                                compute_dtw_fn, min_ratio=1.5):
    """
    Check whether the user's gesture is sufficiently different from
    standard cohort gestures.

    Args:
        user_samples: list of numpy arrays (user's registration samples).
        cohort: list of numpy arrays (synthetic cohort gestures).
        intra_user_max: float, maximum pairwise DTW distance among user samples.
        compute_dtw_fn: callable, the DTW distance function.
        min_ratio: float, minimum ratio of impostor distance to intra-user max.

    Returns:
        dict with:
            - is_unique: bool, True if gesture is unique enough.
            - min_impostor_distance: float, closest cohort distance.
            - ratio: float, impostor_dist / intra_user_max.
            - warning: str or None, warning message if not unique.
    """
    impostor_dists = compute_impostor_distances(
        user_samples, cohort, compute_dtw_fn
    )
    min_impostor = min(impostor_dists) if impostor_dists else float('inf')

    if intra_user_max > 1e-6:
        ratio = min_impostor / intra_user_max
    else:
        ratio = float('inf')

    is_unique = ratio >= min_ratio

    warning = None
    if not is_unique:
        warning = (
            "Your gesture may be too simple or common. "
            "Consider a more dynamic or unique gesture for better security."
        )

    return {
        "is_unique": is_unique,
        "min_impostor_distance": float(min_impostor),
        "impostor_distances": [float(d) for d in impostor_dists],
        "ratio": float(ratio),
        "warning": warning,
    }
