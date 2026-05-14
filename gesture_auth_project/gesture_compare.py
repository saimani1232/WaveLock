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
# More samples = better accuracy but longer registration time.
NUM_REGISTRATION_SAMPLES = 3

# Fallback threshold used ONLY for old single-sample registrations
# that don't have a computed threshold. New registrations auto-compute
# their own threshold from the recorded samples.
DEFAULT_THRESHOLD = 5.0

# Minimum threshold floor — prevents unreasonably strict thresholds
# when a user is very consistent during registration.
MIN_THRESHOLD = 2.0

# Multiplier applied to the max pairwise distance between registration
# samples to compute the threshold. Higher = more forgiving.
# 1.5 means: "allow 50% more variation than seen during registration"
THRESHOLD_MULTIPLIER = 1.5


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


def compute_threshold_from_samples(samples):
    """
    Compute an optimal authentication threshold from registration samples.

    Strategy:
        1. Compute DTW distance between every pair of samples
        2. Take the maximum pairwise distance (= worst natural variation)
        3. Multiply by THRESHOLD_MULTIPLIER to add safety margin
        4. Enforce MIN_THRESHOLD floor to prevent being too strict

    Args:
        samples: list of numpy arrays, each of shape (N, 21, 3).

    Returns:
        tuple of (float, list):
            - float: The computed threshold
            - list: All pairwise distances (for display/debugging)
    """
    pairwise_distances = []

    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            d = compute_dtw_distance(samples[i], samples[j])
            pairwise_distances.append(d)

    if not pairwise_distances:
        return DEFAULT_THRESHOLD, []

    max_distance = max(pairwise_distances)
    threshold = max(MIN_THRESHOLD, max_distance * THRESHOLD_MULTIPLIER)

    return threshold, pairwise_distances


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


# ============================================================================
# AUTHENTICATION FUNCTIONS
# ============================================================================

def authenticate(live_gesture, stored_templates, threshold):
    """
    Authenticate a live gesture against stored templates.

    Compares the live gesture against ALL stored templates and uses
    the MINIMUM distance (best match). This is more forgiving than
    comparing against a single template because different registration
    samples capture different natural variations.

    Args:
        live_gesture: numpy array of shape (N, 21, 3).
        stored_templates: list of numpy arrays (one per registration sample).
        threshold: float, maximum DTW distance to accept.

    Returns:
        tuple of (bool, float, int):
            - bool: True if access granted, False if denied.
            - float: The minimum DTW distance found (best match).
            - int: Index of the best-matching template.
    """
    min_distance = float('inf')
    best_idx = 0

    for i, template in enumerate(stored_templates):
        distance = compute_dtw_distance(live_gesture, template)
        if distance < min_distance:
            min_distance = distance
            best_idx = i

    granted = min_distance <= threshold

    return granted, min_distance, best_idx


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

    # Save config with threshold and metadata
    config = {
        "threshold": round(threshold, 4),
        "num_samples": len(samples),
        "pairwise_distances": [round(d, 4) for d in pairwise_distances],
        "max_pairwise_distance": round(max(pairwise_distances), 4)
        if pairwise_distances else 0.0,
        "threshold_multiplier": THRESHOLD_MULTIPLIER,
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
        print(f"  {user}: {len(templates)} sample(s), "
              f"threshold = {threshold:.2f}")

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
