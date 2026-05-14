"""
Gesture Normalization Utilities

This module handles the two critical normalization steps that make
gesture comparison reliable across different recording conditions:

1. SPATIAL NORMALIZATION (per-frame)
   - Translates all landmarks so the wrist is at origin (0, 0, 0)
   - Scales all landmarks so the hand occupies a unit bounding sphere
   - This makes gestures invariant to: hand position in frame, distance
     from camera, and hand size differences between users

2. TEMPORAL NORMALIZATION (across frames)
   - Resamples the gesture sequence to a fixed number of frames
   - Uses linear interpolation along the time axis
   - This makes gestures invariant to: recording duration, gesture speed,
     and frame rate variations

Without these normalizations, DTW comparison would fail whenever the user
sits at a different distance from the camera or performs the gesture at a
slightly different speed.
"""

import numpy as np
from scipy.interpolate import interp1d


def normalize_spatial(frame_landmarks):
    """
    Apply spatial normalization to a single frame of hand landmarks.

    Step 1: Translate so wrist (landmark 0) is at the origin.
    Step 2: Scale so the maximum distance from wrist equals 1.0.

    Args:
        frame_landmarks: numpy array of shape (21, 3) — one frame of
                         raw (x, y, z) coordinates for 21 landmarks.

    Returns:
        numpy array of shape (21, 3) — normalized coordinates.
    """
    normalized = frame_landmarks.copy()

    # Step 1: Translate — place wrist at origin
    wrist = normalized[0].copy()
    normalized -= wrist

    # Step 2: Scale — normalize by the maximum distance from wrist
    # This makes the hand size consistent regardless of camera distance
    distances_from_wrist = np.linalg.norm(normalized, axis=1)
    max_distance = np.max(distances_from_wrist)

    if max_distance > 1e-6:
        normalized /= max_distance
    # If max_distance is ~0, hand is basically a single point — skip scaling

    return normalized


def normalize_temporal(gesture_sequence, target_length=60):
    """
    Resample a gesture sequence to a fixed number of frames using
    linear interpolation.

    This ensures that two recordings of the same gesture — even if one
    took 2.5 seconds and the other took 3.5 seconds — will both be
    represented as exactly `target_length` frames, making DTW comparison
    fair and accurate.

    Args:
        gesture_sequence: numpy array of shape (N, 21, 3) where N is
                          the original number of recorded frames.
        target_length: int, desired number of frames in the output.
                       Default is 60 (a good balance of detail vs speed).

    Returns:
        numpy array of shape (target_length, 21, 3).
    """
    n_frames = gesture_sequence.shape[0]

    # Already the right length — return a copy
    if n_frames == target_length:
        return gesture_sequence.copy()

    # Edge case: only one frame — repeat it
    if n_frames < 2:
        return np.tile(gesture_sequence, (target_length, 1, 1))

    # Flatten the spatial dimensions: (N, 21, 3) -> (N, 63)
    # This lets us interpolate all 63 features independently along time
    n_landmarks = gesture_sequence.shape[1]
    n_coords = gesture_sequence.shape[2]
    n_features = n_landmarks * n_coords  # 21 * 3 = 63

    flat_sequence = gesture_sequence.reshape(n_frames, n_features)

    # Create normalized time indices for original and target
    original_t = np.linspace(0.0, 1.0, n_frames)
    target_t = np.linspace(0.0, 1.0, target_length)

    # Interpolate each feature along the time axis
    interpolator = interp1d(
        original_t, flat_sequence, axis=0, kind='linear'
    )
    resampled_flat = interpolator(target_t)

    # Reshape back to (target_length, 21, 3)
    resampled = resampled_flat.reshape(target_length, n_landmarks, n_coords)

    return resampled


def normalize_gesture(gesture_sequence, target_length=60):
    """
    Full normalization pipeline for a recorded gesture sequence.

    Applies spatial normalization to every individual frame, then
    resamples the entire sequence to a fixed temporal length.

    This is the main function to call after recording a gesture
    and before saving it as a template or comparing it via DTW.

    Args:
        gesture_sequence: numpy array of shape (N, 21, 3) — raw
                          recorded landmark data from MediaPipe.
        target_length: int, desired number of output frames.
                       Default is 60.

    Returns:
        Fully normalized numpy array of shape (target_length, 21, 3).

    Example:
        raw_frames = [...]  # list of (21, 3) arrays from recording
        raw_gesture = np.array(raw_frames)           # (N, 21, 3)
        normalized = normalize_gesture(raw_gesture)   # (60, 21, 3)
        np.save("templates/alice/gesture.npy", normalized)
    """
    if gesture_sequence.ndim != 3 or gesture_sequence.shape[1:] != (21, 3):
        raise ValueError(
            f"Expected shape (N, 21, 3), got {gesture_sequence.shape}. "
            f"Each frame must have 21 landmarks with 3 coordinates each."
        )

    if gesture_sequence.shape[0] == 0:
        raise ValueError("Cannot normalize an empty gesture sequence.")

    # Step 1: Spatial normalization — per frame
    spatially_normalized = np.array([
        normalize_spatial(frame) for frame in gesture_sequence
    ])

    # Step 2: Temporal normalization — across frames
    fully_normalized = normalize_temporal(
        spatially_normalized, target_length
    )

    return fully_normalized
