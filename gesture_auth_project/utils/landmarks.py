"""
MediaPipe Hand Landmark Extraction Helpers

Provides utility functions to extract hand landmark data from
MediaPipe results and convert them into NumPy arrays suitable
for gesture recording and DTW comparison.

Each hand has 21 landmarks, each with (x, y, z) coordinates:
  - x, y: Normalized to [0.0, 1.0] relative to frame dimensions
  - z: Depth relative to the wrist (negative = closer to camera)
"""

import numpy as np


def extract_landmarks(hand_landmarks):
    """
    Convert a single hand's MediaPipe landmarks to a NumPy array.

    Args:
        hand_landmarks: A single hand's landmark object from MediaPipe.
                        This is one element from result.multi_hand_landmarks.

    Returns:
        numpy.ndarray of shape (21, 3) with (x, y, z) coordinates
        for each of the 21 hand landmarks.
    """
    landmarks = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float64
    )
    return landmarks


def is_hand_detected(results):
    """
    Check whether MediaPipe detected at least one hand in the frame.

    Args:
        results: The full result object from MediaPipe Hands processing.

    Returns:
        bool: True if at least one hand was detected, False otherwise.
    """
    return (results.multi_hand_landmarks is not None
            and len(results.multi_hand_landmarks) > 0)


def get_first_hand_landmarks(results):
    """
    Extract landmarks for the first detected hand from a MediaPipe result.

    Convenience function that combines detection check and extraction.

    Args:
        results: The full result object from MediaPipe Hands processing.

    Returns:
        numpy.ndarray of shape (21, 3) if a hand is detected, or None.
    """
    if not is_hand_detected(results):
        return None
    return extract_landmarks(results.multi_hand_landmarks[0])
