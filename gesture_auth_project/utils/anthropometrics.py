"""
Anthropometric Hand Geometry Invariant Analysis (v2 - Posture-Invariant Metacarpal Engine)

Extracts scale-invariant and posture-invariant skeletal ratios from 3D hand landmarks.
Uses the rigid metacarpal palm plate (which does not bend or curl) and un-occluded
digit proportions, ensuring high stability across open-hand poses, closed fists,
and dynamic in-air signature writing gestures.

Key Invariant Ratios:
1. Palm Aspect Ratio: Palm Width (MCP 5 to MCP 17) / Palm Length (Wrist 0 to MCP 9)
2. Index Metacarpal Diagonal: Wrist 0 to Index MCP 5 / Palm Length
3. Pinky Metacarpal Diagonal: Wrist 0 to Pinky MCP 17 / Palm Length
4. Extended Index-to-Palm Proportion: Total Index (MCP 5 to Tip 8) / Palm Length
5. Thumb-to-Index Metacarpal Span: Thumb CMC/MCP (2) to Index MCP (5) / Palm Length
"""

import numpy as np

NUM_ANTHRO_FEATURES = 5

# Allowable deviation tolerance bounds
MIN_ANTHRO_TOLERANCE = 0.12
DEFAULT_ANTHRO_TOLERANCE = 0.15
MAX_ANTHRO_TOLERANCE = 0.25


def extract_frame_bone_ratios(landmarks):
    """
    Extract 5 posture-invariant skeletal bone ratios from a single (21, 3) frame.
    """
    p = landmarks

    # Rigid Palm Length: Wrist (0) to Middle MCP (9)
    palm_len = np.linalg.norm(p[9] - p[0])
    if palm_len < 1e-6:
        palm_len = 1e-6

    # Rigid Palm Width: Index MCP (5) to Pinky MCP (17)
    palm_width = np.linalg.norm(p[17] - p[5])

    # Index Metacarpal Diagonal: Wrist (0) to Index MCP (5)
    diag_idx = np.linalg.norm(p[5] - p[0])

    # Pinky Metacarpal Diagonal: Wrist (0) to Pinky MCP (17)
    diag_pky = np.linalg.norm(p[17] - p[0])

    # Extended Index Phalanx Span: Index MCP (5) to Index Tip (8)
    idx_tot = np.linalg.norm(p[8] - p[5])

    # Thumb-to-Index Metacarpal Span: Thumb MCP (2) to Index MCP (5)
    thumb_span = np.linalg.norm(p[5] - p[2])

    # Ratios normalized by Palm Length:
    r_palm = palm_width / palm_len
    r_diag_idx = diag_idx / palm_len
    r_diag_pky = diag_pky / palm_len
    r_idx_tot = idx_tot / palm_len
    r_thumb_span = thumb_span / palm_len

    return np.array([r_palm, r_diag_idx, r_diag_pky, r_idx_tot, r_thumb_span], dtype=np.float64)


def extract_gesture_anthropometric_signature(gesture_sequence):
    """
    Extract noise-filtered anthropometric signature across all frames
    using the median ratio vector.
    """
    n_frames = gesture_sequence.shape[0]
    frame_ratios = np.zeros((n_frames, NUM_ANTHRO_FEATURES), dtype=np.float64)

    for f in range(n_frames):
        frame_ratios[f] = extract_frame_bone_ratios(gesture_sequence[f])

    return np.median(frame_ratios, axis=0)


def calibrate_anthropometric_profile(templates):
    """
    Calibrate a user's anthropometric profile from multiple registration templates.
    """
    signatures = [
        extract_gesture_anthropometric_signature(t)
        for t in templates
    ]
    signatures = np.array(signatures)

    baseline = np.mean(signatures, axis=0)
    std = np.std(signatures, axis=0)

    # Inter-template maximum relative deviation
    max_dev = 0.0
    for s in signatures:
        rel_diff = np.abs(s - baseline) / (baseline + 1e-6)
        max_dev = max(max_dev, float(np.max(rel_diff)))

    # Calibrate tolerance with safe floor for dynamic gestures
    calibrated_tolerance = max(MIN_ANTHRO_TOLERANCE, max_dev * 1.6)
    calibrated_tolerance = min(MAX_ANTHRO_TOLERANCE, calibrated_tolerance)

    return {
        "baseline": [round(float(v), 4) for v in baseline],
        "std": [round(float(v), 4) for v in std],
        "tolerance": round(float(calibrated_tolerance), 4),
        "method": "metacarpal_invariant_v2",
    }


def compare_anthropometric_signatures(live_signature, baseline, tolerance=DEFAULT_ANTHRO_TOLERANCE):
    """
    Compare a live gesture's hand morphology signature against the registered baseline
    using robust multi-feature consensus.
    """
    live = np.array(live_signature, dtype=np.float64)
    base = np.array(baseline, dtype=np.float64)

    # Relative absolute deviation per feature
    rel_deviations = np.abs(live - base) / (base + 1e-6)
    mean_deviation = float(np.mean(rel_deviations))
    max_deviation = float(np.max(rel_deviations))

    # Multi-feature consensus:
    # 1. Overall mean deviation must be within tolerance
    # 2. At least 4 of 5 features must be within tolerance * 2.2
    # This prevents a single momentary tracking glitch on one landmark from killing authentic sessions.
    passing_features = sum(1 for d in rel_deviations if d <= tolerance * 2.2)
    matches = (mean_deviation <= tolerance) and (passing_features >= 4)

    # Confidence score: 1.0 when deviation is 0, decaying gracefully
    confidence = max(0.0, 1.0 - (mean_deviation / (tolerance * 1.6)))

    feature_names = [
        "palm_aspect",
        "diag_idx",
        "diag_pky",
        "idx_to_palm",
        "thumb_to_idx",
    ]
    details = {
        name: round(float(dev), 4)
        for name, dev in zip(feature_names, rel_deviations)
    }
    details["mean_deviation"] = round(mean_deviation, 4)
    details["max_deviation"] = round(max_deviation, 4)
    details["tolerance"] = round(tolerance, 4)
    details["confidence"] = round(confidence, 4)
    details["passing_features"] = f"{passing_features}/5"

    return matches, confidence, details
