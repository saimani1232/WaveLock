"""
Kinematic Behavioral Rhythm and Jerk Analysis

Extracts neuromuscular motion dynamics from gesture sequences to distinguish
the genuine user's fluid, subconscious muscle-memory execution from an
observer's hesitant, conscious imitation.

In human motor control (Flash & Hogan minimum-jerk model), well-practiced
gestures exhibit smooth, bell-shaped velocity profiles and low jerk (derivative
of acceleration). An impostor mimicking a gesture sequence hesitates and makes
subconscious micro-corrections, resulting in elevated jerk and mismatched
velocity peaks.

Key Kinematic Features:
1. Fingertip Velocity Profile: Frame-to-frame velocity envelope across active fingertips (8, 12, 16, 20).
2. Kinematic Jerk Metric: Mean squared jerk measuring movement smoothness vs hesitation.
3. Velocity Energy Distribution: Kinetic energy distribution across the temporal phases of the gesture.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

from utils.normalize import normalize_spatial

# Active fingertips: Index (8), Middle (12), Ring (16), Pinky (20)
FINGERTIP_INDICES = [8, 12, 16, 20]

# Default kinematic dissimilarity tolerance
DEFAULT_KINEMATIC_TOLERANCE = 0.35

# Profiles computed from raw (un-resampled) landmarks carry this method tag.
# Profiles without a "method" key are legacy: their baseline was computed
# from the 60-frame linearly resampled templates, so the live signature must
# be computed the same way to stay comparable.
KINEMATIC_METHOD_SMOOTHED = "smoothed_raw_v2"

# Savitzky-Golay smoothing window as a fraction of the recorded frame count.
# Differentiating raw MediaPipe landmarks three times amplifies tracking
# jitter until it dominates the jerk value; linear resampling to 60 frames
# adds interpolation kinks whose size depends on the recorded frame count.
# Smoothing at the native rate, then cubic resampling, removes both.
# ponytail: 0.15 tuned on synthetic jitter (sigma=0.003); retune on real data.
SMOOTHING_WINDOW_FRACTION = 0.15
SMOOTHING_POLYORDER = 3


def prepare_kinematic_sequence(raw_sequence, target_length=60):
    """
    Build the sequence used for kinematic features from RAW landmarks.

    Frames are spatially normalised (same space as the pose templates),
    smoothed along time at the native frame rate, then resampled to
    target_length with a cubic spline (no piecewise-linear kinks).

    Args:
        raw_sequence: raw MediaPipe landmarks, shape (N, 21, 3).
        target_length: frames after resampling.

    Returns:
        numpy array of shape (target_length, 21, 3).
    """
    raw = np.asarray(raw_sequence, dtype=np.float64)
    if raw.ndim != 3 or raw.shape[1:] != (21, 3) or raw.shape[0] == 0:
        raise ValueError(
            f"Expected raw landmarks of shape (N, 21, 3), got {raw.shape}."
        )

    spatial = np.array([normalize_spatial(frame) for frame in raw])
    n_frames = spatial.shape[0]
    if n_frames < 2:
        return np.tile(spatial, (target_length, 1, 1))

    window = max(5, int(n_frames * SMOOTHING_WINDOW_FRACTION) | 1)
    if window > n_frames:
        window = n_frames if n_frames % 2 == 1 else n_frames - 1
    if window > SMOOTHING_POLYORDER:
        spatial = savgol_filter(
            spatial, window, SMOOTHING_POLYORDER, axis=0
        )

    t_src = np.linspace(0.0, 1.0, n_frames)
    t_dst = np.linspace(0.0, 1.0, target_length)
    return CubicSpline(t_src, spatial, axis=0)(t_dst)


def extract_velocity_profile(gesture_sequence):
    """
    Compute the per-frame velocity magnitude summed across active fingertips.

    Args:
        gesture_sequence: numpy array of shape (N, 21, 3).

    Returns:
        numpy array of shape (N-1,) with velocity magnitudes.
    """
    # Differences between consecutive frames for fingertips
    diffs = np.diff(gesture_sequence[:, FINGERTIP_INDICES, :], axis=0)  # (N-1, 4, 3)
    # Speed per fingertip
    speeds = np.linalg.norm(diffs, axis=2)  # (N-1, 4)
    # Aggregate fingertip velocity envelope
    velocity_envelope = np.mean(speeds, axis=1)  # (N-1,)
    return velocity_envelope


def extract_jerk_metric(gesture_sequence):
    """
    Compute discrete Mean Absolute Jerk (MAJ) from the fingertip velocity profile.
    Fluent muscle memory produces lower jerk; hesitant imitation produces higher jerk.
    Evaluated as a ratio relative to the user's calibrated baseline (jerk_ratio <= 1.75).

    Args:
        gesture_sequence: numpy array of shape (N, 21, 3).

    Returns:
        float: mean absolute jerk value.
    """
    vel = extract_velocity_profile(gesture_sequence)  # (N-1,)
    if len(vel) < 3:
        return 0.0

    acc = np.diff(vel)   # (N-2,)
    jerk = np.diff(acc)  # (N-3,)

    mean_jerk = float(np.mean(np.abs(jerk)))
    return mean_jerk


def extract_mid_gesture_dwell(gesture_sequence, stillness_ratio=0.20):
    """
    Measure the longest contiguous window of near-zero velocity during the active
    middle phase of the gesture (frames 10 to 50 of 60 frames).

    A genuine, well-practiced gesture maintains continuous fluid motion.
    An observer who hesitates ("pause-to-recall" partway through) produces
    an unnatural velocity plateau/flatline that is not caught by jerk alone.
    """
    vel = extract_velocity_profile(gesture_sequence)
    if len(vel) < 20:
        return 0

    mean_v = float(np.mean(vel))
    if mean_v < 1e-6:
        return len(vel)

    # Active middle window (frames 10 to 50)
    mid_vel = vel[10:50]
    mid_still = mid_vel < (stillness_ratio * mean_v)

    max_dwell, cur_dwell = 0, 0
    for is_still in mid_still:
        if is_still:
            cur_dwell += 1
            max_dwell = max(max_dwell, cur_dwell)
        else:
            cur_dwell = 0

    return max_dwell


def extract_kinematic_signature(gesture_sequence):
    """
    Extract a compact behavioral kinematic signature for the gesture sequence.

    Args:
        gesture_sequence: numpy array of shape (N, 21, 3).

    Returns:
        dict containing:
            - binned_velocity: list of float (binned velocity profile, 10 bins)
            - jerk: float
            - peak_phase: float (normalized time of maximum velocity [0.0, 1.0])
            - mean_speed: float
            - max_mid_dwell: int (longest contiguous pause frames during motion)
    """
    vel = extract_velocity_profile(gesture_sequence)
    jerk = extract_jerk_metric(gesture_sequence)
    max_mid_dwell = extract_mid_gesture_dwell(gesture_sequence)

    # Bin the velocity curve into 10 temporal phases to compare rhythm
    n = len(vel)
    num_bins = 10
    bin_size = max(1, n // num_bins)
    binned_vel = []
    for b in range(num_bins):
        start = b * bin_size
        end = (b + 1) * bin_size if b < num_bins - 1 else n
        binned_vel.append(float(np.mean(vel[start:end]) if end > start else 0.0))

    # Normalize binned velocity so it reflects rhythm shape rather than absolute camera distance
    vel_sum = sum(binned_vel)
    if vel_sum > 1e-6:
        binned_vel = [v / vel_sum for v in binned_vel]

    peak_frame = int(np.argmax(vel)) if len(vel) > 0 else 0
    peak_phase = float(peak_frame / max(1, n - 1))
    mean_speed = float(np.mean(vel)) if len(vel) > 0 else 0.0

    return {
        "binned_velocity": [round(v, 4) for v in binned_vel],
        "jerk": float(f"{jerk:.6g}"),  # significant digits: jerk can be ~1e-6
        "peak_phase": round(peak_phase, 4),
        "mean_speed": round(mean_speed, 4),
        "max_mid_dwell": max_mid_dwell,
    }


def calibrate_kinematic_profile(templates, method=None):
    """
    Calibrate a user's behavioral kinematic profile from registration templates.

    Args:
        templates: list of numpy arrays, each of shape (N, 21, 3). For
            method=KINEMATIC_METHOD_SMOOTHED these must come from
            prepare_kinematic_sequence(); otherwise they are the 60-frame
            pose templates (legacy).
        method: tag stored in the profile so authentication computes the
            live signature the same way. None keeps the legacy format.

    Returns:
        dict containing:
            - baseline_binned_velocity: list of float
            - baseline_jerk: float
            - baseline_peak_phase: float
            - tolerance: float
    """
    signatures = [
        extract_kinematic_signature(t)
        for t in templates
    ]

    # Average binned velocity
    all_binned = np.array([s["binned_velocity"] for s in signatures])
    mean_binned = np.mean(all_binned, axis=0)

    mean_jerk = float(np.mean([s["jerk"] for s in signatures]))
    mean_peak_phase = float(np.mean([s["peak_phase"] for s in signatures]))

    # Compute intra-user variation in rhythm
    deviations = []
    for s in signatures:
        diff = np.linalg.norm(np.array(s["binned_velocity"]) - mean_binned)
        deviations.append(diff)

    max_dev = max(deviations) if deviations else 0.1
    calibrated_tolerance = max(DEFAULT_KINEMATIC_TOLERANCE, max_dev * 1.6)
    calibrated_tolerance = min(0.55, calibrated_tolerance)

    dwells = [s.get("max_mid_dwell", 0) for s in signatures]
    max_dwell_baseline = max(dwells) if dwells else 2
    max_allowed_dwell = max(5, max_dwell_baseline + 3)

    profile = {
        "baseline_binned_velocity": [round(float(v), 4) for v in mean_binned],
        "baseline_jerk": float(f"{mean_jerk:.6g}"),
        "baseline_peak_phase": round(mean_peak_phase, 4),
        "max_allowed_dwell": max_allowed_dwell,
        "tolerance": round(float(calibrated_tolerance), 4),
    }
    if method is not None:
        profile["method"] = method
    return profile


def compare_kinematic_signatures(live_sig, baseline_profile, tolerance=DEFAULT_KINEMATIC_TOLERANCE):
    """
    Compare a live gesture's kinematic signature against the registered baseline.

    Args:
        live_sig: dict (from extract_kinematic_signature).
        baseline_profile: dict (from calibrate_kinematic_profile).
        tolerance: float, maximum allowable rhythm dissimilarity.

    Returns:
        tuple of (bool, float, dict):
            - bool: True if movement rhythm matches, False if irregular/imitated.
            - float: normalized rhythm confidence (0.0 to 1.0).
            - dict: diagnostic metrics.
    """
    live_vel = np.array(live_sig["binned_velocity"], dtype=np.float64)
    base_vel = np.array(baseline_profile["baseline_binned_velocity"], dtype=np.float64)

    # Euclidean distance between velocity rhythm distributions
    rhythm_distance = float(np.linalg.norm(live_vel - base_vel))

    # Jerk ratio (how much more jittery the live attempt is compared to baseline)
    base_jerk = baseline_profile.get("baseline_jerk", 0.001)
    if base_jerk < 1e-6:
        base_jerk = 1e-6
    live_jerk = live_sig.get("jerk", 0.001)
    jerk_ratio = live_jerk / base_jerk

    # Peak velocity timing phase
    live_peak = live_sig.get("peak_phase", 0.5)
    base_peak = baseline_profile.get("baseline_peak_phase", 0.5)
    phase_shift = abs(live_peak - base_peak)

    # Mid-gesture dwell / plateau verification
    live_dwell = live_sig.get("max_mid_dwell", 0)
    allowed_dwell = baseline_profile.get("max_allowed_dwell", 6)
    dwell_ok = live_dwell <= allowed_dwell

    # Motor fluidity verification:
    # 1. Velocity envelope distribution matches (within tolerance)
    # 2. Movement jerk does not exceed hesitation threshold (<= 1.75x baseline)
    # 3. Peak acceleration phase does not shift wildly (<= 0.40)
    # 4. No unnatural mid-gesture velocity plateaus / hesitation freezes
    rhythm_ok = rhythm_distance <= tolerance
    jerk_ok = jerk_ratio <= 1.75
    phase_ok = phase_shift <= 0.40

    matches = rhythm_ok and jerk_ok and phase_ok and dwell_ok

    # Combined kinematic confidence score (rhythm shape + motor jerk fluidity + continuous dwell)
    conf_rhythm = max(0.0, 1.0 - (rhythm_distance / (tolerance * 1.25)))
    conf_jerk = max(0.0, 1.0 - max(0.0, jerk_ratio - 1.0) / 0.75) if jerk_ratio > 1.0 else 1.0
    conf_dwell = 1.0 if dwell_ok else max(0.0, 1.0 - (live_dwell - allowed_dwell) / 4.0)
    confidence = max(0.0, min(1.0, 0.45 * conf_rhythm + 0.30 * conf_jerk + 0.25 * conf_dwell))

    details = {
        "rhythm_distance": round(rhythm_distance, 4),
        "tolerance": round(tolerance, 4),
        "jerk_ratio": round(jerk_ratio, 2),
        "phase_shift": round(phase_shift, 4),
        "max_mid_dwell": live_dwell,
        "allowed_dwell": allowed_dwell,
        "rhythm_ok": rhythm_ok,
        "jerk_ok": jerk_ok,
        "phase_ok": phase_ok,
        "dwell_ok": dwell_ok,
        "confidence": round(confidence, 4),
    }

    return matches, confidence, details
