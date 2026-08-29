"""
Gesture Capture — Biometric Authentication System

Registers a user's gesture by recording it multiple times (3 samples
by default) to capture natural variation. Automatically computes a
calibrated DTW threshold from the samples for reliable authentication.

Usage:
    python gesture_capture.py

Flow:
    1. Enter a username
    2. Record your gesture 3 times (the system guides you through each)
    3. System auto-computes your personal threshold
    4. Registration saved — ready for authentication

Controls:
    R  — Start recording the current sample
    Q  — Quit the application

Requirements:
    Python 3.9-3.12, mediapipe, opencv-python, numpy, scipy, dtaidistance
"""

import os
import sys
import time

import cv2
import numpy as np
import mediapipe as mp

from utils.landmarks import extract_landmarks, is_hand_detected
from utils.normalize import normalize_gesture
from gesture_compare import (
    compute_dtw_distance,
    compute_finger_state_threshold_details,
    compute_transition_threshold_details,
    compute_segment_threshold_details,
    compute_threshold_from_samples,
    compute_threshold_details,
    save_registration,
    list_registered_users,
    NUM_REGISTRATION_SAMPLES,
    OUTLIER_REJECTION_FACTOR,
)
from cohort_library import (
    generate_cohort_library,
    validate_gesture_uniqueness,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

RECORDING_DURATION_SEC = 3    # Duration of each gesture recording
TARGET_FRAMES = 60            # Frames after temporal normalization

# Resolve paths relative to this script's location
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(_SCRIPT_DIR, "templates")

WINDOW_NAME = "Gesture Auth — Registration"

# MediaPipe Hands configuration
DETECTION_CONFIDENCE = 0.7
TRACKING_CONFIDENCE = 0.5

# Camera settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


# ============================================================================
# UI DRAWING FUNCTIONS
# ============================================================================

def draw_status_bar(frame, text, color=(0, 200, 0)):
    """Draw a semi-transparent status bar at the top of the frame."""
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(
        frame, text, (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
    )


def draw_progress_bar(frame, progress, remaining_sec):
    """Draw a recording progress bar at the bottom of the frame."""
    h, w = frame.shape[:2]
    bar_height = 35
    bar_y = h - bar_height - 15
    bar_x = 20
    bar_width = w - 40

    cv2.rectangle(
        frame, (bar_x, bar_y),
        (bar_x + bar_width, bar_y + bar_height),
        (40, 40, 40), -1
    )

    fill_width = int(bar_width * progress)
    if fill_width > 0:
        cv2.rectangle(
            frame, (bar_x, bar_y),
            (bar_x + fill_width, bar_y + bar_height),
            (0, 0, 220), -1
        )

    cv2.rectangle(
        frame, (bar_x, bar_y),
        (bar_x + bar_width, bar_y + bar_height),
        (255, 255, 255), 2
    )

    text = f"Recording... {remaining_sec:.1f}s remaining"
    cv2.putText(
        frame, text, (bar_x + 10, bar_y + bar_height - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA
    )


def draw_recording_dot(frame):
    """Draw a blinking red recording dot in the top-right corner."""
    if int(time.time() * 2) % 2 == 0:
        h, w = frame.shape[:2]
        cv2.circle(frame, (w - 30, 25), 10, (0, 0, 255), -1)


def draw_instructions(frame):
    """Draw keyboard controls at the bottom of the frame."""
    h, w = frame.shape[:2]
    text = "[R] Record Gesture  |  [Q] Quit"
    cv2.putText(
        frame, text, (15, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA
    )


def draw_hand_not_detected_warning(frame):
    """Draw a warning when hand is lost during recording."""
    h, w = frame.shape[:2]
    text = "! Hand lost — keep your hand visible !"
    text_size = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )[0]
    text_x = (w - text_size[0]) // 2
    cv2.putText(
        frame, text, (text_x, 80),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2, cv2.LINE_AA
    )


def draw_frame_counter(frame, count):
    """Draw the captured frame count during recording."""
    h, w = frame.shape[:2]
    cv2.putText(
        frame, f"Frames captured: {count}", (w - 220, 80),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA
    )


def draw_sample_info(frame, current_sample, total_samples, username):
    """Draw which sample we're on and the username being registered."""
    h, w = frame.shape[:2]
    text = f"User: {username}  |  Sample {current_sample}/{total_samples}"
    cv2.putText(
        frame, text, (15, 80),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 100), 1, cv2.LINE_AA
    )


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def setup_mediapipe():
    """Initialize and return MediaPipe Hands components."""
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=DETECTION_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE,
    )

    return hands, mp_hands, mp_drawing, mp_drawing_styles


def setup_camera():
    """Open the webcam and configure resolution."""
    print(f"  Opening webcam (index {CAMERA_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print()
        print("  ERROR: Could not open webcam!")
        print("  Check that your webcam is connected and not in use.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    ret, test_frame = cap.read()
    if not ret:
        print("  ERROR: Webcam opened but failed to read a frame.")
        cap.release()
        sys.exit(1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Webcam ready — resolution: {actual_w}x{actual_h}")

    return cap


def record_one_sample(cap, hands, mp_hands, mp_drawing, mp_drawing_styles,
                      sample_num, total_samples, username):
    """
    Record a single gesture sample from the webcam.

    Shows the live webcam feed with hand tracking. Waits for the user
    to press [R] to start recording, then captures for RECORDING_DURATION_SEC.

    Args:
        cap: cv2.VideoCapture object.
        hands: MediaPipe Hands detector.
        mp_hands, mp_drawing, mp_drawing_styles: MediaPipe drawing helpers.
        sample_num: int, which sample number this is (1-based).
        total_samples: int, total samples to record.
        username: str, the username being registered.

    Returns:
        numpy array of shape (TARGET_FRAMES, 21, 3) — the normalized sample.
        Returns None if the user quit or the recording failed.
    """
    is_recording = False
    recording_start_time = 0.0
    recorded_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("  ERROR: Failed to read frame from webcam.")
            return None

        frame = cv2.flip(frame, 1)

        # Process with MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        hand_visible = is_hand_detected(results)

        # Draw landmarks
        if hand_visible:
            for hand_lms in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

        # Show sample info
        draw_sample_info(frame, sample_num, total_samples, username)

        # ─── Recording Mode ──────────────────────────────────────────
        if is_recording:
            elapsed = time.time() - recording_start_time
            remaining = max(0.0, RECORDING_DURATION_SEC - elapsed)
            progress = min(1.0, elapsed / RECORDING_DURATION_SEC)

            if hand_visible:
                landmarks = extract_landmarks(
                    results.multi_hand_landmarks[0]
                )
                recorded_frames.append(landmarks)

            draw_status_bar(
                frame,
                f"RECORDING sample {sample_num}/{total_samples} "
                f"— Perform your gesture!",
                (0, 0, 255)
            )
            draw_progress_bar(frame, progress, remaining)
            draw_recording_dot(frame)
            draw_frame_counter(frame, len(recorded_frames))

            if not hand_visible:
                draw_hand_not_detected_warning(frame)

            # Recording complete
            if elapsed >= RECORDING_DURATION_SEC:
                is_recording = False

                if len(recorded_frames) < 10:
                    print(f"  [FAIL] Sample {sample_num}: too few frames "
                          f"({len(recorded_frames)}). Retrying...")
                    recorded_frames = []
                    continue
                else:
                    # Normalize and return
                    raw = np.array(recorded_frames)
                    normalized = normalize_gesture(raw, TARGET_FRAMES)
                    print(f"  [OK] Sample {sample_num}/{total_samples}: "
                          f"captured {len(recorded_frames)} frames")
                    return normalized

        else:
            # ─── Idle Mode — waiting for [R] ─────────────────────────
            if hand_visible:
                draw_status_bar(
                    frame,
                    f"Sample {sample_num}/{total_samples} — "
                    f"Press [R] to record",
                    (0, 230, 0)
                )
            else:
                draw_status_bar(
                    frame,
                    "Show your hand to the camera...",
                    (100, 150, 255)
                )
            draw_instructions(frame)

        # Display
        cv2.imshow(WINDOW_NAME, frame)

        # Handle keys
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q'):
            return None

        elif (key == ord('r') or key == ord('R')) and not is_recording:
            if hand_visible:
                is_recording = True
                recording_start_time = time.time()
                recorded_frames = []
                print(f"  [REC] Sample {sample_num}: recording for "
                      f"{RECORDING_DURATION_SEC} seconds...")
            else:
                print("  [!] Cannot record — no hand detected.")


# ============================================================================
# QUALITY GATE — OUTLIER DETECTION
# ============================================================================

def validate_sample_quality(new_sample, accepted_samples):
    """
    Check whether a newly recorded sample is consistent with the
    previously accepted samples, rejecting outliers caused by
    accidental movement, camera drops, or user hesitation.

    The quality gate only activates after 2+ samples have been accepted,
    since we need enough data to establish a reference cluster.

    Args:
        new_sample: numpy array of shape (60, 21, 3).
        accepted_samples: list of already-accepted numpy arrays.

    Returns:
        tuple of (bool, str):
            - bool: True if the sample is acceptable, False if outlier.
            - str: Diagnostic message explaining the decision.
    """
    # First two samples are always accepted
    if len(accepted_samples) < 2:
        return True, "Accepted (insufficient reference data for quality check)."

    # Compute distances from the new sample to each accepted sample
    distances_to_cluster = []
    for accepted in accepted_samples:
        d = compute_dtw_distance(new_sample, accepted)
        distances_to_cluster.append(d)

    avg_distance = float(np.mean(distances_to_cluster))

    # Compute the median inter-sample distance among accepted samples
    inter_distances = []
    for i in range(len(accepted_samples)):
        for j in range(i + 1, len(accepted_samples)):
            d = compute_dtw_distance(accepted_samples[i], accepted_samples[j])
            inter_distances.append(d)

    median_inter = float(np.median(inter_distances))

    # Reject if the new sample's average distance exceeds the threshold
    reject_limit = OUTLIER_REJECTION_FACTOR * median_inter

    if median_inter < 1e-6:
        # If all accepted samples are nearly identical, use a generous limit
        return True, "Accepted (reference samples are very consistent)."

    if avg_distance > reject_limit:
        return False, (
            f"Rejected: avg distance {avg_distance:.3f} exceeds "
            f"{OUTLIER_REJECTION_FACTOR}x median ({median_inter:.3f}) "
            f"= {reject_limit:.3f}. Irregular motion detected."
        )

    return True, (
        f"Accepted: avg distance {avg_distance:.3f} within "
        f"limit {reject_limit:.3f}."
    )


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main gesture registration application."""

    # ─── Banner ───────────────────────────────────────────────────────
    print()
    print("=" * 58)
    print("   GESTURE REGISTRATION — Biometric Auth System")
    print("=" * 58)
    print()

    # ─── Get username ─────────────────────────────────────────────────
    existing_users = list_registered_users()
    if existing_users:
        print(f"  Existing users: {', '.join(existing_users)}")
    print()
    username = input("  Enter a username to register: ").strip()

    if not username:
        print("  No username entered. Exiting.")
        return

    username = username.lower().replace(" ", "_")

    if username in existing_users:
        overwrite = input(
            f"  User '{username}' already exists. "
            f"Re-register? (y/n): "
        ).strip().lower()
        if overwrite != 'y':
            print("  Cancelled.")
            return

    print()
    print(f"  Registering: {username}")
    print(f"  You will record your gesture {NUM_REGISTRATION_SAMPLES} times.")
    print(f"  Perform the SAME gesture each time for best accuracy.")
    print()

    # ─── Setup ────────────────────────────────────────────────────────
    hands, mp_hands, mp_drawing, mp_drawing_styles = setup_mediapipe()
    cap = setup_camera()
    print()

    # ─── Record multiple samples (with quality gate) ─────────────────
    samples = []
    max_retries_per_sample = 3  # prevent infinite re-record loops

    try:
        sample_num = 1
        while sample_num <= NUM_REGISTRATION_SAMPLES:
            retries = 0
            accepted = False

            while not accepted:
                print(f"  --- Sample {sample_num}/{NUM_REGISTRATION_SAMPLES} "
                      f"--- Press [R] when ready ---")

                sample = record_one_sample(
                    cap, hands, mp_hands, mp_drawing, mp_drawing_styles,
                    sample_num, NUM_REGISTRATION_SAMPLES, username
                )

                if sample is None:
                    print("  Registration cancelled.")
                    return

                # ── Quality Gate: check for outliers ──────────
                is_ok, quality_msg = validate_sample_quality(
                    sample, samples
                )

                if is_ok:
                    samples.append(sample)
                    accepted = True
                    print(f"        ✓ Sample {sample_num} {quality_msg}")
                else:
                    retries += 1
                    print(f"        ✗ Sample {sample_num} had irregular "
                          f"motion. Let's re-record that one.")
                    print(f"          ({quality_msg})")
                    if retries >= max_retries_per_sample:
                        # Accept after max retries to avoid blocking
                        samples.append(sample)
                        accepted = True
                        print(f"        ⚠ Accepted after {retries} "
                              f"retries (max retries reached).")

            # Brief pause message between samples (not on the last one)
            if sample_num < NUM_REGISTRATION_SAMPLES:
                print(f"        Ready for sample "
                      f"{sample_num + 1}/{NUM_REGISTRATION_SAMPLES}.")
                print()

            sample_num += 1

    except KeyboardInterrupt:
        print("\n  Interrupted.")
        return

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

    # ─── Compute threshold ────────────────────────────────────────────
    print()
    print("  Computing optimal threshold from your samples...")

    threshold, pairwise_distances = compute_threshold_from_samples(samples)
    threshold_details = compute_threshold_details(pairwise_distances)
    finger_state_details = compute_finger_state_threshold_details(samples)
    transition_details = compute_transition_threshold_details(samples)
    segment_details = compute_segment_threshold_details(samples)

    print()
    print("  Pairwise distances between your recordings:")
    pair_idx = 0
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            print(f"    Sample {i+1} vs Sample {j+1}: "
                  f"{pairwise_distances[pair_idx]:.4f}")
            pair_idx += 1

    print(f"  Max variation:     {max(pairwise_distances):.4f}")
    print(f"  Consistency score: {threshold_details['consistency_score']:.1f}/100")
    print(f"  Threshold method:  {threshold_details['method']}")
    print(f"  Computed threshold:      {threshold:.4f}")
    print(f"  Finger mismatch limit:   "
          f"{finger_state_details['threshold']:.4f}")
    print(f"  Transition order limit:  "
          f"{transition_details['threshold']:.4f}")
    print(f"  Segment mismatch limit:  "
          f"{segment_details['threshold']:.4f}")

    # ─── Cohort-based uniqueness check ────────────────────────────────
    print()
    print("  Checking gesture uniqueness against common patterns...")
    cohort = generate_cohort_library()
    uniqueness = validate_gesture_uniqueness(
        samples, cohort,
        intra_user_max=max(pairwise_distances),
        compute_dtw_fn=compute_dtw_distance,
    )

    if uniqueness["is_unique"]:
        print(f"  ✓ Gesture is unique! (impostor ratio: "
              f"{uniqueness['ratio']:.2f}x)")
    else:
        print()
        print(f"  ⚠ WARNING: {uniqueness['warning']}")
        print(f"    Impostor ratio: {uniqueness['ratio']:.2f}x "
              f"(minimum: 1.50x)")
        print(f"    Closest cohort distance: "
              f"{uniqueness['min_impostor_distance']:.4f}")
        print()

    # ─── Save registration ────────────────────────────────────────────
    user_dir = save_registration(
        username, samples, threshold, pairwise_distances
    )

    print()
    print("  " + "=" * 50)
    print(f"  REGISTRATION COMPLETE for '{username}'!")
    print("  " + "=" * 50)
    print(f"  Samples saved:  {len(samples)}")
    print(f"  Threshold:      {threshold:.4f}")
    print(f"  Location:       {user_dir}")
    print()
    print("  You can now authenticate with:")
    print("    python gesture_auth.py")
    print()


if __name__ == "__main__":
    main()
