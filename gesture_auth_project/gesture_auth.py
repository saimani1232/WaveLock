"""
Gesture Authentication Script — Live Biometric Login

Opens the webcam, captures a live gesture from the user, and compares
it against their stored templates using Dynamic Time Warping (DTW).

Uses multi-sample comparison: the live gesture is compared against ALL
stored registration samples and the best match (minimum distance) is
used for the authentication decision.

Usage:
    python gesture_auth.py

Requirements:
    - Register first with gesture_capture.py
    - Python 3.9-3.12, mediapipe, opencv-python, numpy, scipy, dtaidistance
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
    load_all_user_templates,
    load_user_threshold,
    load_user_finger_state_threshold,
    load_user_transition_threshold,
    load_user_segment_threshold,
    authenticate_with_details,
    update_templates_if_high_confidence,
    list_registered_users,
)
from gesture_capture import (
    setup_mediapipe,
    setup_camera,
    draw_status_bar,
    draw_progress_bar,
    draw_recording_dot,
    draw_hand_not_detected_warning,
    draw_frame_counter,
    RECORDING_DURATION_SEC,
    TARGET_FRAMES,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

WINDOW_NAME = "Gesture Auth — Login"


# ============================================================================
# AUTH-SPECIFIC UI FUNCTIONS
# ============================================================================

def draw_auth_instructions(frame):
    """Draw auth-specific instructions at the bottom of the frame."""
    h, w = frame.shape[:2]
    text = "[R] Authenticate  |  [T] Try Again  |  [Q] Quit"
    cv2.putText(
        frame, text, (15, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.43, (180, 180, 180), 1, cv2.LINE_AA
    )


def draw_result_overlay(frame, granted, distance, threshold,
                        finger_mismatch=None, finger_threshold=None,
                        transition_dissim=None, transition_threshold=None,
                        segment_max=None, segment_threshold_val=None,
                        fused_score=None):
    """
    Draw the authentication result as a large overlay on the frame.
    ACCESS GRANTED = green overlay, ACCESS DENIED = red overlay.
    Now includes the Fused Confidence Score.
    """
    h, w = frame.shape[:2]

    # Semi-transparent color overlay
    overlay = frame.copy()
    if granted:
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 80, 0), -1)
    else:
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 80), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    # Main result text
    main_text = "ACCESS GRANTED" if granted else "ACCESS DENIED"
    main_color = (0, 255, 0) if granted else (0, 0, 255)

    text_size = cv2.getTextSize(
        main_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3
    )[0]
    text_x = (w - text_size[0]) // 2
    text_y = (h // 2) - 70

    # Shadow + text
    cv2.putText(
        frame, main_text, (text_x + 2, text_y + 2),
        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4, cv2.LINE_AA
    )
    cv2.putText(
        frame, main_text, (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX, 1.5, main_color, 3, cv2.LINE_AA
    )

    # Confidence score line
    if fused_score is not None:
        conf_pct = fused_score * 100
        if conf_pct >= 75:
            conf_color = (0, 255, 0)       # green
        elif conf_pct >= 55:
            conf_color = (0, 200, 255)     # yellow-orange
        else:
            conf_color = (0, 0, 255)       # red
        conf_text = f"Confidence: {conf_pct:.0f}%"
        conf_size = cv2.getTextSize(
            conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )[0]
        conf_x = (w - conf_size[0]) // 2
        cv2.putText(
            frame, conf_text, (conf_x, text_y + 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, conf_color, 2, cv2.LINE_AA
        )

    # Line 1: DTW distance and finger-state info
    info_text = f"DTW: {distance:.2f}/{threshold:.2f}"
    if finger_mismatch is not None and finger_threshold is not None:
        info_text += (f"  |  Fingers: {finger_mismatch:.0%}/"
                      f"{finger_threshold:.0%}")
    info_size = cv2.getTextSize(
        info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )[0]
    info_x = (w - info_size[0]) // 2
    cv2.putText(
        frame, info_text, (info_x, text_y + 70),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA
    )

    # Line 2: Transition order and segment info
    if transition_dissim is not None and transition_threshold is not None:
        line2 = (f"Order: {transition_dissim:.0%}/{transition_threshold:.0%}")
        if segment_max is not None and segment_threshold_val is not None:
            line2 += (f"  |  Segment: {segment_max:.0%}/"
                      f"{segment_threshold_val:.0%}")
        line2_size = cv2.getTextSize(
            line2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )[0]
        line2_x = (w - line2_size[0]) // 2
        cv2.putText(
            frame, line2, (line2_x, text_y + 95),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA
        )

    # Retry instruction
    retry_text = "Press [T] to try again  |  Press [Q] to quit"
    retry_size = cv2.getTextSize(
        retry_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )[0]
    retry_x = (w - retry_size[0]) // 2
    cv2.putText(
        frame, retry_text, (retry_x, text_y + 135),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA
    )


def draw_user_banner(frame, username, threshold):
    """Show which user is being authenticated and their threshold."""
    text = f"User: {username}  |  Threshold: {threshold:.2f}"
    cv2.putText(
        frame, text, (15, 80),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 100), 1, cv2.LINE_AA
    )


# ============================================================================
# MAIN AUTHENTICATION FLOW
# ============================================================================

def select_user():
    """Prompt user to select their username."""
    users = list_registered_users()

    if len(users) == 0:
        print("  No registered users found!")
        print("  Run gesture_capture.py first to register.")
        return None

    print(f"  Registered users: {', '.join(users)}")
    print()

    while True:
        username = input("  Enter your username: ").strip().lower()

        if not username:
            print("  Cancelled.")
            return None

        username = username.replace(" ", "_")

        if username in users:
            return username
        else:
            print(f"  User '{username}' not found. "
                  f"Available: {', '.join(users)}")
            print()


def main():
    """Main authentication application."""

    # ─── Banner ───────────────────────────────────────────────────────
    print()
    print("=" * 58)
    print("   GESTURE AUTH — Biometric Authentication System")
    print("=" * 58)
    print()

    # ─── Select User ─────────────────────────────────────────────────
    username = select_user()
    if username is None:
        return

    # Load stored templates and threshold
    try:
        stored_templates = load_all_user_templates(username)
        threshold = load_user_threshold(username)
        finger_state_threshold = load_user_finger_state_threshold(username)
        transition_threshold = load_user_transition_threshold(username)
        segment_threshold = load_user_segment_threshold(username)
        print(f"  Loaded {len(stored_templates)} template(s) for '{username}'")
        print(f"  Authentication threshold: {threshold:.4f}")
        print(f"  Finger mismatch limit:    {finger_state_threshold:.4f}")
        print(f"  Transition order limit:   {transition_threshold:.4f}")
        print(f"  Segment mismatch limit:   {segment_threshold:.4f}")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return

    print()
    print("  Controls:")
    print(f"    [R]  Record gesture to authenticate "
          f"({RECORDING_DURATION_SEC} seconds)")
    print("    [Q]  Quit")
    print()

    # ─── Initialize ───────────────────────────────────────────────────
    hands, mp_hands, mp_drawing, mp_drawing_styles = setup_mediapipe()
    cap = setup_camera()

    print()
    print(f"  Authenticating as: {username}")
    print("  Show your hand and press [R] to begin.")
    print("  ─" * 20)
    print()

    # ─── State ────────────────────────────────────────────────────────
    is_recording = False
    recording_start_time = 0.0
    recorded_frames = []
    show_result = False
    result_granted = False
    result_distance = 0.0
    result_finger_mismatch = 0.0
    result_transition_dissim = 0.0
    result_segment_max = 0.0
    result_fused_score = 0.0
    result_frame = None

    # ─── Main Loop ────────────────────────────────────────────────────
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("  ERROR: Failed to read frame.")
                break

            frame = cv2.flip(frame, 1)

            # ─── Result Display Mode ──────────────────────────────────
            if show_result:
                display = result_frame.copy()
                draw_result_overlay(
                    display, result_granted,
                    result_distance, threshold,
                    result_finger_mismatch, finger_state_threshold,
                    result_transition_dissim, transition_threshold,
                    result_segment_max, segment_threshold,
                    result_fused_score
                )
                cv2.imshow(WINDOW_NAME, display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('t') or key == ord('T'):
                    show_result = False
                    print("  Trying again... Press [R] when ready.")
                continue

            # ─── Process Frame ────────────────────────────────────────
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = hands.process(rgb_frame)
            rgb_frame.flags.writeable = True

            hand_visible = is_hand_detected(results)

            if hand_visible:
                for hand_lms in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

            draw_user_banner(frame, username, threshold)

            # ─── Recording Logic ──────────────────────────────────────
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
                    "AUTHENTICATING — Perform your gesture!",
                    (0, 150, 255)
                )
                draw_progress_bar(frame, progress, remaining)
                draw_recording_dot(frame)
                draw_frame_counter(frame, len(recorded_frames))

                if not hand_visible:
                    draw_hand_not_detected_warning(frame)

                if elapsed >= RECORDING_DURATION_SEC:
                    is_recording = False
                    result_frame = frame.copy()

                    if len(recorded_frames) < 10:
                        print(f"  [FAIL] Too few frames "
                              f"({len(recorded_frames)}). Retry.")
                        recorded_frames = []
                    else:
                        raw = np.array(recorded_frames)
                        normalized = normalize_gesture(raw, TARGET_FRAMES)

                        print(f"  Comparing against "
                              f"{len(stored_templates)} template(s)...")

                        (result_granted, result_distance, best_idx,
                         auth_details) = authenticate_with_details(
                            normalized,
                            stored_templates,
                            threshold,
                            finger_state_threshold,
                            transition_threshold,
                            segment_threshold,
                        )
                        result_finger_mismatch = \
                            auth_details["finger_state_mismatch"]
                        result_transition_dissim = \
                            auth_details["transition_dissimilarity"]
                        result_segment_max = \
                            auth_details["segment_max_mismatch"]
                        result_fused_score = \
                            auth_details["fused_score"]

                        # Terminal output
                        if result_granted:
                            print()
                            print("  =============================")
                            print("     ACCESS GRANTED")
                            print("  =============================")
                        else:
                            print()
                            print("  =============================")
                            print("     ACCESS DENIED")
                            print("  =============================")

                        conf_pct = result_fused_score * 100
                        print(f"  Confidence:      "
                              f"{conf_pct:.1f}%")
                        print(f"  DTW Distance:    {result_distance:.4f}"
                              f"  (best match: sample {best_idx + 1})")
                        print(f"  Threshold:       {threshold:.4f}")
                        print(f"  Finger mismatch: "
                              f"{result_finger_mismatch:.4f}")
                        print(f"  Finger limit:    "
                              f"{finger_state_threshold:.4f}")
                        print(f"  Transition diff: "
                              f"{result_transition_dissim:.4f}")
                        print(f"  Transition limit:"
                              f" {transition_threshold:.4f}")
                        print(f"  Segment max:     "
                              f"{result_segment_max:.4f}")
                        print(f"  Segment limit:   "
                              f"{segment_threshold:.4f}")
                        if not result_granted:
                            print(f"  Reject reason:   "
                                  f"{auth_details['failure_reason']}")
                        print(f"  Frames captured: {len(recorded_frames)}")

                        # ── Adaptive Template Aging ────────────
                        if result_granted:
                            aging_result = \
                                update_templates_if_high_confidence(
                                    username, normalized,
                                    result_distance, threshold,
                                    stored_templates,
                                )
                            if aging_result:
                                print(
                                    f"  [ADAPT] Template "
                                    f"{aging_result['replaced_template']}"
                                    f" updated (high-confidence"
                                    f" match)."
                                )
                                # Reload templates and thresholds
                                stored_templates = \
                                    load_all_user_templates(username)
                                threshold = load_user_threshold(
                                    username
                                )
                                finger_state_threshold = \
                                    load_user_finger_state_threshold(
                                        username
                                    )
                                transition_threshold = \
                                    load_user_transition_threshold(
                                        username
                                    )
                                segment_threshold = \
                                    load_user_segment_threshold(
                                        username
                                    )

                        print()
                        print("  Press [T] to try again, [Q] to quit.")

                        show_result = True
                        recorded_frames = []

            else:
                # ─── Idle State ───────────────────────────────────────
                if hand_visible:
                    draw_status_bar(
                        frame,
                        "Ready — Press [R] to Authenticate",
                        (0, 230, 0)
                    )
                else:
                    draw_status_bar(
                        frame,
                        "Show your hand to the camera...",
                        (100, 150, 255)
                    )
                draw_auth_instructions(frame)

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == ord('Q'):
                break
            elif (key == ord('r') or key == ord('R')) and not is_recording:
                if hand_visible:
                    is_recording = True
                    recording_start_time = time.time()
                    recorded_frames = []
                    print(f"  [REC] Authenticating — perform your gesture "
                          f"for {RECORDING_DURATION_SEC} seconds...")
                else:
                    print("  [!] Cannot start — no hand detected.")

    except KeyboardInterrupt:
        print("\n  Interrupted.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print("  Resources released. Goodbye!")
        print()


if __name__ == "__main__":
    main()
