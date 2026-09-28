"""
WaveLock — Multimodal Biometric Authentication Terminal
Runs real-time authentication by coupling:
  1. Face Identity Anchor (YuNet + SFace Cosine Similarity & Anti-Sibling Defense)
  2. Macro Gesture Trajectory (DTW, Finger States, Transition Order, Segment Mismatch)
  3. Hand Morphology Invariants (5 Bone Length Ratios) & Neuromotor Kinematics (Jerk & Dwell)
  4. Multimodal Decision Fusion Engine

Usage:
    python gesture_auth.py
"""

import os
import sys
import time
import json
import cv2
import numpy as np

from utils.landmarks import extract_landmarks, is_hand_detected
from utils.normalize import normalize_gesture
from gesture_compare import (
    load_all_user_templates,
    load_user_threshold,
    load_user_finger_state_threshold,
    load_user_transition_threshold,
    load_user_segment_threshold,
    load_user_anthropometric_profile,
    load_user_kinematic_profile,
    load_user_face_profile,
    authenticate_with_details,
    update_templates_if_high_confidence,
    list_registered_users,
)
from utils.face_auth import (
    verify_face,
    SFACE_COSINE_THRESHOLD,
    SFACE_LOOKALIKE_FLOOR,
)
from gesture_capture import (
    setup_mediapipe,
    setup_camera,
    RECORDING_DURATION_SEC,
    TARGET_FRAMES,
)
from utils.camera_utils import print_camera_diagnostics


# ============================================================================
# CONFIGURATION & UI CONSTANTS
# ============================================================================

WINDOW_NAME = "WaveLock - Multimodal Biometric Terminal"

CANVAS_WIDTH = 980
CANVAS_HEIGHT = 560
CAM_X = 15
CAM_Y = 40
CAM_W = 640
CAM_H = 480

# Theme Palette (BGR)
COLOR_BG = (22, 17, 13)            # Dark Navy Background (#0d1116)
COLOR_CARD = (35, 27, 22)          # Surface Card (#161b23)
COLOR_BORDER = (60, 50, 45)        # Border (#2d323c)
COLOR_TEXT_WHITE = (245, 245, 245)
COLOR_TEXT_MUTED = (160, 150, 140)
COLOR_ACCENT = (255, 175, 60)      # Neon Blue/Cyan
COLOR_GREEN = (80, 210, 80)        # Verified Green
COLOR_AMBER = (0, 165, 255)        # Sibling Lookalike Amber
COLOR_RED = (70, 70, 245)          # Impostor Red


# ============================================================================
# DASHBOARD DRAWING ENGINE
# ============================================================================

def draw_card(canvas, x, y, w, h, title=""):
    """Draw a dark rounded rectangle card with a subtle border."""
    cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_CARD, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_BORDER, 1)
    if title:
        cv2.putText(canvas, title, (x + 10, y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_ACCENT, 1, cv2.LINE_AA)
        cv2.line(canvas, (x + 10, y + 22), (x + w - 10, y + 22), COLOR_BORDER, 1)


def draw_meter_bar(canvas, x, y, w, h, value, threshold=None, color=COLOR_GREEN):
    """Draw a horizontal fill meter with an optional vertical threshold marker."""
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (45, 35, 30), -1)
    fill_w = int(max(0.0, min(1.0, value)) * w)
    if fill_w > 0:
        cv2.rectangle(canvas, (x, y), (x + fill_w, y + h), color, -1)
    if threshold is not None:
        tx = x + int(max(0.0, min(1.0, threshold)) * w)
        cv2.line(canvas, (tx, y - 2), (tx, y + h + 2), (0, 255, 255), 2)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_BORDER, 1)


def build_dashboard(cam_frame, state="IDLE", username="User", threshold=2.13,
                    num_templates=5, face_info=None, gesture_info=None,
                    hand_info=None, result_info=None, recording_info=None):
    """
    Compose the 980x560 Enterprise Split-View Biometric Dashboard.
    """
    canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), COLOR_BG, dtype=np.uint8)

    # ─── 1. Top Global Navigation Header ──────────────────────────────────
    cv2.rectangle(canvas, (0, 0), (CANVAS_WIDTH, 32), (30, 22, 17), -1)
    cv2.line(canvas, (0, 32), (CANVAS_WIDTH, 32), COLOR_BORDER, 1)

    cv2.putText(canvas, "WAVELOCK // MULTIMODAL BIOMETRIC AUTHENTICATION",
                (15, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)

    if state == "GRANTED":
        status_tag = "STATE: ACCESS GRANTED"
        tag_color = COLOR_GREEN
    elif state == "DENIED":
        status_tag = "STATE: ACCESS DENIED"
        tag_color = COLOR_RED
    elif state == "RECORDING":
        status_tag = "STATE: RECORDING GESTURE"
        tag_color = COLOR_AMBER
    else:
        status_tag = "STATE: IDLE (READY)"
        tag_color = COLOR_ACCENT

    cv2.putText(canvas, status_tag, (CANVAS_WIDTH - 220, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, tag_color, 1, cv2.LINE_AA)

    # ─── 2. Left Camera Viewport (640x480) ────────────────────────────────
    if cam_frame.shape[0] != CAM_H or cam_frame.shape[1] != CAM_W:
        cam_frame = cv2.resize(cam_frame, (CAM_W, CAM_H))

    canvas[CAM_Y:CAM_Y + CAM_H, CAM_X:CAM_X + CAM_W] = cam_frame
    cv2.rectangle(canvas, (CAM_X, CAM_Y), (CAM_X + CAM_W, CAM_Y + CAM_H), COLOR_BORDER, 1)

    # ─── 3. Action / Status Bar (Underneath Camera) ───────────────────────
    bar_y = CAM_Y + CAM_H + 6
    if state == "RECORDING" and recording_info:
        rem = recording_info.get("remaining", 0.0)
        prog = recording_info.get("progress", 0.0)
        n_frames = recording_info.get("frames", 0)
        cv2.rectangle(canvas, (CAM_X, bar_y), (CAM_X + CAM_W, bar_y + 25), COLOR_CARD, -1)
        fill_px = int(prog * CAM_W)
        if fill_px > 0:
            cv2.rectangle(canvas, (CAM_X, bar_y), (CAM_X + fill_px, bar_y + 25), (0, 110, 220), -1)
        cv2.rectangle(canvas, (CAM_X, bar_y), (CAM_X + CAM_W, bar_y + 25), COLOR_BORDER, 1)
        rec_text = f"[REC] Recording Gesture... {rem:.1f}s remaining ({n_frames} frames captured)"
        cv2.putText(canvas, rec_text, (CAM_X + 15, bar_y + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)
    else:
        cv2.rectangle(canvas, (CAM_X, bar_y), (CAM_X + CAM_W, bar_y + 25), (25, 20, 16), -1)
        cv2.rectangle(canvas, (CAM_X, bar_y), (CAM_X + CAM_W, bar_y + 25), COLOR_BORDER, 1)
        if state in ("GRANTED", "DENIED", "RESULT"):
            instruct = "[T] Try Again   |   [Q] Quit System"
        else:
            instruct = "[R] Start Recording Gesture   |   [Q] Quit System"
        cv2.putText(canvas, instruct, (CAM_X + 15, bar_y + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

    # ─── 4. Right Sidebar Telemetry Dashboard ─────────────────────────────
    sx = 665
    sw = 300
    sy = CAM_Y

    # Card 1: Target Profile (h: 58)
    draw_card(canvas, sx, sy, sw, 58, "TARGET PROFILE")
    cv2.putText(canvas, f"User: {username}", (sx + 15, sy + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Threshold: {threshold:.2f}  |  Enrolled: {num_templates} templates", (sx + 15, sy + 51),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
    sy += 64

    # Card 2: Stage 1 Face Anchor (h: 92)
    draw_card(canvas, sx, sy, sw, 92, "STAGE 1: FACE IDENTITY ANCHOR")
    f_match = face_info.get("match", None) if face_info else None
    f_score = face_info.get("score", 0.0) if face_info else 0.0
    f_fail = face_info.get("fail", None) if face_info else None
    f_enrolled = face_info.get("enrolled", True) if face_info else True

    if not f_enrolled:
        f_status = "DISABLED (Legacy Profile)"
        f_color = COLOR_TEXT_MUTED
    elif f_match is True:
        f_status = f"VERIFIED (Cosine {f_score:.2f})"
        f_color = COLOR_GREEN
    elif f_fail == "lookalike_sibling_detected":
        f_status = f"ALERT: SIBLING LOOKALIKE ({f_score:.2f})"
        f_color = COLOR_AMBER
    elif f_match is False:
        f_status = f"REJECTED: IMPOSTOR ({f_score:.2f})"
        f_color = COLOR_RED
    elif state == "RECORDING":
        f_status = "TRACKING & VERIFYING ENFORCED"
        f_color = COLOR_ACCENT
    else:
        f_status = "MONITORING ACTIVE"
        f_color = COLOR_TEXT_MUTED

    cv2.putText(canvas, f_status, (sx + 15, sy + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, f_color, 1, cv2.LINE_AA)

    # Meter bar for Cosine Similarity: Normalized [-0.1, 0.8] to [0, 1]
    norm_cosine = max(0.0, min(1.0, (f_score + 0.1) / 0.9))
    norm_thresh = (SFACE_COSINE_THRESHOLD + 0.1) / 0.9
    bar_col = f_color if f_enrolled else COLOR_BORDER
    draw_meter_bar(canvas, sx + 15, sy + 48, sw - 30, 10, norm_cosine, norm_thresh, bar_col)

    cv2.putText(canvas, f"Req: {SFACE_COSINE_THRESHOLD:.3f} Cosine [Yellow Line] (Floor: {SFACE_LOOKALIKE_FLOOR:.3f})",
                (sx + 15, sy + 76), cv2.FONT_HERSHEY_SIMPLEX, 0.33, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
    sy += 98

    # Card 3: Stage 2 Gesture Password (h: 104)
    draw_card(canvas, sx, sy, sw, 104, "STAGE 2: GESTURE DYNAMICS")
    if gesture_info:
        g_frames = gesture_info.get("frames", 0)
        cv2.putText(canvas, f"Captured Frames: {g_frames} / 60", (sx + 15, sy + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)

        dtw_val = gesture_info.get("dtw", None)
        dtw_str = f"DTW: {dtw_val:.2f}" if dtw_val is not None else "DTW: Pending"
        order_val = gesture_info.get("order", None)
        order_str = f"Order: {order_val:.0%}" if order_val is not None else "Order: Pending"
        cv2.putText(canvas, f"{dtw_str}  |  {order_str}", (sx + 15, sy + 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

        seg_val = gesture_info.get("segment", None)
        seg_str = f"Segment Max: {seg_val:.0%}" if seg_val is not None else "Segment: Pending"
        cv2.putText(canvas, seg_str, (sx + 15, sy + 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

        gate_status = gesture_info.get("gate_status", "Evaluating...")
        gate_col = COLOR_GREEN if ("PASS" in gate_status and "FAIL" not in gate_status) else (COLOR_RED if "FAIL" in gate_status else COLOR_ACCENT)
        cv2.putText(canvas, f"Macro Gates: {gate_status}", (sx + 15, sy + 94),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, gate_col, 1, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "Status: Ready for performance", (sx + 15, sy + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.putText(canvas, "4 Security Gates Armed & Monitoring", (sx + 15, sy + 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_ACCENT, 1, cv2.LINE_AA)
    sy += 110

    # Card 4: Stage 3 Hand Biometrics (h: 76)
    draw_card(canvas, sx, sy, sw, 76, "STAGE 3: HAND BIOMETRICS")
    if hand_info:
        h_anthro = hand_info.get("anthro_conf", 1.0)
        h_match = hand_info.get("anthro_match", True)
        h_color = COLOR_GREEN if h_match else COLOR_RED
        h_text = f"Anatomy: {h_anthro:.0%} match [PASS]" if h_match else "Anatomy: Impostor Bone Ratios! [FAIL]"
        cv2.putText(canvas, h_text, (sx + 15, sy + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, h_color, 1, cv2.LINE_AA)

        h_jerk = hand_info.get("jerk_ratio", 1.0)
        j_ok = hand_info.get("kin_match", True)
        j_color = COLOR_GREEN if j_ok else COLOR_RED
        j_text = f"Motor Jerk: Fluid ({h_jerk:.1f}x) [PASS]" if j_ok else f"Motor Jerk: Hesitation ({h_jerk:.1f}x) [FAIL]"
        cv2.putText(canvas, j_text, (sx + 15, sy + 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, j_color, 1, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "Bone Ratios & Motor Jerk Ready", (sx + 15, sy + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
    sy += 82

    # Card 5: Stage 4 Multimodal Decision (h: 96)
    draw_card(canvas, sx, sy, sw, 96, "STAGE 4: MULTIMODAL CONSENSUS")
    if result_info:
        granted = result_info.get("granted", False)
        fused = result_info.get("fused_score", 0.0)
        reason = result_info.get("reason", None)
        dec_color = COLOR_GREEN if granted else COLOR_RED
        dec_text = "[ ACCESS GRANTED ]" if granted else "[ ACCESS DENIED ]"

        cv2.rectangle(canvas, (sx + 12, sy + 26), (sx + sw - 12, sy + 56), dec_color, -1)
        d_size = cv2.getTextSize(dec_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
        d_x = sx + (sw - d_size[0]) // 2
        cv2.putText(canvas, dec_text, (d_x, sy + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0) if granted else (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(canvas, f"Fused Multi-Score: {fused:.1%}", (sx + 15, sy + 74),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)
        if not granted and reason:
            cv2.putText(canvas, f"Reason: {reason[:26]}", (sx + 15, sy + 88),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, COLOR_AMBER, 1, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "Conjunctive Consensus Enforced", (sx + 15, sy + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.putText(canvas, "Weights: 40% Face | 35% Gesture | 25% Hand", (sx + 15, sy + 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (140, 140, 140), 1, cv2.LINE_AA)

    return canvas


# ============================================================================
# MAIN AUTHENTICATION FLOW
# ============================================================================

def main():
    print("=" * 60)
    print("   WAVELOCK -- Multimodal Biometric Authentication Terminal")
    print("=" * 60)

    import argparse
    parser = argparse.ArgumentParser(description="WaveLock Multimodal Biometric Terminal")
    parser.add_argument("user", nargs="?", default=None, help="Username to authenticate")
    parser.add_argument("--user", "-u", dest="opt_user", type=str, default=None, help="Username to authenticate")
    parser.add_argument("--camera", "--cam", "-c", dest="camera", type=int, default=None, help="Camera index (default: auto-detect)")
    parser.add_argument("--list-cams", action="store_true", help="List available cameras and exit")
    args = parser.parse_args()

    if args.list_cams:
        print_camera_diagnostics()
        return

    # ─── User Selection ───────────────────────────────────────────────
    registered_users = list_registered_users()

    if not registered_users:
        print("\n  [!] No registered users found.")
        print("  Please run gesture_capture.py first to register a gesture.")
        print()
        return

    print(f"\n  Registered users: {', '.join(registered_users)}")

    raw_user = args.opt_user or args.user
    if raw_user:
        username = raw_user.strip()
        if username not in registered_users:
            print(f"  ERROR: User '{username}' not found.")
            return
        print(f"  Using username from arguments: {username}")
    else:
        username = input("  Enter your username: ").strip()
        if not username:
            print("  Username cannot be empty.")
            return
        if username not in registered_users:
            print(f"  ERROR: User '{username}' not registered.")
            return

    # ─── Load User Profile ────────────────────────────────────────────
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

        # Load Biometric Identity Baselines (Anti-Shoulder-Surfing Layer)
        anthro_profile = load_user_anthropometric_profile(
            username, stored_templates
        )
        kinematic_profile = load_user_kinematic_profile(
            username, stored_templates
        )
        print(f"  Hand bone aspect ratio:   "
              f"{anthro_profile['baseline'][0]:.4f} "
              f"(+/-{anthro_profile['tolerance']:.0%})")
        print("  Biometric Identity Layer: ACTIVE (Anti-Shoulder-Surfing)")

        # Load Multimodal Face Profile
        face_profile = load_user_face_profile(username)
        face_enrolled = face_profile is not None
        if face_enrolled:
            print("  Face Identity Layer:      [ACTIVE] Multimodal Gate Enforced")
            print(f"                            Enrolled: templates/{username}/face_embedding.npy")
        else:
            print("  Face Identity Layer:      [DISABLED] Legacy Mode (No face enrolled)")
            print(f"                            To enable: python enroll_face.py --user {username}")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return

    print()
    print("  Controls:")
    print(f"    [R]  Record gesture to authenticate ({RECORDING_DURATION_SEC} seconds)")
    print("    [Q]  Quit")
    print()

    # ─── Initialize ───────────────────────────────────────────────────
    hands, mp_hands, mp_drawing, mp_drawing_styles = setup_mediapipe()
    cap = setup_camera(camera_index=args.camera)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    print()
    print(f"  Authenticating as: {username}")
    print("  Show your hand and press [R] to begin.")
    print("  " + "-" * 30)
    print()

    # ─── State ────────────────────────────────────────────────────────
    is_recording = False
    recording_start_time = 0.0
    recorded_frames = []
    clean_frames_recorded = []
    show_result = False
    result_granted = False
    result_distance = 0.0
    result_finger_mismatch = 0.0
    result_transition_dissim = 0.0
    result_segment_max = 0.0
    result_fused_score = 0.0
    result_anthro_match = True
    result_anthro_conf = 1.0
    result_kin_match = True
    result_kin_conf = 1.0
    result_jerk_ratio = 1.0
    result_face_match = None
    result_face_conf = 1.0
    result_face_score = 0.0
    result_face_fail = None
    result_frame = None
    result_details = None

    live_face_match = None
    live_face_score = 0.0
    live_face_conf = 1.0
    live_face_fail = None

    # ─── Main Loop ────────────────────────────────────────────────────
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("  ERROR: Failed to read frame.")
                break

            frame = cv2.flip(frame, 1)
            clean_frame = frame.copy()
            h, w = frame.shape[:2]

            # ─── Result Display Mode ──────────────────────────────────
            if show_result:
                state_label = "GRANTED" if result_granted else "DENIED"
                display_cam = result_frame.copy()

                # Elegant centered result overlay on camera viewport
                overlay = display_cam.copy()
                tint_color = (0, 70, 0) if result_granted else (0, 0, 70)
                cv2.rectangle(overlay, (0, 0), (w, h), tint_color, -1)
                cv2.addWeighted(overlay, 0.22, display_cam, 0.78, 0, display_cam)

                banner_text = "ACCESS GRANTED" if result_granted else "ACCESS DENIED"
                b_col = COLOR_GREEN if result_granted else COLOR_RED
                t_size = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                t_x = (w - t_size[0]) // 2
                cv2.putText(display_cam, banner_text, (t_x, h // 2 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5, cv2.LINE_AA)
                cv2.putText(display_cam, banner_text, (t_x, h // 2 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, b_col, 3, cv2.LINE_AA)

                face_info_dict = {
                    "match": result_face_match,
                    "score": result_face_score,
                    "fail": result_face_fail,
                    "enrolled": face_enrolled,
                }
                n_macro = result_details.get("macro_gates_passed", 0) if result_details else 0
                if result_details.get("passes_macro", False) or n_macro == 4:
                    gate_status_str = "4/4 PASSED"
                elif n_macro >= 3:
                    gate_status_str = f"{n_macro}/4 PASSED"
                else:
                    gate_status_str = f"{n_macro}/4 GATES PASSED (FAILED)"

                gesture_info_dict = {
                    "frames": 60,
                    "dtw": result_distance,
                    "order": result_transition_dissim,
                    "segment": result_segment_max,
                    "gate_status": gate_status_str,
                }
                hand_info_dict = {
                    "anthro_match": result_anthro_match,
                    "anthro_conf": result_anthro_conf,
                    "kin_match": result_kin_match,
                    "jerk_ratio": result_jerk_ratio,
                }
                result_info_dict = {
                    "granted": result_granted,
                    "fused_score": result_fused_score,
                    "reason": result_details.get("failure_reason") if result_details else None,
                }

                dashboard = build_dashboard(
                    display_cam, state=state_label, username=username, threshold=threshold,
                    num_templates=len(stored_templates), face_info=face_info_dict,
                    gesture_info=gesture_info_dict, hand_info=hand_info_dict,
                    result_info=result_info_dict
                )
                cv2.imshow(WINDOW_NAME, dashboard)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('t') or key == ord('T'):
                    show_result = False
                    clean_frames_recorded = []
                    print("  Trying again... Press [R] when ready.")
                continue

            # ─── Process Hand Landmarks ───────────────────────────────
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

            # ─── Real-Time Facial Identity Tracking (Idle Mode) ───────
            if not is_recording:
                if face_profile is not None:
                    live_face_match, live_face_score, live_face_conf, live_bbox, live_face_fail = verify_face(clean_frame, face_profile)
                    if live_bbox is not None:
                        fx, fy, fw, fh = live_bbox
                        if live_face_match:
                            box_color = COLOR_GREEN
                            tag = f"[MATCH] {username} ({live_face_conf:.0%})"
                        elif live_face_fail == "lookalike_sibling_detected":
                            box_color = COLOR_AMBER
                            tag = f"[ALERT] Sibling / Lookalike ({live_face_score:.2f})"
                        else:
                            box_color = COLOR_RED
                            tag = f"[ALERT] Impostor ({live_face_score:.2f})"
                        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), box_color, 2)
                        cv2.putText(frame, tag, (fx, max(20, fy - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA)

            # ─── Recording Logic ──────────────────────────────────────
            if is_recording:
                elapsed = time.time() - recording_start_time
                remaining = max(0.0, RECORDING_DURATION_SEC - elapsed)
                progress = min(1.0, elapsed / RECORDING_DURATION_SEC)

                # Store clean un-annotated frame for face biometric verification
                clean_frames_recorded.append(clean_frame)

                if hand_visible:
                    landmarks = extract_landmarks(
                        results.multi_hand_landmarks[0]
                    )
                    recorded_frames.append(landmarks)
                else:
                    # Subtle warning badge on camera frame
                    cv2.rectangle(frame, (w // 2 - 120, h - 35), (w // 2 + 120, h - 10), (0, 0, 160), -1)
                    cv2.putText(frame, "HAND NOT DETECTED", (w // 2 - 95, h - 17),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

                if elapsed >= RECORDING_DURATION_SEC:
                    is_recording = False
                    result_frame = frame.copy()

                    if len(recorded_frames) < 10:
                        print(f"  [FAIL] Too few frames ({len(recorded_frames)}). Please try again.")
                        show_result = True
                        result_granted = False
                        result_details = {"failure_reason": "too_few_frames", "passes_macro": False}
                    else:
                        raw = np.array(recorded_frames)
                        normalized = normalize_gesture(raw, TARGET_FRAMES)

                        # Stage 1: Multimodal Facial Identity Verification
                        if face_profile is not None:
                            n_clean = len(clean_frames_recorded)
                            sample_indices = np.linspace(0, n_clean - 1, min(6, n_clean), dtype=int) if n_clean > 0 else []
                            verified_results = []
                            for idx in sample_indices:
                                f_match, f_score, f_conf, f_bbox, f_fail = verify_face(
                                    clean_frames_recorded[idx], face_profile
                                )
                                if f_bbox is not None:
                                    verified_results.append((f_match, f_score, f_conf, f_bbox, f_fail))

                            if verified_results:
                                scores = [it[1] for it in verified_results]
                                median_score = float(np.median(scores))

                                sorted_by_score = sorted(verified_results, key=lambda it: it[1])
                                med_face = sorted_by_score[len(sorted_by_score) // 2]

                                n_passes = sum(1 for it in verified_results if it[0])
                                majority_passed = n_passes >= ((len(verified_results) + 1) // 2)

                                final_match = majority_passed and (median_score >= SFACE_COSINE_THRESHOLD)
                                if final_match:
                                    f_fail_reason = None
                                elif median_score >= SFACE_LOOKALIKE_FLOOR:
                                    f_fail_reason = "lookalike_sibling_detected"
                                else:
                                    f_fail_reason = "impostor_face_identity"

                                result_face_match = final_match
                                result_face_score = median_score
                                result_face_conf = med_face[2]
                                result_face_fail = f_fail_reason
                            else:
                                result_face_match, result_face_score, result_face_conf, result_face_fail = False, 0.0, 0.0, "no_face_detected"
                        else:
                            result_face_match, result_face_score, result_face_conf, result_face_fail = None, 1.0, 1.0, None

                        clean_frames_recorded = []

                        print(f"  Comparing against {len(stored_templates)} template(s)...")

                        (result_granted, result_distance, best_idx,
                         auth_details) = authenticate_with_details(
                            normalized,
                            stored_templates,
                            threshold,
                            finger_state_threshold,
                            transition_threshold,
                            segment_threshold,
                            anthro_profile,
                            kinematic_profile,
                            face_match=result_face_match,
                            face_confidence=result_face_conf,
                            face_details={"score": result_face_score, "reason": result_face_fail},
                        )
                        result_details = auth_details
                        result_finger_mismatch = auth_details["finger_state_mismatch"]
                        result_transition_dissim = auth_details["transition_dissimilarity"]
                        result_segment_max = auth_details["segment_max_mismatch"]
                        result_fused_score = auth_details["fused_score"]
                        result_anthro_match = auth_details.get("anthropometric_match", True)
                        result_anthro_conf = auth_details.get("anthropometric_confidence", 1.0)
                        result_kin_match = auth_details.get("kinematic_match", True)
                        result_kin_conf = auth_details.get("kinematic_confidence", 1.0)
                        result_jerk_ratio = auth_details.get("kinematic_details", {}).get("jerk_ratio", 1.0)

                        # Terminal logging
                        print()
                        print("  =============================")
                        print(f"     {'ACCESS GRANTED' if result_granted else 'ACCESS DENIED'}")
                        print("  =============================")
                        print(f"  Confidence:      {result_fused_score * 100:.1f}%")
                        if face_profile is not None:
                            print(f"  Face Identity:   {'VERIFIED' if result_face_match else result_face_fail} (Cosine {result_face_score:.3f})")
                        print(f"  Hand Anatomy:    {'MATCH' if result_anthro_match else 'IMPOSTOR'} ({result_anthro_conf:.0%})")
                        print(f"  Movement Rhythm: {result_kin_conf:.0%} fluidity (Jerk: {result_jerk_ratio:.2f}x)")
                        print(f"  DTW Distance:    {result_distance:.4f} (best match: sample {best_idx + 1})")
                        print(f"  Frames captured: {len(recorded_frames)}")

                        # ── Hardened Multimodal Template Aging ───
                        if result_granted:
                            aging_result = update_templates_if_high_confidence(
                                username, normalized,
                                result_distance, threshold,
                                stored_templates,
                                anthropometric_match=result_anthro_match,
                                anthropometric_confidence=result_anthro_conf,
                                kinematic_match=result_kin_match,
                                kinematic_confidence=result_kin_conf,
                                face_match=result_face_match,
                                face_confidence=result_face_conf,
                            )
                            if aging_result:
                                print(f"  [ADAPT] Template {aging_result['replaced_template']} updated.")
                                stored_templates = load_all_user_templates(username)
                                threshold = load_user_threshold(username)
                                finger_state_threshold = load_user_finger_state_threshold(username)
                                transition_threshold = load_user_transition_threshold(username)
                                segment_threshold = load_user_segment_threshold(username)

                        print()
                        print("  Press [T] to try again, [Q] to quit.")
                        show_result = True
                        recorded_frames = []

            # ─── Render Active Dashboard ──────────────────────────────
            cur_state = "RECORDING" if is_recording else "IDLE"
            face_info_dict = {
                "match": live_face_match if not is_recording else result_face_match,
                "score": live_face_score if not is_recording else result_face_score,
                "fail": live_face_fail if not is_recording else result_face_fail,
                "enrolled": face_enrolled,
            }
            rec_dict = {
                "remaining": remaining if is_recording else 0.0,
                "progress": progress if is_recording else 0.0,
                "frames": len(recorded_frames),
            }
            gest_dict = {
                "frames": len(recorded_frames),
            }

            dashboard = build_dashboard(
                frame, state=cur_state, username=username, threshold=threshold,
                num_templates=len(stored_templates), face_info=face_info_dict,
                gesture_info=gest_dict if is_recording else None,
                recording_info=rec_dict if is_recording else None,
            )
            cv2.imshow(WINDOW_NAME, dashboard)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif (key == ord('r') or key == ord('R')) and not is_recording:
                if hand_visible:
                    is_recording = True
                    recording_start_time = time.time()
                    recorded_frames = []
                    print(f"  [REC] Authenticating -- perform your gesture for {RECORDING_DURATION_SEC} seconds...")
                else:
                    print("  [!] Cannot start -- no hand detected.")

    except KeyboardInterrupt:
        print("\n  Interrupted.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print("  Resources released. Goodbye!\n")


if __name__ == "__main__":
    main()
