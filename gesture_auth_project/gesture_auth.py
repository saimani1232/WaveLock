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

import time
import argparse

import cv2
import numpy as np

from utils.landmarks import extract_landmarks, is_hand_detected
from utils.normalize import normalize_gesture, extract_wrist_trajectory
from utils.kinematics import prepare_kinematic_sequence
from gesture_compare import (
    load_user_profile,
    normalize_username,
    authenticate_with_details,
    update_templates_if_high_confidence,
    list_registered_users,
)
from utils.face_auth import (
    verify_face,
    verify_face_over_frames,
    SFACE_COSINE_THRESHOLD,
    SFACE_LOOKALIKE_FLOOR,
)
from gesture_capture import (
    setup_mediapipe,
    setup_camera,
    RECORDING_DURATION_SEC,
    TARGET_FRAMES,
    MIN_RECORDED_FRAMES,
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

# Idle preview face check runs two DNNs; doing it every frame starves the
# camera loop, so it is throttled and the last result is redrawn.
FACE_PREVIEW_INTERVAL_SEC = 0.25


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
    elif f_fail == "no_face_detected":
        f_status = "REJECTED: NO FACE DETECTED"
        f_color = COLOR_RED
    elif f_fail == "unstable_face_match":
        f_status = f"REJECTED: UNSTABLE MATCH ({f_score:.2f})"
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
# ATTEMPT EVALUATION (camera-independent, unit-tested)
# ============================================================================

def new_result():
    """A fresh result record. Every attempt starts from this, so values from a
    previous attempt can never leak into the display of the next one."""
    return {
        "granted": False,
        "reason": None,
        "frames": 0,
        "distance": None,
        "best_index": None,
        "transition_dissimilarity": None,
        "segment_max_mismatch": None,
        "fused_score": None,
        "details": None,
        "face_match": None,
        "face_score": None,
        "face_confidence": None,
        "face_reason": None,
        "pose": None,
        "raw": None,
    }


def evaluate_attempt(recorded_frames, clean_frames, frame_aspect, profile):
    """
    Run the full multimodal decision for one recorded attempt.

    Args:
        recorded_frames: list of raw (21, 3) landmark arrays (hand frames).
        clean_frames: un-annotated BGR frames captured during recording.
        frame_aspect: camera frame width / height.
        profile: dict from gesture_compare.load_user_profile().

    Returns:
        dict (see new_result()).
    """
    result = new_result()
    result["frames"] = len(recorded_frames)
    if len(recorded_frames) < MIN_RECORDED_FRAMES:
        result["reason"] = "too_few_frames"
        result["details"] = {"failure_reason": "too_few_frames", "passes_macro": False}
        return result

    raw = np.asarray(recorded_frames, dtype=np.float64)
    pose = normalize_gesture(raw, TARGET_FRAMES)

    # Stage 1: face identity across the recording (median + majority vote).
    if profile["face_embedding"] is not None:
        face = verify_face_over_frames(clean_frames, profile["face_embedding"])
        face_match = face["match"]
        face_score = face["score"]
        face_conf = face["confidence"]
        face_reason = face["reason"]
    else:
        face_match, face_score, face_conf, face_reason = None, None, 1.0, None

    live_trajectory = None
    if profile["trajectory_templates"] is not None:
        live_trajectory = extract_wrist_trajectory(raw, frame_aspect, TARGET_FRAMES)

    granted, distance, best_idx, details = authenticate_with_details(
        pose,
        profile["templates"],
        profile["threshold"],
        profile["finger_state_threshold"],
        profile["transition_threshold"],
        profile["segment_threshold"],
        profile["anthropometric_profile"],
        profile["kinematic_profile"],
        face_match=face_match,
        face_confidence=face_conf,
        face_details={"score": face_score, "reason": face_reason},
        live_trajectory=live_trajectory,
        stored_trajectories=profile["trajectory_templates"],
        trajectory_threshold=profile["trajectory_threshold"],
        live_kinematic_sequence=prepare_kinematic_sequence(raw, TARGET_FRAMES),
    )

    result.update({
        "granted": bool(granted),
        "reason": details["failure_reason"],
        "distance": distance,
        "best_index": best_idx,
        "transition_dissimilarity": details["transition_dissimilarity"],
        "segment_max_mismatch": details["segment_max_mismatch"],
        "fused_score": details["fused_score"],
        "details": details,
        "face_match": face_match,
        "face_score": face_score,
        "face_confidence": face_conf,
        "face_reason": face_reason,
        "pose": pose,
        "raw": raw,
    })
    return result


def face_info_from_result(result, face_enrolled):
    return {
        "match": result["face_match"],
        "score": result["face_score"] or 0.0,
        "fail": result["face_reason"],
        "enrolled": face_enrolled,
    }


def gesture_info_from_result(result):
    details = result["details"] or {}
    n_macro = details.get("macro_gates_passed", 0)
    if details.get("passes_macro", False):
        gate_status = f"{n_macro}/4 PASSED"
    else:
        gate_status = f"{n_macro}/4 GATES PASSED (FAILED)"
    if details.get("trajectory_active") and not details.get("passes_trajectory", True):
        gate_status += " | PATH FAIL"
    return {
        "frames": result["frames"],
        "dtw": result["distance"],
        "order": result["transition_dissimilarity"],
        "segment": result["segment_max_mismatch"],
        "gate_status": gate_status,
    }


def hand_info_from_result(result):
    details = result["details"]
    if not details or "anthropometric_match" not in details:
        return None          # not evaluated (e.g. too few frames)
    return {
        "anthro_match": details["anthropometric_match"],
        "anthro_conf": details["anthropometric_confidence"],
        "kin_match": details["kinematic_match"],
        "jerk_ratio": details.get("kinematic_details", {}).get("jerk_ratio", 1.0),
    }


def print_result(result, face_enrolled):
    details = result["details"] or {}
    print()
    print("  =============================")
    print(f"     {'ACCESS GRANTED' if result['granted'] else 'ACCESS DENIED'}")
    print("  =============================")
    if result["reason"] == "too_few_frames":
        print(f"  Too few hand frames ({result['frames']}). Keep your hand in view.")
        return
    print(f"  Confidence:      {details['fused_score'] * 100:.1f}%")
    if face_enrolled:
        status = "VERIFIED" if result["face_match"] else result["face_reason"]
        print(f"  Face Identity:   {status} (Cosine {result['face_score']:.3f})")
    print(f"  Hand Anatomy:    {'MATCH' if details['anthropometric_match'] else 'IMPOSTOR'} "
          f"({details['anthropometric_confidence']:.0%})")
    print(f"  Movement Rhythm: {details['kinematic_confidence']:.0%} fluidity "
          f"(Jerk: {details['kinematic_details'].get('jerk_ratio', 0):.2f}x)")
    print(f"  DTW Distance:    {result['distance']:.4f} (best match: sample {result['best_index'] + 1})")
    if details.get("trajectory_active"):
        print(f"  Hand Path:       {details['trajectory_distance']:.3f} "
              f"(limit {details['trajectory_threshold']:.3f})")
    if not result["granted"]:
        print(f"  Reason:          {result['reason']}")
    print(f"  Frames captured: {result['frames']}")


def print_profile_summary(profile):
    print(f"  Loaded {len(profile['templates'])} template(s) for '{profile['username']}'")
    print(f"  Authentication threshold: {profile['threshold']:.4f}")
    print(f"  Finger mismatch limit:    {profile['finger_state_threshold']:.4f}")
    print(f"  Transition order limit:   {profile['transition_threshold']:.4f}")
    print(f"  Segment mismatch limit:   {profile['segment_threshold']:.4f}")
    anthro = profile["anthropometric_profile"]
    print(f"  Hand bone aspect ratio:   {anthro['baseline'][0]:.4f} (+/-{anthro['tolerance']:.0%})")
    if profile["trajectory_threshold"] is not None:
        print(f"  Hand path limit:          {profile['trajectory_threshold']:.4f}")
    else:
        print("  Hand path gate:           [DISABLED] Legacy profile without raw recordings")
        print("                            Re-register to enable path matching and")
        print("                            frame-rate-independent kinematics.")
    if profile["face_embedding"] is not None:
        print("  Face Identity Layer:      [ACTIVE] Multimodal Gate Enforced")
    else:
        print("  Face Identity Layer:      [DISABLED] Legacy Mode (No face enrolled)")
        print(f"                            To enable: python enroll_face.py --user {profile['username']}")


# ============================================================================
# MAIN AUTHENTICATION FLOW
# ============================================================================

def main():
    print("=" * 60)
    print("   WAVELOCK -- Multimodal Biometric Authentication Terminal")
    print("=" * 60)

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
    if not raw_user:
        raw_user = input("  Enter your username: ")
    try:
        username = normalize_username(raw_user)
    except ValueError as e:
        print(f"  ERROR: {e}")
        return
    if username not in registered_users:
        print(f"  ERROR: User '{username}' not registered.")
        return

    # ─── Load User Profile ────────────────────────────────────────────
    try:
        profile = load_user_profile(username)
    except (FileNotFoundError, ValueError) as e:
        print(f"  ERROR: {e}")
        return
    print_profile_summary(profile)
    face_enrolled = profile["face_embedding"] is not None

    print()
    print("  Controls:")
    print(f"    [R]  Record gesture to authenticate ({RECORDING_DURATION_SEC} seconds)")
    print("    [Q]  Quit")
    print()

    # ─── Initialize ───────────────────────────────────────────────────
    hands, mp_hands, mp_drawing, mp_drawing_styles = setup_mediapipe()
    cap = None
    try:
        cap = setup_camera(camera_index=args.camera)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

        print()
        print(f"  Authenticating as: {username}")
        print("  Show your hand and press [R] to begin.")
        print("  " + "-" * 30)
        print()

        # ─── State ────────────────────────────────────────────────────
        is_recording = False
        recording_start_time = 0.0
        recorded_frames = []
        clean_frames_recorded = []
        show_result = False
        result = new_result()
        result_frame = None

        live_face = (None, 0.0, 1.0, None, None)   # match, score, conf, bbox, fail
        last_face_check = 0.0

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("  ERROR: Failed to read frame.")
                break

            frame = cv2.flip(frame, 1)
            clean_frame = frame.copy()
            h, w = frame.shape[:2]

            # ─── Result Display Mode ──────────────────────────────────
            if show_result:
                granted = result["granted"]
                display_cam = result_frame.copy()

                overlay = display_cam.copy()
                tint_color = (0, 70, 0) if granted else (0, 0, 70)
                cv2.rectangle(overlay, (0, 0), (w, h), tint_color, -1)
                cv2.addWeighted(overlay, 0.22, display_cam, 0.78, 0, display_cam)

                banner_text = "ACCESS GRANTED" if granted else "ACCESS DENIED"
                b_col = COLOR_GREEN if granted else COLOR_RED
                t_size = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                t_x = (w - t_size[0]) // 2
                cv2.putText(display_cam, banner_text, (t_x, h // 2 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5, cv2.LINE_AA)
                cv2.putText(display_cam, banner_text, (t_x, h // 2 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, b_col, 3, cv2.LINE_AA)

                dashboard = build_dashboard(
                    display_cam, state="GRANTED" if granted else "DENIED",
                    username=username, threshold=profile["threshold"],
                    num_templates=len(profile["templates"]),
                    face_info=face_info_from_result(result, face_enrolled),
                    gesture_info=gesture_info_from_result(result),
                    hand_info=hand_info_from_result(result),
                    result_info={
                        "granted": granted,
                        "fused_score": result["fused_score"] or 0.0,
                        "reason": result["reason"],
                    },
                )
                cv2.imshow(WINDOW_NAME, dashboard)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q')):
                    break
                elif key in (ord('t'), ord('T')):
                    show_result = False
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
            if not is_recording and face_enrolled:
                now = time.time()
                if now - last_face_check >= FACE_PREVIEW_INTERVAL_SEC:
                    live_face = verify_face(clean_frame, profile["face_embedding"])
                    last_face_check = now
                live_match, live_score, live_conf, live_bbox, live_fail = live_face
                if live_bbox is not None:
                    fx, fy, fw, fh = live_bbox
                    if live_match:
                        box_color = COLOR_GREEN
                        tag = f"[MATCH] {username} ({live_conf:.0%})"
                    elif live_fail == "lookalike_sibling_detected":
                        box_color = COLOR_AMBER
                        tag = f"[ALERT] Sibling / Lookalike ({live_score:.2f})"
                    else:
                        box_color = COLOR_RED
                        tag = f"[ALERT] Impostor ({live_score:.2f})"
                    cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), box_color, 2)
                    cv2.putText(frame, tag, (fx, max(20, fy - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA)

            # ─── Recording Logic ──────────────────────────────────────
            remaining, progress = 0.0, 0.0
            if is_recording:
                elapsed = time.time() - recording_start_time
                remaining = max(0.0, RECORDING_DURATION_SEC - elapsed)
                progress = min(1.0, elapsed / RECORDING_DURATION_SEC)

                # Store clean un-annotated frame for face biometric verification
                clean_frames_recorded.append(clean_frame)

                if hand_visible:
                    recorded_frames.append(extract_landmarks(results.multi_hand_landmarks[0]))
                else:
                    cv2.rectangle(frame, (w // 2 - 120, h - 35), (w // 2 + 120, h - 10), (0, 0, 160), -1)
                    cv2.putText(frame, "HAND NOT DETECTED", (w // 2 - 95, h - 17),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

                if elapsed >= RECORDING_DURATION_SEC:
                    is_recording = False
                    result_frame = frame.copy()
                    print(f"  Comparing against {len(profile['templates'])} template(s)...")
                    result = evaluate_attempt(
                        recorded_frames, clean_frames_recorded, w / h, profile
                    )
                    recorded_frames = []
                    clean_frames_recorded = []
                    print_result(result, face_enrolled)

                    # ── Hardened Multimodal Template Aging ───
                    if result["granted"]:
                        d = result["details"]
                        aging = update_templates_if_high_confidence(
                            username, result["pose"], result["distance"],
                            profile["threshold"], profile["templates"],
                            anthropometric_match=d["anthropometric_match"],
                            anthropometric_confidence=d["anthropometric_confidence"],
                            kinematic_match=d["kinematic_match"],
                            kinematic_confidence=d["kinematic_confidence"],
                            face_match=result["face_match"],
                            face_confidence=result["face_confidence"],
                            live_raw=result["raw"],
                        )
                        if aging:
                            print(f"  [ADAPT] Template {aging['replaced_template']} updated.")
                            profile = load_user_profile(username)

                    print()
                    print("  Press [T] to try again, [Q] to quit.")
                    show_result = True

            # ─── Render Active Dashboard ──────────────────────────────
            cur_state = "RECORDING" if is_recording else "IDLE"
            if is_recording:
                face_info_dict = {"match": None, "score": 0.0, "fail": None,
                                  "enrolled": face_enrolled}
            else:
                face_info_dict = {"match": live_face[0] if face_enrolled else None,
                                  "score": live_face[1], "fail": live_face[4],
                                  "enrolled": face_enrolled}

            dashboard = build_dashboard(
                frame, state=cur_state, username=username, threshold=profile["threshold"],
                num_templates=len(profile["templates"]), face_info=face_info_dict,
                gesture_info={"frames": len(recorded_frames)} if is_recording else None,
                recording_info={"remaining": remaining, "progress": progress,
                                "frames": len(recorded_frames)} if is_recording else None,
            )
            cv2.imshow(WINDOW_NAME, dashboard)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break
            elif key in (ord('r'), ord('R')) and not is_recording and not show_result:
                if hand_visible:
                    is_recording = True
                    recording_start_time = time.time()
                    recorded_frames = []
                    clean_frames_recorded = []
                    result = new_result()
                    print(f"  [REC] Authenticating -- perform your gesture for {RECORDING_DURATION_SEC} seconds...")
                else:
                    print("  [!] Cannot start -- no hand detected.")

    except KeyboardInterrupt:
        print("\n  Interrupted.")

    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print("  Resources released. Goodbye!\n")


if __name__ == "__main__":
    main()
