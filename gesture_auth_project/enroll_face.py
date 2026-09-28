"""
Face Enrollment Tool for WaveLock Multimodal Authentication.

Captures frontal face frames from webcam, computes 128-D SFace feature embeddings,
and saves the master face profile to templates/<username>/face_embedding.npy.
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np

from utils.face_auth import (
    detect_primary_face,
    extract_face_embedding,
    save_face_embedding,
    load_face_embedding,
    verify_face,
)
from utils.camera_utils import open_camera, print_camera_diagnostics

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
WINDOW_NAME = "WaveLock — Face Enrollment"


def enroll_face_interactive(username, camera_index=0):
    """Interactively capture and enroll face from webcam."""
    print()
    print("=" * 60)
    print(f"  WAVELOCK FACE ENROLLMENT — User: '{username}'")
    print("=" * 60)
    print("  Instructions:")
    print("    1. Look directly at the webcam.")
    print("    2. Ensure good lighting on your face.")
    print("    3. Press [SPACE] or [F] when the green face box appears.")
    print("    4. Press [Q] to cancel.")
    print()

    cap = open_camera(camera_index=camera_index, width=640, height=480)

    embeddings = []
    REQUIRED_FRAMES = 5
    enrolling = False
    enrolled_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("  ERROR: Camera frame dropped.")
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            display = frame.copy()

            face_row, bbox = detect_primary_face(frame)
            face_detected = face_row is not None

            # Overlay guide
            cv2.putText(
                display, f"Face Enrollment: {username}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 100), 2, cv2.LINE_AA
            )

            if face_detected:
                fx, fy, fw, fh = bbox
                cv2.rectangle(display, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
                score = face_row[-1]
                cv2.putText(
                    display, f"Face Detected ({score:.0%})", (fx, max(20, fy - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
                )
                if not enrolling:
                    cv2.putText(
                        display, "Press [SPACE] to capture (Anti-Sibling High Security)", (15, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2, cv2.LINE_AA
                    )
            else:
                cv2.putText(
                    display, "Position your face in center of camera...", (15, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 150, 255), 1, cv2.LINE_AA
                )

            # Capture burst
            if enrolling:
                if face_detected:
                    emb = extract_face_embedding(frame, face_row)
                    embeddings.append(emb)
                    enrolled_count += 1
                    cv2.putText(
                        display, f"Capturing high-fidelity frame {enrolled_count}/{REQUIRED_FRAMES}...",
                        (w // 2 - 190, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA
                    )

                if enrolled_count >= REQUIRED_FRAMES:
                    # Average and normalize master embedding
                    master_emb = np.mean(embeddings, axis=0)
                    master_emb /= np.linalg.norm(master_emb)
                    save_path = save_face_embedding(username, master_emb, TEMPLATES_DIR)

                    print()
                    print("  [SUCCESS] High-security face profile enrolled successfully!")
                    print(f"  Master 128-D anti-sibling embedding saved to: {save_path}")
                    print()

                    # Show confirmation
                    cv2.rectangle(display, (0, h // 2 - 40), (w, h // 2 + 40), (0, 180, 0), -1)
                    cv2.putText(
                        display, "FACE ENROLLED SUCCESSFULLY!", (w // 2 - 180, h // 2 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
                    )
                    cv2.imshow(WINDOW_NAME, display)
                    cv2.waitKey(1200)
                    return True

            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == ord('Q'):
                print("  Face enrollment cancelled.")
                return False

            elif (key == ord(' ') or key == ord('f') or key == ord('F')) and face_detected and not enrolling:
                enrolling = True
                print("  Capturing face frames...")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return False


def main():
    parser = argparse.ArgumentParser(description="WaveLock Face Enrollment Tool")
    parser.add_argument("--user", "-u", type=str, default="saimani", help="Username to enroll face for")
    parser.add_argument("--camera", "--cam", "-c", dest="cam", type=int, default=None, help="Webcam index (default: auto-detect)")
    parser.add_argument("--list-cams", action="store_true", help="List available cameras and exit")
    args = parser.parse_args()

    if args.list_cams:
        print_camera_diagnostics()
        return

    user = args.user.lower().replace(" ", "_")
    enroll_face_interactive(user, camera_index=args.cam)


if __name__ == "__main__":
    main()
