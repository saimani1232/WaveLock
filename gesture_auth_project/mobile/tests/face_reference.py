"""
OpenCV reference for tests/face_parity.mjs (desktop face pipeline,
utils/face_auth.py, OpenCV 4.11). Writes, for each test image, the raw RGBA
pixels for Node and the desktop results: all YuNet detections, the primary
face, the 112 x 112 aligned crop, the SFace embedding and the yaw proxy.

    python tests/face_reference.py <image.png> <out_dir>
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import cv2                                   # noqa: E402
import numpy as np                           # noqa: E402
from utils import face_auth as fa            # noqa: E402
from utils import liveness as lv             # noqa: E402


def variants(img):
    """The image as given, as a 640 x 480 webcam-like frame, and mirrored."""
    frame = np.zeros((480, 640, 3), np.uint8)
    small = cv2.resize(img, (480, 480), interpolation=cv2.INTER_AREA)
    frame[:, 80:560] = small
    return {"as_is": img, "frame_640x480": frame, "mirrored": cv2.flip(frame, 1)}


def main(image_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    img = cv2.imread(image_path)
    results = {}
    for name, bgr in variants(img).items():
        h, w = bgr.shape[:2]
        det = fa.get_face_detector(input_size=(w, h))
        _, faces = det.detect(bgr)
        row, _ = fa.detect_primary_face(bgr)
        entry = {"width": w, "height": h,
                 "faces": [] if faces is None else faces.tolist(),
                 "primary": None if row is None else row.tolist()}
        if row is not None:
            rec = fa.get_face_recognizer()
            aligned = rec.alignCrop(bgr, row)
            entry["aligned_rgb"] = aligned[:, :, ::-1].reshape(-1).tolist()
            entry["embedding"] = fa.extract_face_embedding(bgr, row).tolist()
            entry["yaw"] = lv.yaw_proxy(row)
        rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
        rgba.tofile(os.path.join(out_dir, f"{name}.rgba"))
        results[name] = entry
    with open(os.path.join(out_dir, "reference.json"), "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
