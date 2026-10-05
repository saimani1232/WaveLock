"""
Face Biometrics Engine using OpenCV YuNet (Detector) and SFace (128-D Recognizer).

Provides 1:1 facial identity verification to anchor multimodal authentication,
closing the hand-geometry collision vulnerability among twins/similar-build individuals.
"""

import os
import tempfile

import cv2
import numpy as np

# Model Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
YUNET_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_PATH = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")

# Standard SFace Verification Thresholds
# OpenCV Model Zoo baseline for unrelated strangers: 0.363
# High-Security Anti-Sibling / Lookalike Threshold: 0.530
# Sibling Lookalike Detection Floor: 0.400
SFACE_PUBLIC_THRESHOLD = 0.363
SFACE_COSINE_THRESHOLD = 0.530   # Enforced operational threshold to reject sibling presentation attacks
SFACE_LOOKALIKE_FLOOR = 0.400    # Floor for detecting familial/lookalike facial correlations
SFACE_L2_THRESHOLD = 1.128

# Cached detector and recognizer singletons
_DETECTOR = None
_RECOGNIZER = None
_DETECTOR_SIZE = None


def get_face_detector(input_size=(320, 320)):
    """Initialize or update the OpenCV YuNet face detector."""
    global _DETECTOR, _DETECTOR_SIZE
    if not os.path.exists(YUNET_PATH):
        raise FileNotFoundError(f"YuNet model not found at: {YUNET_PATH}")

    if _DETECTOR is None:
        _DETECTOR = cv2.FaceDetectorYN.create(
            model=YUNET_PATH,
            config="",
            input_size=input_size,
            score_threshold=0.70,
            nms_threshold=0.30,
            top_k=5000,
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU
        )
        _DETECTOR_SIZE = input_size
    elif _DETECTOR_SIZE != input_size:
        _DETECTOR.setInputSize(input_size)
        _DETECTOR_SIZE = input_size

    return _DETECTOR


def get_face_recognizer():
    """Initialize the OpenCV SFace 128-D deep feature recognizer."""
    global _RECOGNIZER
    if not os.path.exists(SFACE_PATH):
        raise FileNotFoundError(f"SFace model not found at: {SFACE_PATH}")

    if _RECOGNIZER is None:
        _RECOGNIZER = cv2.FaceRecognizerSF.create(
            model=SFACE_PATH,
            config="",
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU
        )
    return _RECOGNIZER


def detect_primary_face(frame, min_confidence=0.70):
    """
    Detect faces in the frame and return the most prominent frontal face.

    Args:
        frame: BGR numpy image.
        min_confidence: minimum detection confidence score.

    Returns:
        tuple (face_row, bbox) or (None, None):
            - face_row: 1D numpy array with 15 elements (bbox, 5 landmarks, score)
            - bbox: tuple of (x, y, w, h)
    """
    if frame is None or frame.size == 0:
        return None, None

    h, w = frame.shape[:2]
    detector = get_face_detector(input_size=(w, h))

    _, faces = detector.detect(frame)
    if faces is None or len(faces) == 0:
        return None, None

    # Filter by score
    valid_faces = [f for f in faces if f[-1] >= min_confidence]
    if not valid_faces:
        return None, None

    # Select the largest face by bounding box area (most prominent subject)
    primary = max(valid_faces, key=lambda f: f[2] * f[3])
    bbox = (int(primary[0]), int(primary[1]), int(primary[2]), int(primary[3]))
    return primary, bbox


def extract_face_embedding(frame, face_row):
    """
    Align, crop, and compute the 128-D feature embedding for the face.

    Args:
        frame: BGR numpy image.
        face_row: 1D numpy array from YuNet detection.

    Returns:
        numpy array of shape (128,) normalized on unit sphere.
    """
    recognizer = get_face_recognizer()
    # Align and crop 112x112 face chip
    aligned_face = recognizer.alignCrop(frame, face_row)
    # Extract 128-D embedding
    embedding = recognizer.feature(aligned_face)
    return embedding.flatten()


def verify_face(live_frame, enrolled_embedding, threshold=SFACE_COSINE_THRESHOLD):
    """
    Perform 1:1 facial verification of the live camera frame against enrolled embedding.

    Args:
        live_frame: BGR numpy frame from camera.
        enrolled_embedding: numpy array of shape (128,) from user profile.
        threshold: cosine similarity match threshold (default 0.363).

    Returns:
        tuple: (is_match, cosine_sim, confidence_pct, bbox, failure_reason)
    """
    if enrolled_embedding is None:
        # Legacy user without enrolled face
        return True, 1.0, 1.0, None, None

    face_row, bbox = detect_primary_face(live_frame)
    if face_row is None:
        return False, 0.0, 0.0, None, "no_face_detected"

    recognizer = get_face_recognizer()
    live_emb = extract_face_embedding(live_frame, face_row)

    # Compute cosine similarity
    score = float(recognizer.match(
        enrolled_embedding.reshape(1, 128),
        live_emb.reshape(1, 128),
        cv2.FaceRecognizerSF_FR_COSINE
    ))

    is_match, conf, failure_reason = classify_face_score(score, threshold)
    return is_match, score, conf, bbox, failure_reason


def classify_face_score(score, threshold=SFACE_COSINE_THRESHOLD):
    """
    Map an SFace cosine similarity to (is_match, confidence, failure_reason).

    Single source of truth for the genuine / lookalike / stranger tiers, used
    by live verification and by the evaluation scripts.
    """
    score = float(score)
    is_match = score >= threshold

    # Multi-tier confidence mapping separating Genuine vs Sibling Lookalikes vs Strangers
    if is_match:
        # Genuine Zone [threshold, 0.78] -> [0.60, 1.0]
        conf = 0.60 + 0.40 * min(1.0, max(0.0, (score - threshold) / (0.78 - threshold)))
        failure_reason = None
    elif score >= SFACE_LOOKALIKE_FLOOR:
        # Sibling / Lookalike Warning Zone: [0.400, threshold)
        # Cosine is elevated due to familial genetics or disguise accessories (e.g. spectacles)
        conf = 0.35 + 0.15 * ((score - SFACE_LOOKALIKE_FLOOR) / (threshold - SFACE_LOOKALIKE_FLOOR))
        failure_reason = "lookalike_sibling_detected"
    else:
        # Clear Impostor / Stranger Zone: [0.0, 0.400)
        conf = max(0.0, min(0.35, 0.35 * (max(0.0, score) / SFACE_LOOKALIKE_FLOOR)))
        failure_reason = "impostor_face_identity"

    return is_match, conf, failure_reason


def verify_face_over_frames(frames, enrolled_embedding, max_samples=6):
    """
    Verify identity across several frames of one authentication attempt.

    Frames are sampled evenly. The attempt matches only if the MEDIAN cosine
    clears the threshold AND a majority of frames with a detected face pass
    individually, so one lucky frame cannot unlock and one occluded frame
    cannot lock out.

    Returns:
        dict with match (bool), score (median cosine), confidence, reason
        (None when matched), frames_checked, faces_detected.
    """
    n_frames = len(frames) if frames is not None else 0
    if n_frames == 0:
        return {"match": False, "score": 0.0, "confidence": 0.0,
                "reason": "no_face_detected", "frames_checked": 0,
                "faces_detected": 0}

    indices = np.linspace(0, n_frames - 1, min(max_samples, n_frames)).astype(int)
    results = [verify_face(frames[i], enrolled_embedding) for i in indices]
    detected = [r for r in results if r[3] is not None]
    if not detected:
        return {"match": False, "score": 0.0, "confidence": 0.0,
                "reason": "no_face_detected", "frames_checked": len(indices),
                "faces_detected": 0}

    median_score = float(np.median([r[1] for r in detected]))
    n_pass = sum(1 for r in detected if r[0])
    majority = n_pass >= (len(detected) + 1) // 2
    median_ok, confidence, tier_reason = classify_face_score(median_score)
    match = bool(majority and median_ok)

    if match:
        reason = None
    elif median_ok:
        # Median clears the bar but most individual frames do not.
        reason = "unstable_face_match"
    else:
        reason = tier_reason

    return {"match": match, "score": median_score, "confidence": confidence,
            "reason": reason, "frames_checked": len(indices),
            "faces_detected": len(detected)}


def save_face_embedding(username, embedding, templates_dir):
    """Atomically save the 128-D face embedding to templates/<username>/face_embedding.npy."""
    user_dir = os.path.join(templates_dir, username)
    os.makedirs(user_dir, exist_ok=True)
    out_path = os.path.join(user_dir, "face_embedding.npy")
    fd, tmp_path = tempfile.mkstemp(dir=user_dir, prefix=".tmp_", suffix=".npy")
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, np.asarray(embedding, dtype=np.float32).reshape(128))
        os.replace(tmp_path, out_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return out_path


def load_face_embedding(username, templates_dir):
    """
    Load user's enrolled face embedding.

    Returns:
        numpy array of shape (128,) or None if not enrolled (legacy user).
    """
    user_dir = os.path.join(templates_dir, username)
    face_path = os.path.join(user_dir, "face_embedding.npy")
    if os.path.exists(face_path):
        try:
            emb = np.load(face_path)
            if emb.shape == (128,):
                return emb.astype(np.float32)
        except Exception as e:
            print(f"Warning: Failed to load face embedding: {e}")
    return None
