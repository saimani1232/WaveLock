"""
Camera Management & Hardware Discovery Utilities for WaveLock.

Provides robust camera initialization, device enumeration, auto-detection
of external webcams, DirectShow backend optimization for Windows, and
warmup frame flushing to eliminate initial black sensor frames.
"""

import os
import sys
import time
import cv2
import numpy as np

# Silence noisy OpenCV internal C++ warnings
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
if hasattr(cv2, "setLogLevel"):
    cv2.setLogLevel(0)


def get_preferred_backend():
    """Return the preferred cv2 VideoCapture backend for the current OS."""
    if os.name == "nt":
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


def get_windows_camera_devices():
    """
    Query Windows PnP entity database for connected camera devices.
    Returns list of dicts: [{'name': '...', 'status': 'OK'|'Error', 'instance_id': '...'}]
    """
    if os.name != "nt":
        return []

    try:
        import subprocess, json
        cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-PnpDevice -Class Camera | Select-Object FriendlyName, Status, InstanceId | ConvertTo-Json"
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=5).strip()
        if not out:
            return []
        data = json.loads(out)
        if isinstance(data, dict):
            data = [data]
        devices = []
        for item in data:
            devices.append({
                "name": item.get("FriendlyName", "Unknown Camera"),
                "status": item.get("Status", "Unknown"),
                "instance_id": item.get("InstanceId", "")
            })
        return devices
    except Exception:
        return []


def list_available_cameras(max_tested=4):
    """
    Probe connected camera devices and return their operational status.
    Fast, non-blocking scan using DirectShow.
    """
    backend = get_preferred_backend()
    backend_name = "DirectShow" if backend == cv2.CAP_DSHOW else "Default"
    results = []

    for idx in range(max_tested):
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened() and idx == 0 and backend == cv2.CAP_DSHOW:
            cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
            if cap.isOpened():
                backend_name = "Default"

        if not cap.isOpened():
            cap.release()
            # DirectShow camera indices in Windows are strictly contiguous (0, 1, ...).
            # If index 1 does not open, index 2 will never exist.
            if idx > 0:
                break
            continue

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # USB webcams take ~0.6-1.0s (10-12 frames) to warm up auto-exposure
        mean_b = 0.0
        for _ in range(16):
            ret, frame = cap.read()
            if ret and frame is not None:
                mean_b = float(np.mean(frame))
                if mean_b > 5.0:
                    break
            time.sleep(0.04)

        status = "READY" if mean_b > 5.0 else "BLACK_FRAME (Lens covered or initializing)"

        results.append({
            "index": idx,
            "opened": True,
            "backend": backend_name,
            "width": w,
            "height": h,
            "fps": fps,
            "mean_brightness": round(mean_b, 2),
            "status": status,
        })
        cap.release()

    return results


def print_camera_diagnostics(max_tested=4):
    """Print formatted camera diagnostics to console."""
    print("\n  ============================================================")
    print("                    WAVELOCK CAMERA DIAGNOSTICS")
    print("  ============================================================")

    # Windows PnP Hardware
    hw_devices = get_windows_camera_devices()
    if hw_devices:
        print("  Windows Hardware PnP Status:")
        for hw in hw_devices:
            status_tag = "[ACTIVE / OK]" if hw["status"] == "OK" else f"[{hw['status']}]"
            print(f"    * {hw['name']}: {status_tag}")
        print()

    print("  OpenCV VideoCapture Scan:")
    cams = list_available_cameras(max_tested=max_tested)
    found_any = False
    for c in cams:
        if c["opened"]:
            found_any = True
            status_tag = "[OK] READY" if c["status"] == "READY" else f"[WARN] {c['status']}"
            print(f"    Camera Index {c['index']}:")
            print(f"      - Status:      {status_tag}")
            print(f"      - Resolution:  {c['width']}x{c['height']}")
            print(f"      - Backend:     {c['backend']}")
            print(f"      - Brightness:  {c['mean_brightness']:.1f} / 255.0")
            print()

    if not found_any:
        print("    [!] No camera feeds could be opened.")
        print("        Check USB webcam connection and Windows privacy settings.\n")
    print("  ============================================================\n")


def detect_best_camera(preferred_index=None, max_tested=4):
    """
    Detect the best working camera index.

    If preferred_index is provided and functional, returns it.
    Otherwise, scans available indices and returns the first healthy camera.
    """
    if preferred_index is not None:
        idx = int(preferred_index)
        backend = get_preferred_backend()
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened() and backend == cv2.CAP_DSHOW:
            cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
        if cap.isOpened():
            cap.release()
            return idx
        else:
            print(f"  [WARN] Preferred camera index {idx} failed to open.")
            print("         Scanning for alternative working cameras...")

    cams = list_available_cameras(max_tested=max_tested)
    ready_cams = [c for c in cams if c["status"] == "READY"]

    if ready_cams:
        # If multiple cameras are connected, prioritize external webcam (index >= 1)
        external_cams = [c for c in ready_cams if c["index"] > 0]
        if external_cams:
            chosen = external_cams[0]["index"]
            print(f"  [CAM] Auto-detected external camera (Index {chosen}). Use '--camera 0' for internal camera.")
            return chosen
        return ready_cams[0]["index"]

    opened_cams = [c for c in cams if c["opened"]]
    if opened_cams:
        return opened_cams[0]["index"]

    return 0


def open_camera(camera_index=None, width=640, height=480, warmup_frames=18):
    """
    Open the webcam with DirectShow/MSMF handling and sensor warmup.

    Flushes initial sensor warmup frames so the very first frame received
    by WaveLock is properly exposed and ready for computer vision.

    Args:
        camera_index: int or None (auto-detect if None).
        width: Desired frame width (default: 640).
        height: Desired frame height (default: 480).
        warmup_frames: Max warmup frames to flush (default: 18).

    Returns:
        cv2.VideoCapture: Initialized and warmed-up VideoCapture object.
    """
    if camera_index is None:
        chosen_index = detect_best_camera()
    else:
        chosen_index = int(camera_index)

    backend = get_preferred_backend()
    backend_name = "DirectShow" if backend == cv2.CAP_DSHOW else "Default"

    print(f"  [CAM] Connecting to camera index {chosen_index} ({backend_name})...")
    cap = cv2.VideoCapture(chosen_index, backend)

    if not cap.isOpened() and backend == cv2.CAP_DSHOW:
        print("  [CAM] DirectShow failed, retrying with default backend...")
        cap = cv2.VideoCapture(chosen_index, cv2.CAP_ANY)
        backend_name = "Default"

    if not cap.isOpened():
        print()
        print(f"  ERROR: Could not open camera index {chosen_index}!")
        print("  Tips to troubleshoot:")
        print("    1. Check that your USB webcam is securely plugged in.")
        print("    2. Ensure another app (Teams, Zoom, Camera app) is not using it.")
        print("    3. Run 'python gesture_auth.py --list-cams' to inspect connected devices.")
        print(f"    4. Try specifying an explicit camera: 'python gesture_auth.py --camera {chosen_index}'")
        print()
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # ── Sensor Warmup Phase ───────────────────────────────────────────────────
    # USB webcams (e.g. Logitech C270) take ~0.6-1.0s (10-12 frames) to negotiate
    # auto-exposure and white-balance. Flush warmup frames until non-black arrives.
    warmed_up = False
    last_mean = 0.0
    for _ in range(warmup_frames):
        ret, test_frame = cap.read()
        if ret and test_frame is not None:
            last_mean = float(np.mean(test_frame))
            if last_mean > 8.0:
                warmed_up = True
                break
        time.sleep(0.04)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if warmed_up:
        print(f"  [CAM] Ready: Camera {chosen_index} ({actual_w}x{actual_h}, {backend_name}, auto-exposure locked).")
    else:
        print(f"  [CAM] Active: Camera {chosen_index} ({actual_w}x{actual_h}, warning: low brightness mean={last_mean:.1f}).")

    return cap
