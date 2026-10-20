"""OpenCV is not used from Python in the mobile app (see _missing.py)."""
from _missing import Missing


def setLogLevel(level):
    """utils/camera_utils.py silences OpenCV logging at import time; there is
    no OpenCV log in the browser, so this is the one call that does nothing."""


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    return Missing(f"cv2.{name}")
