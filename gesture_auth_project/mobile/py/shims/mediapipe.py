"""MediaPipe runs as its web build in the mobile app (see _missing.py)."""
from _missing import Missing


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    return Missing(f"mediapipe.{name}")
