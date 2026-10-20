"""Placeholder for desktop-only modules in the WaveLock mobile app.

In the browser the camera, MediaPipe Hands, YuNet and SFace run in
JavaScript; the Python modules only import cv2 / mediapipe at the top. Any
attribute can be looked up (module-level names stay importable), but using
one raises, so an accidental desktop-only call fails loudly instead of
doing something wrong.
"""


class Missing:
    def __init__(self, name):
        self._name = name

    def __getattr__(self, attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        return Missing(f"{self._name}.{attr}")

    def __call__(self, *args, **kwargs):
        raise RuntimeError(f"{self._name} is not available in the WaveLock mobile app")

    def __repr__(self):
        return f"<missing {self._name}>"
