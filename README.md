# WaveLock

WaveLock is a webcam-based gesture biometric authentication prototype. It lets a user register a short hand movement, saves normalized landmark templates locally, and later verifies a live gesture using Dynamic Time Warping (DTW).

The current implementation is a Python desktop prototype built with OpenCV, MediaPipe, NumPy, SciPy, and `dtaidistance`.

## Highlights

- Real-time hand tracking through a standard webcam.
- MediaPipe landmark extraction with 21 hand points per frame.
- Spatial normalization to reduce sensitivity to hand position, distance, and hand size.
- Temporal normalization to resample gestures to a consistent frame count.
- Multi-sample registration for stronger matching across natural user variation.
- Per-user statistical threshold calibration based on pairwise DTW distances.
- Finger-state sequence checking to reject similar motions made with the wrong fingers.
- Live authentication with an OpenCV status overlay and access result.
- Standalone comparison mode for inspecting saved registrations.

## How It Works

1. `gesture_capture.py` records five samples of the same gesture for a user.
2. Each frame is converted into a `(21, 3)` landmark array.
3. `utils/normalize.py` normalizes every frame spatially and resamples the whole gesture to 60 frames.
4. `gesture_compare.py` computes DTW distances between registration samples and derives a robust per-user threshold from mean, standard deviation, percentile, and capped max-distance margins.
5. A second per-user finger-state threshold is calibrated from the same samples.
6. `gesture_auth.py` captures a live gesture and compares it against every stored sample.
7. Access is granted only when both the DTW distance and finger-state mismatch pass their thresholds.

## Project Structure

```text
.
|-- idea.txt
|-- README.md
`-- gesture_auth_project/
    |-- gesture_auth.py        # Live login/authentication flow
    |-- gesture_capture.py     # User registration and sample capture
    |-- gesture_compare.py     # DTW comparison, thresholds, template loading
    |-- requirements.txt
    |-- templates/             # Local generated gesture templates
    `-- utils/
        |-- landmarks.py       # MediaPipe landmark helpers
        `-- normalize.py       # Spatial and temporal normalization
```

## Requirements

- Python 3.9 to 3.12
- A working webcam
- Windows, macOS, or Linux with OpenCV webcam support

Python packages:

```text
mediapipe
opencv-python
numpy
scipy
dtaidistance
```

## Setup

From the repository root:

```powershell
cd gesture_auth_project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS or Linux:

```bash
cd gesture_auth_project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Register a gesture:

```bash
python gesture_capture.py
```

During registration:

- Enter a username.
- Show your hand to the webcam.
- Press `R` to record each sample.
- Repeat the same gesture for all five samples.
- Press `Q` to quit.

Authenticate with a saved gesture:

```bash
python gesture_auth.py
```

During authentication:

- Enter an existing username.
- Show your hand to the webcam.
- Press `R` to record the login attempt.
- Press `T` to try again after a result.
- Press `Q` to quit.

Inspect stored template distances:

```bash
python gesture_compare.py
```

## Configuration

The main capture settings live near the top of `gesture_capture.py`:

```python
RECORDING_DURATION_SEC = 3
TARGET_FRAMES = 60
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
```

Matching settings live near the top of `gesture_compare.py`:

```python
NUM_REGISTRATION_SAMPLES = 5
DEFAULT_THRESHOLD = 5.0
MIN_THRESHOLD = 2.0
THRESHOLD_MULTIPLIER = 1.5
THRESHOLD_STD_FACTOR = 2.0
THRESHOLD_PERCENTILE = 90
THRESHOLD_SAFETY_MARGIN = 1.15
THRESHOLD_MAX_MARGIN = 1.25
DEFAULT_FINGER_STATE_THRESHOLD = 0.28
MAX_FINGER_STATE_THRESHOLD = 0.30
```

## Privacy Note

Gesture templates are generated under `gesture_auth_project/templates/`. These files represent biometric-style authentication material and should stay local unless you intentionally want to share test data. The repository keeps the directory but ignores generated template files by default.

## Troubleshooting

- If the webcam does not open, close other apps using the camera and check `CAMERA_INDEX`.
- If no hand is detected, improve lighting and keep your full hand visible in the frame.
- If authentication is too strict, re-register with five consistent samples or tune `THRESHOLD_STD_FACTOR` / `THRESHOLD_SAFETY_MARGIN`.
- If authentication is too loose, lower `THRESHOLD_SAFETY_MARGIN` / `THRESHOLD_MAX_MARGIN` or record a more distinctive gesture.
- If similar movements with different fingers are accepted, check the terminal's `Finger mismatch` value and lower `MAX_FINGER_STATE_THRESHOLD`.

## Roadmap

- Flask API for browser-based registration and authentication.
- React interface with live webcam capture.
- Secure user/session management.
- Admin dashboard for registrations and login history.
- Encrypted template storage for production-style deployments.
