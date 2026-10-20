# WaveLock mobile (on-device web app)

Live: https://wavelock-mobile.vercel.app (Vercel project `wavelock-mobile`, static files only).

The phone runs everything itself: camera, hand and face models, and the desktop
matcher. Nothing biometric is uploaded; Vercel only serves the files.

| Part | How it runs on the phone |
|---|---|
| Decisions (gesture gates, calibration, liveness rules, attempt limit, aging, face decision, registration report) | The desktop Python modules, unchanged, in Pyodide 0.29.5 (Python 3.13, numpy 2.2.5, scipy 1.14.1) in a web worker; glue in `py/mobile_core.py` |
| `dtaidistance` (C extension) | `py/shims/dtaidistance`: NumPy DTW, identical distances |
| `cv2`, `mediapipe` imports | `py/shims`: placeholders; any use raises |
| Hand landmarks | MediaPipe Tasks Vision 0.10.35 hand landmarker (web build), one hand, detection 0.7 / tracking 0.5 |
| Face detection / identity | The desktop's YuNet and SFace models in ONNX Runtime Web 1.29.0; pre/post-processing in `src/js/local/face_core.js` |
| Profiles | IndexedDB of this site on this device (Pyodide IDBFS), not encrypted |
| Screens | The web UI pages (`web/`) unchanged, with a local `api.js`; own Settings and Security pages |

## Build, test, deploy

```bash
cd gesture_auth_project/mobile
npm install                     # pinned packages (package.json)
python build.py                 # -> dist/ (115 MB); downloads checked against SHA-256
python tests/test_mobile_core.py            # mobile Python side vs desktop (27 checks)
node tests/pyodide_core.mjs                 # same checks inside Pyodide
node tests/face_parity.mjs .cache/testdata/astronaut.png   # face pipeline vs OpenCV
python tests/e2e_mobile.py                  # headless Chrome, phone size, production headers
cd dist && vercel deploy --prod             # deploys dist/ only
```

`tests/face_parity.mjs` needs a test image with a face; the one used is the public-domain NASA
astronaut portrait from scikit-image v0.24.0 (`skimage/data/astronaut.png`), kept in `.cache/`.

## Verified (2026-10-20)

- Python side: identical to the desktop within 1e-9 for the quality gate, registration report,
  saved config, sign-in payloads with details, template aging (CPython and Pyodide).
- DTW stand-in: 0.0 difference to `dtaidistance` (random data and all real template pairs).
- Face: identical YuNet detections (0.000 px), SFace embedding cosine >= 0.99994 vs OpenCV.
- End to end (synthetic hand input, real camera loop with Chrome's fake camera): boot ~11 s,
  registration, liveness, granted / denied / liveness failed, data kept after reload, no console
  errors or CSP violations; also run against the deployed site.

## Limits

- Profiles are stored unencrypted in the browser; protected by the phone lock and per-site isolation.
- Anyone holding the unlocked phone can switch on developer mode or switch off the liveness check.
- Phone profiles are separate from desktop profiles (different hand model build, camera, frame rate).
  The face check uses six frames at evenly spaced times (the desktop samples six frames by index).
- Not measured with real users or real attacks on phones; the MEASURED results are desktop ones.
- First start downloads about 70 MB (cached afterwards; SFace only when a face is used).
