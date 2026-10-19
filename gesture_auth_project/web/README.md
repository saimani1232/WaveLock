# WaveLock Web UI: architecture and integration plan

Start: `python web_app.py` (opens http://127.0.0.1:8765). Options: `--port N`, `--camera N`, `--debug` (developer mode: per-gate scores), `--no-browser`.

The desktop programs (`gesture_capture.py`, `gesture_auth.py`, `enroll_face.py`) keep working unchanged. The web UI is a second front end on the same backend code and the same `templates/` folder. No existing `.py` file was modified for it.

## 1. Architecture

```
 Browser (single-page app, web/)                Python process (web_app.py)
 ┌─────────────────────────────┐   HTTP JSON    ┌───────────────────────────────────────────┐
 │ views: Dashboard, Sign in,  │ ─────────────► │ web_backend/server.py   routes, security   │
 │ Register, Users, Security,  │ ◄───────────── │ web_backend/service.py  workflows          │
 │ Settings                    │   /api/status  │   ├ auth  -> gesture_auth.evaluate_attempt │
 │ camera panel <img> ◄────────┼── JPEG frames ─│   │         attempt_limit, template aging  │
 │ theme toggle (light/dark)   │                │   ├ register -> collect_gesture_samples,   │
 └─────────────────────────────┘   /api/frame   │   │   validate_sample_quality, cohort,     │
                                                │   │   save_registration                    │
                                                │   └ face enrol -> save_face_embedding      │
                                                │ web_backend/camera.py  one camera thread:  │
                                                │   OpenCV capture -> MediaPipe Hands ->     │
                                                │   YuNet face box -> JPEG; recordings       │
                                                └───────────────────────────────────────────┘
```

Why the backend owns the camera: hand landmarks must come from the same MediaPipe Python build (0.10.21) and the same capture settings (640x480, mirrored) as the stored templates. Running MediaPipe in the browser could give slightly different landmarks and silently change the match scores.

## 2. Integration rules

- Workflows call the existing functions; decision logic is not re-implemented: `evaluate_attempt` (full multimodal decision), `collect_gesture_samples` + `validate_sample_quality` (quality gate and restart policy), `compute_*_threshold_details`, `extract_finger_transitions`, `validate_gesture_uniqueness`, `save_registration`, `save_face_embedding`, `set_face_enrolled_flag`, `update_templates_if_high_confidence`, `record_attempt` / `seconds_locked`.
- Recording follows the CLI: starts only with a hand in view, lasts `RECORDING_DURATION_SEC`, needs `MIN_RECORDED_FRAMES` hand frames, keeps raw landmarks + clean frames (face check).
- Face capture follows the CLI: 5 frames with a detected face, mean embedding, L2-normalised; saved together with the gesture profile at the end of registration.
- One activity at a time (sign-in, registration or face enrolment); the camera is shared.

## 3. Security model

- Server binds to 127.0.0.1 only; requests with another Host header are refused (DNS rebinding).
- Every state-changing request is POST with `Content-Type: application/json` and header `X-WaveLock-Client: 1`; the server sends no CORS headers, so other web sites cannot trigger actions. Cross-site loads of the camera stream are refused (`Sec-Fetch-Site`).
- Strict Content-Security-Policy (`'self'` only), `X-Content-Type-Options`, `Referrer-Policy: no-referrer`, `Cache-Control: no-store` on API responses. Static files are served from `web/` only.
- Normal mode returns only GRANTED / DENIED, attempts left and lockout time. Per-gate scores, cosines, thresholds and failure reasons are returned only when the server was started with `--debug`. The browser cannot switch this on.
- Attempt limit and lockout are the same as the CLI (`utils/attempt_limit.py`), shared across both front ends.
- Known limits (same as the CLI, not changed here): re-registering an existing username and re-enrolling a face require only a confirmation, not the owner's login. Anyone who can use the computer can use the UI.

## 4. Pages and workflows

| Page | Purpose | Backend |
|---|---|---|
| Dashboard | Users, face coverage, today's attempts, active lockouts, camera and model status, recent activity, quick actions | `/api/system`, `/api/users`, `/api/activity` |
| Sign in | Username, then live camera with hand/face indicators, record (button, R or Space), progress ring, decision with attempts left / lockout countdown; developer mode adds the four stage cards (face, gesture gates, hand biometrics, fusion) | `/api/auth/*`, `/api/status`, `/api/stream` |
| Register | Wizard: Account (username, overwrite confirmation, pose method) → Face (capture or skip) → Gesture (5 samples, accepted / rejected per sample, restarts) → Review (calibrated thresholds, consistency, warnings) | `/api/register/*` |
| Users | Profile cards: samples, face, pose method, lock state; actions: sign in, enrol / replace face, re-register | `/api/users`, `/api/face/enroll` |
| Security | Protections in force, per-user attempt log; developer mode: SIMULATED benchmark runner | `/api/activity`, `/api/benchmark/*` |
| Settings | Theme, camera selection, developer-mode status, version info | `/api/cameras`, `/api/system` |

Feedback at every step: hand / face chips on the camera, countdown and progress ring while recording, "processing" state while the decision runs, toast messages for errors (no hand, too few frames, locked, busy), and a result panel.

## 5. Files

```
web_app.py                 entry point
web_backend/camera.py      CameraService (+ injectable source, tracker, face ops for tests)
web_backend/service.py     WaveLockService: workflows, JSON-safe results
web_backend/server.py      HTTP routes, MJPEG stream, static files, security checks
web/index.html, web/css/app.css, web/js/*.js
test_web_backend.py        API tests with a simulated camera (no webcam needed)
```

## 6. Testing

1. `python test_web_backend.py`: HTTP-level tests with a simulated camera that replays synthetic landmark sequences (security checks, registration, sign-in granted / denied, lockout, hidden scores, face enrolment).
2. `python test_wavelock.py`: the existing backend self-tests must still pass.
3. `node --check web/js/*.js`, then a headless Chrome run of the main workflows with screenshots in light and dark mode.

## 7. Camera preview and troubleshooting

The preview pulls one JPEG at a time from `/api/frame?after=<id>` (long-poll, about 1 s). Short requests recover by themselves when the camera restarts and never tie up the browser's connection pool. `/api/stream` (MJPEG) is kept for other clients.

Fixed 2026-10-14: with a real webcam the preview stayed black although the chips showed "Live" and "Face in view". The camera was reported as running before its first frame existed (MediaPipe / YuNet warm up on the first frame, about 1 s), and the stream gave up at once because the frame wait returned immediately when no frame existed yet. Now the camera is "running" only after its first frame, the wait really waits, and the preview uses `/api/frame`. Checked with the real webcam: picture after 2.5 s (cold start) and 1.5 s (restart), about 15 frames/s. `test_web_backend.py` now simulates a slow first frame so this cannot come back unnoticed.

| Symptom | What to do |
|---|---|
| "Could not open camera N" | Close other apps that use the webcam (Teams, Zoom, Camera, the desktop WaveLock programs), or pick another camera in Settings. |
| Preview says "The camera is still starting…" for long | The webcam is slow to deliver its first frame; wait, or unplug and reconnect it. |
| "Lost connection to the WaveLock server" | `web_app.py` was stopped; start it again; the page reconnects by itself. |
| "A registration ... is in progress" | Open Register (it resumes the running registration) and continue or cancel it. |
| Low frame rate (about 15 fps) | Normal on a laptop CPU: every frame runs MediaPipe Hands; the desktop programs do the same work. |

## 8. Visual design (2026-10-18)

- Brand: `js/brand.js` draws the mark from the 21 MediaPipe hand landmarks with a keyhole in the palm; `assets/logo.svg` is generated from the same geometry. No `style=""` attributes anywhere: the CSP (`style-src 'self'`) blocks them, so animation delays are set as CSS variables from script.
- Camera panel (`js/camera.js`): picture hidden until a real frame; scan ring (SVG rect, `pathLength=100`) fills during recording and turns green / red; coaching hints from `hand_visible`, `face_in_view` and `brightness` (below 40 = low light); status chips under the picture; result banner inside it.
- Protected mode never shows per-stage verdicts: stages read "Checked", only the decision is coloured.
- Colour tokens meet WCAG AA (4.5:1) for text in both themes; all motion stops under `prefers-reduced-motion`.

## 9. Keyboard shortcuts

Single keys, shown as small badges on their buttons; **?** lists the ones available on the current screen.

| Screen | Keys |
|---|---|
| Sign in, ready | **R** or **Space** record |
| Sign in, after a result | **T** try again / sign in again, **D** done |
| Sign in, ready or after a result | **U** switch user |
| Register, face step | **C** capture face, **S** skip |
| Register, gesture step | **R** or **Space** record the next sample |
| Register, in progress | **Esc** cancel (asks first) |
| Register, finished | **S** sign in now, **N** register another |
| Anywhere | **?** shortcut list, **Esc** close a dialog |

Rules (`useShortcuts` in `js/ui.js`): a key only works when its button is usable; nothing fires while typing in a text box, with Ctrl / Alt / Cmd held, while a dialog is open, or on key repeat; Space on a focused button is left to the browser.
