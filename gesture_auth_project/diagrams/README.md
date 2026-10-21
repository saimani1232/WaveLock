# WaveLock diagrams

Hand-drawn style diagrams of the whole project, made on 2026-10-21. Every number and rule in
them was checked against the code on that date (`gesture_compare.py`, `gesture_auth.py`,
`gesture_capture.py`, `utils/`, `mobile/`). If the code changes, the code wins.

| # | File | Read it to understand |
|---|---|---|
| 01 | `01_system_overview` | The big picture: desktop, web and mobile front ends, the camera models, the shared Python core, where profiles live |
| 02 | `02_registration_flow` | How a profile is made: owner check, face enrolment, 5 samples, still-frame trim, quality gate, threshold calibration, uniqueness check, atomic save |
| 03 | `03_sign_in_flow` | One sign-in attempt from start to GRANTED / DENIED: lockout, liveness challenge, ready-to-record, recording, face check |
| 04 | `04_decision_logic` | The exact decision rule: gates G1-G4, wrist path, fused score, hand shape, movement, soft consensus, face |
| 05 | `05_after_the_decision` | What happens afterwards: attempt counter, lockout times, owner window, template aging with the drift anchor |
| 06 | `06_profile_storage` | What is inside `templates/<user>/` and the other folders; encryption and privacy |
| 07 | `07_mobile_app` | How the phone app runs everything on the device (Pyodide, MediaPipe web, ONNX Runtime Web, IndexedDB) |
| 08 | `08_evidence_and_tests` | What is MEASURED, what is SIMULATED, what was never tested, and the automatic tests |

Suggested order for someone new: 01, 03, 04, 02, 05, 06, 07, 08.

## Files

- `*.excalidraw`: the editable diagrams. Open them at https://excalidraw.com (menu, Open) or with
  the Excalidraw extension in VS Code. Opening a file at excalidraw.com reads it in the browser;
  nothing is uploaded unless you use its share or collaboration features.
- `png/*.png`: the same diagrams as images, for quick viewing or for the report.
- `src/*.json`: the element lists the diagrams were built from (Excalidraw "skeleton" format).

The diagrams contain no biometric data: only file names, rules and thresholds.
