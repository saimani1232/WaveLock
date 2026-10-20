"""
Build the WaveLock mobile app (static files) into mobile/dist/.

    cd gesture_auth_project/mobile
    npm install            # pinned: pyodide, @mediapipe/tasks-vision, onnxruntime-web
    python build.py        # -> dist/  (deploy this folder; nothing else)

What goes into dist/:
  - the app (mobile/src) and the web UI views shared with web_app.py (web/),
  - the desktop Python modules the phone runs in Pyodide (py/) - the same
    files as the desktop programs, plus mobile/py (glue and stand-ins),
  - Pyodide with numpy / scipy, MediaPipe Tasks Vision (web build) and
    ONNX Runtime Web (vendor/), and the models (models/).

Downloads (Pyodide packages, the MediaPipe hand model) are checked against
pinned SHA-256 hashes and cached in mobile/.cache/. No profile, recording,
note or document is ever copied: only the paths listed below.
"""

import hashlib
import json
import os
import shutil
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
NODE = os.path.join(HERE, "node_modules")
CACHE = os.path.join(HERE, ".cache")
DIST = os.path.join(HERE, "dist")

PYODIDE_VERSION = "0.29.5"
PYODIDE_CDN = f"https://cdn.jsdelivr.net/pyodide/v{PYODIDE_VERSION}/full/"
PYODIDE_PACKAGES = ["numpy", "scipy"]

# Desktop Python modules the phone runs (decision logic only). Paths are
# relative to gesture_auth_project/; nothing else from the project is copied.
PY_MODULES = [
    "gesture_compare.py", "gesture_auth.py", "gesture_capture.py", "cohort_library.py",
    "utils/__init__.py", "utils/anthropometrics.py", "utils/attempt_limit.py",
    "utils/camera_utils.py", "utils/face_auth.py", "utils/kinematics.py",
    "utils/landmarks.py", "utils/liveness.py", "utils/normalize.py", "utils/secure_store.py",
    "web_backend/__init__.py", "web_backend/service.py", "web_backend/camera.py",
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url, dest, expected_sha256):
    """Download once into the cache; refuse a file whose hash differs."""
    if os.path.exists(dest) and sha256(dest) == expected_sha256:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    print(f"  downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    got = sha256(tmp)
    if got != expected_sha256:
        os.remove(tmp)
        raise SystemExit(f"SHA-256 mismatch for {url}: {got} != {expected_sha256}")
    os.replace(tmp, dest)
    return dest


def copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


# ---------------------------------------------------------------- Pyodide
def pyodide_files(out_dir):
    """Pyodide core (npm package) + numpy / scipy and their dependencies."""
    core = os.path.join(NODE, "pyodide")
    if not os.path.isdir(core):
        raise SystemExit("node_modules/pyodide missing: run `npm install` in mobile/ first.")
    for name in ("pyodide.js", "pyodide.mjs", "pyodide.asm.js", "pyodide.asm.wasm",
                 "python_stdlib.zip", "pyodide-lock.json", "package.json"):
        copy(os.path.join(core, name), os.path.join(out_dir, name))
    lock = json.load(open(os.path.join(core, "pyodide-lock.json")))
    if lock["info"].get("version") not in (None, PYODIDE_VERSION):
        raise SystemExit(f"pyodide-lock.json is for {lock['info'].get('version')}, expected {PYODIDE_VERSION}")
    needed, todo = [], list(PYODIDE_PACKAGES)
    while todo:
        name = todo.pop()
        if name in needed:
            continue
        needed.append(name)
        todo.extend(lock["packages"][name].get("depends", []))
    for name in needed:
        pkg = lock["packages"][name]
        cached = fetch(PYODIDE_CDN + pkg["file_name"],
                       os.path.join(CACHE, "pyodide", pkg["file_name"]), pkg["sha256"])
        copy(cached, os.path.join(out_dir, pkg["file_name"]))
    return needed


# ---------------------------------------------------------------- Python sources
def python_sources(out_dir):
    files = []
    for rel in PY_MODULES:
        copy(os.path.join(PROJECT, rel), os.path.join(out_dir, rel))
        files.append(rel)
    mobile_py = os.path.join(HERE, "py")
    for root, _, names in os.walk(mobile_py):
        for n in names:
            if n.endswith(".py"):
                rel = os.path.relpath(os.path.join(root, n), mobile_py).replace(os.sep, "/")
                copy(os.path.join(root, n), os.path.join(out_dir, rel))
                files.append(rel)
    with open(os.path.join(out_dir, "files.json"), "w") as f:
        json.dump(sorted(files), f, indent=1)
    return files


def main(argv):
    only = set(argv) or {"python", "pyodide"}
    os.makedirs(DIST, exist_ok=True)
    if "python" in only:
        files = python_sources(os.path.join(DIST, "py"))
        print(f"python: {len(files)} files")
    if "pyodide" in only:
        pkgs = pyodide_files(os.path.join(DIST, "vendor", "pyodide"))
        print(f"pyodide {PYODIDE_VERSION}: core + {', '.join(sorted(pkgs))}")


if __name__ == "__main__":
    main(sys.argv[1:])
