"""
Build the WaveLock mobile app (static files) into mobile/dist/.

    cd gesture_auth_project/mobile
    npm install            # pinned: pyodide, @mediapipe/tasks-vision, onnxruntime-web
    python build.py        # -> dist/  (deploy this folder; nothing else)

What goes into dist/:
  - the app (mobile/src) and the web UI files shared with web_app.py (web/),
  - the desktop Python modules the phone runs in Pyodide (py/) - the same
    files as the desktop programs, plus mobile/py (glue and stand-ins),
  - Pyodide with numpy / scipy, MediaPipe Tasks Vision (web build) and
    ONNX Runtime Web (vendor/), and the models (models/),
  - sw.js (offline cache, versioned by the content) and vercel.json (headers).

Downloads (Pyodide packages, the MediaPipe hand model) are checked against
pinned SHA-256 hashes and cached in mobile/.cache/. No profile, recording,
note or document is ever copied: only the paths listed below, and
check_dist() refuses to finish if anything else appears.
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

# Files of the web UI (web/) used unchanged by the mobile app.
WEB_FILES = {
    "css/app.css": "css/app.css",
    "assets/logo.svg": "assets/logo.svg",
    "js/app.js": "js/app.js",
    "js/ui.js": "js/ui.js",
    "js/brand.js": "js/brand.js",
    "js/theme-init.js": "js/theme-init.js",
    "js/camera.js": "js/camera_web.js",          # extended by mobile/src/js/camera.js
    "js/views/dashboard.js": "js/views/dashboard.js",
    "js/views/signin.js": "js/views/signin.js",
    "js/views/register.js": "js/views/register.js",
    "js/views/users.js": "js/views/users.js",
}

VENDOR_FILES = {
    "@mediapipe/tasks-vision/vision_bundle.mjs": "mediapipe/vision_bundle.mjs",
    "@mediapipe/tasks-vision/wasm/vision_wasm_internal.js": "mediapipe/wasm/vision_wasm_internal.js",
    "@mediapipe/tasks-vision/wasm/vision_wasm_internal.wasm": "mediapipe/wasm/vision_wasm_internal.wasm",
    "@mediapipe/tasks-vision/wasm/vision_wasm_nosimd_internal.js": "mediapipe/wasm/vision_wasm_nosimd_internal.js",
    "@mediapipe/tasks-vision/wasm/vision_wasm_nosimd_internal.wasm": "mediapipe/wasm/vision_wasm_nosimd_internal.wasm",
    "onnxruntime-web/dist/ort.wasm.min.mjs": "ort/ort.wasm.min.mjs",
    "onnxruntime-web/dist/ort-wasm-simd-threaded.mjs": "ort/ort-wasm-simd-threaded.mjs",
    "onnxruntime-web/dist/ort-wasm-simd-threaded.wasm": "ort/ort-wasm-simd-threaded.wasm",
}

# Models: the desktop's face models (gesture_auth_project/models) and the
# official MediaPipe hand landmarker, all pinned by SHA-256.
HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                  "hand_landmarker/float16/1/hand_landmarker.task")
MODELS = {
    "hand_landmarker.task": ("url", HAND_MODEL_URL,
                             "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"),
    "face_detection_yunet_2023mar.onnx": ("file", "models/face_detection_yunet_2023mar.onnx",
                                          "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
    "face_recognition_sface_2021dec.onnx": ("file", "models/face_recognition_sface_2021dec.onnx",
                                            "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"),
}

# Only this site's own files and WebAssembly; no inline scripts or styles.
CSP = ("default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; worker-src 'self'; "
       "connect-src 'self'; img-src 'self' data: blob:; style-src 'self'; "
       "media-src 'self' blob: mediastream:; object-src 'none'; base-uri 'self'; "
       "frame-ancestors 'none'; form-action 'self'")


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


# ---------------------------------------------------------------- app, vendor, models
def app_files():
    """The mobile shell, the shared web UI files and the mobile overrides."""
    web = os.path.join(PROJECT, "web")
    for src, dst in WEB_FILES.items():
        copy(os.path.join(web, src), os.path.join(DIST, dst))
    src_root = os.path.join(HERE, "src")
    for root, _, names in os.walk(src_root):
        for n in names:
            if n == "sw.template.js":
                continue
            rel = os.path.relpath(os.path.join(root, n), src_root)
            copy(os.path.join(root, n), os.path.join(DIST, rel))


def vendor_files():
    for src, dst in VENDOR_FILES.items():
        path = os.path.join(NODE, *src.split("/"))
        if not os.path.exists(path):
            raise SystemExit(f"{src} missing: run `npm install` in mobile/ first.")
        copy(path, os.path.join(DIST, "vendor", dst))


def model_files():
    for name, (kind, where, digest) in MODELS.items():
        if kind == "url":
            src = fetch(where, os.path.join(CACHE, "models", name), digest)
        else:
            src = os.path.join(PROJECT, where)
            if sha256(src) != digest:
                raise SystemExit(f"{where}: SHA-256 differs from the pinned model")
        copy(src, os.path.join(DIST, "models", name))


def service_worker_and_config():
    """sw.js with a version derived from every built file, and vercel.json."""
    h = hashlib.sha256()
    shell = []
    for root, dirs, names in os.walk(DIST):
        dirs[:] = sorted(x for x in dirs if x != ".vercel")     # not part of the app
        for n in sorted(names):
            path = os.path.join(root, n)
            rel = os.path.relpath(path, DIST).replace(os.sep, "/")
            if rel in ("sw.js", "vercel.json"):
                continue
            h.update(rel.encode())
            h.update(sha256(path).encode())
            if not rel.startswith(("vendor/", "models/")):
                shell.append(rel)
    version = h.hexdigest()[:12]
    shell = ["./"] + sorted(shell)
    tpl = open(os.path.join(HERE, "src", "sw.template.js"), encoding="utf-8").read()
    with open(os.path.join(DIST, "sw.js"), "w", encoding="utf-8") as f:
        f.write(tpl.replace("__VERSION__", version).replace("__SHELL__", json.dumps(shell, indent=1)))
    config = {
        "headers": [
            {"source": "/(.*)", "headers": [
                {"key": "Content-Security-Policy", "value": CSP},
                {"key": "X-Content-Type-Options", "value": "nosniff"},
                {"key": "Referrer-Policy", "value": "no-referrer"},
                {"key": "Permissions-Policy", "value": "camera=(self), microphone=(), geolocation=()"},
                {"key": "Cross-Origin-Opener-Policy", "value": "same-origin"},
            ]},
            {"source": "/sw.js", "headers": [{"key": "Cache-Control", "value": "no-cache"}]},
            {"source": "/(vendor|models)/(.*)", "headers": [
                {"key": "Cache-Control", "value": "public, max-age=86400"}]},
        ],
    }
    with open(os.path.join(DIST, "vercel.json"), "w") as f:
        json.dump(config, f, indent=2)
    return version, len(shell)


def check_dist():
    """Nothing but the listed files: no profiles, recordings, notes or documents."""
    bad = []
    for root, _, names in os.walk(DIST):
        for n in names:
            rel = os.path.relpath(os.path.join(root, n), DIST).replace(os.sep, "/")
            if (rel.startswith(("templates", "notes", "docs", "report"))
                    or n.endswith((".npy", ".npz", ".docx", ".pdf", ".log"))
                    or n in ("config.json", "CHANGES_SINCE_SEMESTER_1.md")):
                bad.append(rel)
    if bad:
        raise SystemExit(f"refusing to build: unexpected files in dist/: {bad[:5]}")


def main(argv):
    only = set(argv) or {"clean", "python", "pyodide", "app", "vendor", "models"}
    if "clean" in only and os.path.isdir(DIST):
        for entry in os.listdir(DIST):
            if entry == ".vercel":              # keep the Vercel project link
                continue
            path = os.path.join(DIST, entry)
            shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
    os.makedirs(DIST, exist_ok=True)
    if "python" in only:
        files = python_sources(os.path.join(DIST, "py"))
        print(f"python: {len(files)} files")
    if "pyodide" in only:
        pkgs = pyodide_files(os.path.join(DIST, "vendor", "pyodide"))
        print(f"pyodide {PYODIDE_VERSION}: core + {', '.join(sorted(pkgs))}")
    if "app" in only:
        app_files()
        print("app: shell, web UI files, mobile overrides")
    if "vendor" in only:
        vendor_files()
        print("vendor: MediaPipe Tasks Vision, ONNX Runtime Web")
    if "models" in only:
        model_files()
        print("models: hand landmarker, YuNet, SFace (SHA-256 checked)")
    check_dist()
    version, n = service_worker_and_config()
    total = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(DIST) for f in fs)
    print(f"dist/: version {version}, {n} shell files, {total / 1e6:.1f} MB in total")


if __name__ == "__main__":
    main(sys.argv[1:])
