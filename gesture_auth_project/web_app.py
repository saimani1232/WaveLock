"""
WaveLock Web UI — local browser front end for registration and sign-in.

Usage:
    python web_app.py [--port 8765] [--camera N] [--debug] [--no-browser]

Opens http://127.0.0.1:8765 in the default browser. The server only listens
on this computer. --debug (developer mode) shows per-gate scores, cosine
values and failure reasons, and enables the SIMULATED benchmark runner; keep
it off for real sign-ins because those numbers help an observer tune an
imitation. The desktop programs (gesture_capture.py, gesture_auth.py,
enroll_face.py) keep working and share the same templates/ folder.

Architecture and API: web/README.md
"""

import argparse
import sys
import threading
import webbrowser

from web_backend.camera import CameraService
from web_backend.service import WaveLockService
from web_backend.server import WaveLockServer


def main(argv=None):
    parser = argparse.ArgumentParser(description="WaveLock Web UI")
    parser.add_argument("--port", type=int, default=8765, help="Port on 127.0.0.1 (default 8765)")
    parser.add_argument("--camera", "--cam", "-c", dest="camera", type=int, default=None,
                        help="Camera index (default: auto-detect)")
    parser.add_argument("--debug", action="store_true",
                        help="Developer mode: show scores and reasons (not for real sign-ins)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser window")
    parser.add_argument("--verbose", action="store_true", help="Log every HTTP request")
    args = parser.parse_args(argv)

    # The backend prints progress with symbols such as ✓; never let a
    # console that cannot encode them stop a registration thread.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    camera = CameraService()
    service = WaveLockService(camera, debug=args.debug, default_camera=args.camera)
    try:
        server = WaveLockServer(service, "127.0.0.1", args.port, verbose=args.verbose)
    except OSError as e:
        print(f"  ERROR: cannot listen on 127.0.0.1:{args.port} ({e}).")
        print("  Another WaveLock window may be open; use --port to pick another port.")
        return 1

    url = f"http://127.0.0.1:{args.port}/"
    print("=" * 60)
    print("   WAVELOCK -- Web UI")
    print("=" * 60)
    print(f"  Open: {url}")
    print(f"  Mode: {'DEVELOPER (scores visible)' if args.debug else 'normal (scores hidden)'}")
    print("  Stop: Ctrl+C")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        print("\n  Stopping...")
    finally:
        server.stopping = True
        service.shutdown()
        server.server_close()
        print("  Camera released. Goodbye!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
