"""
HTTP layer of the WaveLock web UI (Python standard library only).

Local-only by design: binds to 127.0.0.1, refuses foreign Host headers
(DNS rebinding), requires JSON + a custom header on every POST (blocks
cross-site form/fetch requests because no CORS headers are ever sent),
refuses cross-site reads of the API and the camera stream, and serves
static files from web/ only.
"""

import json
import mimetypes
import os
import posixpath
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

from web_backend.service import ApiError

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
CLIENT_HEADER = "X-WaveLock-Client"
MAX_BODY_BYTES = 64 * 1024

SECURITY_HEADERS = {
    "Content-Security-Policy": ("default-src 'self'; img-src 'self' data: blob:; style-src 'self'; "
                                "script-src 'self'; connect-src 'self'; object-src 'none'; "
                                "base-uri 'none'; frame-ancestors 'none'; form-action 'none'"),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}

mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("image/svg+xml", ".svg")


def _routes(service):
    """(method, path) -> handler(body, query). Handlers return JSON-able data."""
    s = service
    return {
        ("GET", "/api/system"): lambda b, q: s.system(),
        ("GET", "/api/users"): lambda b, q: s.users(),
        ("GET", "/api/activity"): lambda b, q: s.activity_log(
            limit=_int(q.get("limit", ["50"])[0], 50),
            username=s._username(q["user"][0]) if q.get("user") else None),
        ("GET", "/api/status"): lambda b, q: s.status(),
        ("GET", "/api/cameras"): lambda b, q: s.cameras(),
        ("GET", "/api/benchmark/status"): lambda b, q: s.benchmark_status(),
        ("POST", "/api/camera/start"): lambda b, q: s.camera_start(b.get("camera_index")),
        ("POST", "/api/camera/stop"): lambda b, q: s.camera_stop(),
        ("POST", "/api/auth/begin"): lambda b, q: s.auth_begin(b.get("username", "")),
        ("POST", "/api/auth/record"): lambda b, q: s.auth_record(),
        ("POST", "/api/auth/reset"): lambda b, q: s.auth_reset(),
        ("POST", "/api/auth/end"): lambda b, q: s.auth_end(),
        ("POST", "/api/register/begin"): lambda b, q: s.register_begin(
            b.get("username", ""), bool(b.get("overwrite", False)),
            b.get("pose_method", "legacy")),
        ("POST", "/api/register/face/capture"): lambda b, q: s.register_face_capture(),
        ("POST", "/api/register/face/skip"): lambda b, q: s.register_face_skip(),
        ("POST", "/api/register/record"): lambda b, q: s.register_record(),
        ("POST", "/api/register/cancel"): lambda b, q: s.register_cancel(),
        ("POST", "/api/register/close"): lambda b, q: s.register_close(),
        ("POST", "/api/face/enroll"): lambda b, q: s.face_enroll_begin(b.get("username", "")),
        ("POST", "/api/face/close"): lambda b, q: s.face_close(),
        ("POST", "/api/benchmark/run"): lambda b, q: s.benchmark_run(
            b.get("username", ""), b.get("kind", "")),
    }


def _int(text, default):
    try:
        return max(1, min(500, int(text)))
    except (TypeError, ValueError):
        return default


class WaveLockHandler(BaseHTTPRequestHandler):
    server_version = "WaveLock"
    sys_version = ""
    protocol_version = "HTTP/1.1"

    # ---- plumbing
    def log_message(self, fmt, *args):          # keep the console readable
        if self.server.verbose:
            super().log_message(fmt, *args)

    def _allowed_hosts(self):
        port = self.server.server_address[1]
        return {f"127.0.0.1:{port}", f"localhost:{port}"}

    def _send(self, status, body=b"", content_type="application/json; charset=utf-8",
              extra=None):
        self.send_response(status)
        for k, v in SECURITY_HEADERS.items():
            self.send_header(k, v)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status, data):
        body = json.dumps(data, allow_nan=False).encode("utf-8")
        self._send(status, body, extra={"Cache-Control": "no-store"})

    def _error(self, status, message, code=None):
        self._json(status, {"error": message, "code": code})

    def _guard(self, api):
        """Return an error tuple when the request must be refused."""
        if self.headers.get("Host", "") not in self._allowed_hosts():
            return HTTPStatus.FORBIDDEN, "Unknown host.", "bad_host"
        site = self.headers.get("Sec-Fetch-Site")
        if api and site is not None and site not in ("same-origin", "none"):
            return HTTPStatus.FORBIDDEN, "Cross-site request refused.", "cross_site"
        return None

    # ---- verbs
    def do_GET(self):
        url = urlparse(self.path)
        api = url.path.startswith("/api/")
        refused = self._guard(api)
        if refused:
            return self._error(*refused)
        if url.path == "/api/stream":
            return self._stream()
        if url.path == "/api/frame":
            return self._frame(parse_qs(url.query))
        if api:
            return self._dispatch("GET", url)
        return self._static(url.path)

    def do_HEAD(self):
        self.do_GET()

    def do_POST(self):
        url = urlparse(self.path)
        refused = self._guard(True)
        if refused:
            return self._error(*refused)
        if self.headers.get(CLIENT_HEADER) != "1":
            return self._error(HTTPStatus.FORBIDDEN, "Missing client header.", "no_client_header")
        if not (self.headers.get("Content-Type", "").split(";")[0].strip() == "application/json"):
            return self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Send JSON.", "content_type")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = -1
        if length < 0 or length > MAX_BODY_BYTES:
            return self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Body too large.", "too_large")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except (UnicodeDecodeError, ValueError):
            return self._error(HTTPStatus.BAD_REQUEST, "Invalid JSON.", "bad_json")
        if not isinstance(body, dict):
            return self._error(HTTPStatus.BAD_REQUEST, "JSON object expected.", "bad_json")
        return self._dispatch("POST", url, body)

    def _dispatch(self, method, url, body=None):
        handler = self.server.routes.get((method, url.path))
        if handler is None:
            return self._error(HTTPStatus.NOT_FOUND, "No such endpoint.", "not_found")
        try:
            data = handler(body or {}, parse_qs(url.query))
        except ApiError as e:
            return self._error(e.status, e.message, e.code)
        except Exception as e:                                  # report, keep serving
            self.log_error("Unhandled error on %s: %r", url.path, e)
            return self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"Server error: {e}", "server")
        return self._json(HTTPStatus.OK, {"data": data})

    # ---- single frames (what the UI uses)
    def _frame(self, query):
        """Long-poll: the next JPEG newer than ?after=<id>, or 204 after ~1 s.
        Short requests recover by themselves when the camera restarts and
        never tie up browser connections the way an endless stream can."""
        camera = self.server.service.camera
        try:
            after = int(query.get("after", ["-1"])[0])
        except ValueError:
            after = -1
        frame_id, jpeg = camera.wait_jpeg(after, timeout=1.0)
        if jpeg is None:
            return self._send(HTTPStatus.NO_CONTENT, b"", "text/plain",
                              {"Cache-Control": "no-store", "X-Camera-State": camera.state})
        return self._send(HTTPStatus.OK, jpeg, "image/jpeg",
                          {"Cache-Control": "no-store", "X-Frame-Id": str(frame_id),
                           "X-Camera-State": camera.state})

    # ---- camera stream (MJPEG; kept for other clients)
    def _stream(self):
        camera = self.server.service.camera
        self.send_response(HTTPStatus.OK)
        for k, v in SECURITY_HEADERS.items():
            self.send_header(k, v)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=wavelockframe")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        last = -1
        stopped_for = 0
        try:
            while not self.server.stopping:
                last, jpeg = camera.wait_jpeg(last, timeout=1.0)
                if jpeg is None:
                    # Keep waiting while the camera starts up or is live;
                    # end only when it has been stopped for a few seconds.
                    stopped_for = 0 if camera.state in ("starting", "running") else stopped_for + 1
                    if stopped_for >= 3:
                        break
                    continue
                stopped_for = 0
                self.wfile.write(b"--wavelockframe\r\nContent-Type: image/jpeg\r\n"
                                 + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                                 + jpeg + b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass

    # ---- static files
    def _static(self, path):
        path = unquote(path)
        if path in ("", "/"):
            path = "/index.html"
        rel = posixpath.normpath(path).lstrip("/")
        if rel.startswith("..") or "\\" in rel or "\x00" in rel:
            return self._error(HTTPStatus.NOT_FOUND, "Not found.", "not_found")
        full = os.path.realpath(os.path.join(WEB_DIR, *rel.split("/")))
        if not full.startswith(os.path.realpath(WEB_DIR) + os.sep) or not os.path.isfile(full):
            return self._error(HTTPStatus.NOT_FOUND, "Not found.", "not_found")
        ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/json", "image/svg+xml"):
            ctype += "; charset=utf-8"
        with open(full, "rb") as f:
            body = f.read()
        self._send(HTTPStatus.OK, body, ctype, {"Cache-Control": "no-cache"})


class WaveLockServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, service, host="127.0.0.1", port=8765, verbose=False):
        if host not in ("127.0.0.1", "localhost"):
            raise ValueError("The WaveLock UI only listens on 127.0.0.1.")
        self.service = service
        self.routes = _routes(service)
        self.verbose = verbose
        self.stopping = False
        super().__init__((host, port), WaveLockHandler)

    def handle_error(self, request, client_address):
        # Browsers drop idle keep-alive and stream connections all the time.
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)

    def shutdown(self):
        self.stopping = True
        super().shutdown()
