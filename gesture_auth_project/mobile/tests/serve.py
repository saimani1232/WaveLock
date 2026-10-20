"""
Serve mobile/dist/ locally with the headers of dist/vercel.json (the same
Content-Security-Policy, caching and permission headers Vercel will send),
so tests see the deployed behaviour. http://localhost counts as a secure
context, so the camera works.

    python tests/serve.py [port]
"""
import functools
import http.server
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.join(os.path.dirname(HERE), "dist")

TYPES = {".mjs": "text/javascript", ".js": "text/javascript", ".wasm": "application/wasm",
         ".webmanifest": "application/manifest+json", ".task": "application/octet-stream",
         ".onnx": "application/octet-stream", ".whl": "application/octet-stream",
         ".zip": "application/zip", ".json": "application/json", ".svg": "image/svg+xml"}


def load_rules():
    config = json.load(open(os.path.join(DIST, "vercel.json")))
    rules = []
    for block in config.get("headers", []):
        # Vercel source patterns used here: "/(.*)", "/sw.js", "/(vendor|models)/(.*)"
        rules.append((re.compile("^" + block["source"] + "$"), block["headers"]))
    return rules


class Handler(http.server.SimpleHTTPRequestHandler):
    rules = []

    def end_headers(self):
        path = self.path.split("?", 1)[0]
        for pattern, headers in self.rules:
            if pattern.match(path):
                for h in headers:
                    self.send_header(h["key"], h["value"])
        super().end_headers()

    def guess_type(self, path):
        ext = os.path.splitext(path)[1].lower()
        return TYPES.get(ext) or super().guess_type(path)

    def log_message(self, fmt, *args):
        pass


def make_server(port=0):
    Handler.rules = load_rules()
    handler = functools.partial(Handler, directory=DIST)
    return http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)


if __name__ == "__main__":
    srv = make_server(int(sys.argv[1]) if len(sys.argv) > 1 else 8780)
    print(f"serving {DIST} on http://localhost:{srv.server_address[1]}/")
    srv.serve_forever()
