// Runs the mobile Python side inside the real Pyodide build that the app
// ships (dist/vendor/pyodide: Python 3.13, numpy / scipy built for wasm) and
// compares it with the desktop modules (CPython, real OpenCV / dtaidistance).
//
//   cd gesture_auth_project/mobile && python build.py && node tests/pyodide_core.mjs
//
// The desktop reference comes from tests/test_mobile_core.py (--inputs /
// --reference); the checks are its mobile_checks(), unchanged.

import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const mobile = path.dirname(here);
const project = path.dirname(mobile);
const dist = path.join(mobile, "dist");
const python = process.env.PYTHON || "python";

const work = fs.mkdtempSync(path.join(os.tmpdir(), "wl_pyodide_"));
try {
  const inputs = path.join(work, "inputs.npz");
  const ref = path.join(work, "reference.json");
  const test = path.join(here, "test_mobile_core.py");
  const env = { ...process.env, PYTHONIOENCODING: "utf-8" };
  execFileSync(python, [test, "--inputs", inputs], { cwd: project, env, stdio: "pipe" });
  execFileSync(python, [test, "--reference", inputs, path.join(work, "desktop"), ref],
    { cwd: project, env, stdio: "pipe" });

  const { loadPyodide } = await import(pathToUrl(path.join(dist, "vendor", "pyodide", "pyodide.mjs")));
  const t0 = Date.now();
  const py = await loadPyodide({ indexURL: path.join(dist, "vendor", "pyodide") + path.sep });
  await py.loadPackage(["numpy", "scipy"], { messageCallback: () => {} });
  const loadMs = Date.now() - t0;

  // the app's Python files (dist/py) + the test helpers, into Pyodide's FS
  const files = JSON.parse(fs.readFileSync(path.join(dist, "py", "files.json"), "utf8"));
  for (const rel of files) writeFs(py, "/app/" + rel, fs.readFileSync(path.join(dist, "py", rel)));
  for (const name of ["test_wavelock.py"]) writeFs(py, "/app/" + name, fs.readFileSync(path.join(project, name)));
  writeFs(py, "/app/test_mobile_core.py", fs.readFileSync(test));
  writeFs(py, "/work/inputs.npz", fs.readFileSync(inputs));
  writeFs(py, "/work/reference.json", fs.readFileSync(ref));

  const out = py.runPython(`
import sys, json, io, contextlib, time, platform
sys.path[:0] = ["/app/shims", "/app"]
import numpy, scipy
import test_mobile_core as t
failures, lines = [], []
def check(cond, label):
    lines.append(("PASS  " if cond else "FAIL  ") + label)
    if not cond:
        failures.append(label)
t0 = time.time()
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    t.mobile_checks("/work/inputs.npz", json.load(open("/work/reference.json")), "/work/mobile", check)
json.dumps({"lines": lines, "failures": failures, "seconds": round(time.time() - t0, 1),
            "python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__})
`);
  const res = JSON.parse(out);
  for (const l of res.lines) console.log(l);
  console.log(`Pyodide: Python ${res.python}, numpy ${res.numpy}, scipy ${res.scipy}; ` +
    `load ${loadMs} ms, checks ${res.seconds} s (Node, this computer)`);
  console.log(res.failures.length ? `${res.failures.length} FAILED` : "ALL PASSED");
  process.exitCode = res.failures.length ? 1 : 0;
} finally {
  fs.rmSync(work, { recursive: true, force: true });
}

function writeFs(py, file, data) {
  py.FS.mkdirTree(path.posix.dirname(file));
  py.FS.writeFile(file, data);
}

function pathToUrl(p) {
  return new URL("file:///" + p.replace(/\\/g, "/")).href;
}
