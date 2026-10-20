// WaveLock mobile: the Python worker.
//
// Runs Pyodide (Python 3.13 + numpy / scipy, compiled to WebAssembly) off the
// page's thread, loads the desktop modules listed in py/files.json and
// mobile_core.py, and answers {id, method, kwargs} with mobile_core.call().
// Profiles live in /data/templates, an IDBFS mount: the browser's own
// storage for this site on this device. Calls run one at a time; after a
// call that changes profiles, the files are flushed to IndexedDB before the
// answer is sent, so a closed tab never loses a saved profile.

importScripts("../../vendor/pyodide/pyodide.js");

const BASE = new URL("../../", self.location).href;
const WRITES = new Set(["evaluate", "liveness_failed", "reg_save", "face_enroll_save"]);
let py = null;
let call = null;
let queue = Promise.resolve();

function progress(text) {
  self.postMessage({ progress: text });
}

function syncfs(populate) {
  return new Promise((resolve, reject) => py.FS.syncfs(populate, (err) => (err ? reject(err) : resolve())));
}

async function boot({ debug = false, liveness = true }) {
  progress("Starting Python");
  py = await self.loadPyodide({ indexURL: BASE + "vendor/pyodide/" });
  progress("Loading numpy and scipy");
  await py.loadPackage(["numpy", "scipy"], { messageCallback: () => {}, errorCallback: () => {} });
  progress("Loading the WaveLock matcher");
  const files = await (await fetch(BASE + "py/files.json", { cache: "no-cache" })).json();
  await Promise.all(files.map(async (rel) => {
    const text = await (await fetch(BASE + "py/" + rel)).text();
    const path = "/app/" + rel;
    py.FS.mkdirTree(path.slice(0, path.lastIndexOf("/")));
    py.FS.writeFile(path, text);
  }));
  py.FS.mkdirTree("/data");
  py.FS.mount(py.FS.filesystems.IDBFS, {}, "/data");
  await syncfs(true);
  progress("Opening your profiles");
  py.runPython(`
import sys, json
sys.path[:0] = ["/app/shims", "/app"]
import mobile_core
`);
  py.globals.get("mobile_core").init("/data/templates", debug, liveness);
  call = py.runPython("lambda m, k: json.dumps(mobile_core.call(m, json.loads(k)))");
  await syncfs(false);
  const info = JSON.parse(py.runPython(
    "import platform, numpy, scipy; json.dumps({'python': platform.python_version(), " +
    "'numpy': numpy.__version__, 'scipy': scipy.__version__})"));
  return info;
}

async function handle({ id, method, kwargs }) {
  try {
    let result;
    if (method === "__boot__") {
      result = await boot(kwargs || {});
    } else {
      if (!call) throw new Error("Python is not ready yet.");
      result = JSON.parse(call(method, JSON.stringify(kwargs || {})));
      if (WRITES.has(method)) await syncfs(false);
    }
    self.postMessage({ id, result });
  } catch (err) {
    self.postMessage({ id, error: String((err && err.message) || err) });
  }
}

self.onmessage = (e) => {
  queue = queue.then(() => handle(e.data));
};
