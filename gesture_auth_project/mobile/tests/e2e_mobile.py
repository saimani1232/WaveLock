"""
End-to-end test of the mobile app in headless Chrome (phone size), served
with the production headers (tests/serve.py + dist/vercel.json).

Chrome's fake camera shows a test pattern, so the test replaces the OUTPUT of
the hand landmarker with synthetic recordings (test_wavelock.synth_raw, the
desktop tests' gestures), from the test side: it imports the app's own
api.js module in the page and swaps service.camera.landmarker. Nothing in
the shipped app has a test hook. Everything else is real: camera loop and
recording timing, liveness challenge, Python (Pyodide) decisions, IndexedDB
storage, and the pages, driven with keys and clicks.

    cd gesture_auth_project/mobile && python build.py && python tests/e2e_mobile.py [shots_dir]

Against a deployed copy instead of dist/: set WAVELOCK_URL (e.g. the Vercel URL).
"""
import json
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MOBILE = os.path.dirname(HERE)
PROJECT = os.path.dirname(MOBILE)
sys.path[:0] = [HERE, PROJECT]

import numpy as np                      # noqa: E402
from selenium import webdriver          # noqa: E402
from selenium.webdriver.common.by import By    # noqa: E402

import test_wavelock as tw              # noqa: E402
from serve import make_server           # noqa: E402

failures = []


def check(cond, label):
    print(("PASS  " if cond else "FAIL  ") + label, flush=True)
    if not cond:
        failures.append(label)


def gestures():
    """Synthetic recordings (raw landmarks, mirrored convention) + finger poses."""
    def window(r, a, b):
        r = np.asarray(r, float)
        return np.concatenate([np.repeat(r[:1], a, axis=0), r, np.repeat(r[-1:], b, axis=0)])
    genuine = [window(r, 12, 12) for r in tw.genuine_set()]
    seqs = {f"enrol{i}": g.tolist() for i, g in enumerate(genuine)}
    seqs["genuine"] = window(tw.genuine_set(1, seed=31)[0], 12, 12).tolist()
    seqs["wrong"] = window(tw.synth_raw(order=("ring", "pinky", "index", "middle"), seed=7), 12, 12).tolist()
    poses = {str(n): tw.finger_pose(n).tolist() for n in range(0, 5)}
    return {"seqs": seqs, "poses": poses}


INSTALL = """
const done = arguments[arguments.length - 1];
const data = arguments[0];
import('/js/api.js').then(({ service }) => {
  const cam = service.camera;
  window.__wl = { mode: 'enrol0', answer: true, hide: false, data, service };
  const fake = { detectForVideo() {
    const st = window.__wl;
    if (st.hide) return { landmarks: [] };
    const seq = st.data.seqs[st.mode];
    const rec = cam.recording, a = service.auth;
    let frame;
    if (rec && rec.started !== null) {
      frame = seq[Math.min(seq.length - 1, Math.floor(rec.progress() * (seq.length - 1)))];
    } else if (a && a.phase === 'challenge' && a.challenge && a.challenge.state === 'running') {
      const step = a.challenge.steps[a.challenge.index];
      frame = st.answer ? st.data.poses[String(step.count)] : st.data.poses['0'];
    } else {
      frame = seq[0];
    }
    return { landmarks: [frame.map(([x, y, z]) => ({ x, y, z }))] };
  } };
  cam._createLandmarker = async () => fake;
  if (cam.landmarker) cam.landmarker = fake;
  done(true);
}).catch((e) => done('ERR ' + e.message));
"""


def main():
    shots = sys.argv[1] if len(sys.argv) > 1 else None
    srv = None
    base = os.environ.get("WAVELOCK_URL")
    if base:
        base = base.rstrip("/") + "/"
    else:
        srv = make_server(0)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://localhost:{srv.server_address[1]}/"
    print(f"testing {base}", flush=True)
    o = webdriver.ChromeOptions()
    for a in ("--headless=new", "--window-size=390,844", "--use-fake-ui-for-media-stream",
              "--use-fake-device-for-media-stream", "--autoplay-policy=no-user-gesture-required"):
        o.add_argument(a)
    o.set_capability("goog:loggingPrefs", {"browser": "ALL"})
    d = webdriver.Chrome(options=o)
    d.set_script_timeout(60)

    def shot(name):
        if shots:
            time.sleep(0.4)
            d.save_screenshot(os.path.join(shots, f"mobile_{name}.png"))

    def body():
        return d.find_element(By.TAG_NAME, "body").text

    def wait_text(text, timeout=60):
        end = time.time() + timeout
        while time.time() < end:
            if text in body():
                return True
            time.sleep(0.2)
        return False

    def js(code, *args):
        return d.execute_script(code, *args)

    def key(k):
        d.find_element(By.TAG_NAME, "body").send_keys(k)

    def tap(el):
        """Bring the element to the middle of the screen (clear of the fixed
        top and bottom bars, as a person scrolls), then click it."""
        d.execute_script("arguments[0].scrollIntoView({block: 'center'})", el)
        time.sleep(0.2)
        el.click()

    def boot():
        end = time.time() + 180
        while time.time() < end:
            if js("const b = document.getElementById('boot'); return b ? b.hidden : false"):
                return True
            time.sleep(0.3)
        return False

    try:
        t0 = time.time()
        d.get(base)
        check(boot(), f"app boots (Python, numpy, scipy, matcher) in {time.time() - t0:.1f} s")

        # real camera stack with Chrome's fake device: MediaPipe and the face worker load
        d.get(base + "#/register")
        check(wait_text("Create a profile", 30), "register page")
        js("document.getElementById('reg-user').value = 'alice'")
        tap(d.find_element(By.CSS_SELECTOR, "form button[type=submit]"))
        check(wait_text("Skip (gesture only)", 60), "face step shown")
        end = time.time() + 60
        while time.time() < end and js("return (window.__cam = window.__cam || 0), true") and \
                d.execute_async_script("const done = arguments[0]; import('/js/api.js').then(({service}) => done(service.camera.state))") != "running":
            time.sleep(0.5)
        state = d.execute_async_script("const done = arguments[0]; import('/js/api.js').then(({service}) => done(service.camera.status()))")
        check(state["state"] == "running" and state["fps"] > 0,
              f"camera running with the real MediaPipe hand landmarker ({state['fps']} fps, {state['process_ms']} ms per frame)")
        shot("01_register_face")

        ok = d.execute_async_script(INSTALL, gestures())
        check(ok is True, f"synthetic hand input installed ({ok})")

        # registration: skip the face, five samples
        key("s")
        check(wait_text("Sample 1 of 5", 30) or wait_text("Record sample 1", 5), "gesture step")
        samples = 0
        end = time.time() + 300
        while time.time() < end:
            st = js("const r = window.__wl.service.reg; return r && {a: r.awaiting_record, s: r.status, n: r.current_sample, acc: r.accepted, ph: r.phase}")
            if not st or st["s"] != "active":
                break
            if st["a"]:
                js("window.__wl.mode = 'enrol' + arguments[0]", min(4, st["acc"]))
                bar = d.find_elements(By.CSS_SELECTOR, ".cam-actions .btn.primary")
                if samples == 0:
                    check(bool(bar) and bar[0].is_displayed() and "Record sample" in bar[0].text,
                          "action bar under the camera mirrors 'Record sample'")
                    geo = js("const c = document.querySelector('.camera').getBoundingClientRect();"
                             "const b = document.querySelector('.cam-actions').getBoundingClientRect();"
                             "return {gap: b.top - c.bottom, bottom: b.bottom, vh: window.innerHeight};")
                    check(0 <= geo["gap"] < 80 and geo["bottom"] <= geo["vh"],
                          f"action bar directly below the camera and on screen ({geo})")
                    shot("01b_register_action_bar")
                tap(bar[0]) if bar and bar[0].is_enabled() else key("r")
                samples += 1
                time.sleep(0.6)
            time.sleep(0.2)
        check(wait_text("Registration complete for alice", 120), f"registration complete ({samples} recordings)")
        shot("02_register_done")

        # sign-in: liveness (answered by the synthetic hand), then start with R
        d.get(base + "#/signin?user=alice")
        check(wait_text("Your hand is your password", 30), "sign-in page")
        tap(d.find_element(By.CSS_SELECTOR, "form button[type=submit]"))
        check(wait_text("Start recording", 30), "sign-in session")
        time.sleep(1.5)
        js("window.__wl.mode = 'genuine'; window.__wl.answer = true")
        bar = d.find_elements(By.CSS_SELECTOR, ".cam-actions .btn.primary")
        check(bool(bar) and "Start recording" in bar[0].text, "action bar mirrors 'Start recording'")
        tap(bar[0])
        check(wait_text("Liveness passed", 30), "liveness challenge passed (finger counts)")
        shot("03_liveness_passed")
        time.sleep(1.0)
        check("Liveness passed" in body(), "nothing starts by itself after the challenge")
        bar = d.find_elements(By.CSS_SELECTOR, ".cam-actions .btn.primary")
        check(bool(bar) and "Start recording" in bar[0].text and bar[0].is_enabled(),
              "after the challenge the action bar offers 'Start recording'")
        tap(bar[0])
        check(wait_text("Access granted", 90), "genuine gesture: access granted")
        shot("04_granted")

        # a different gesture
        key("t")
        check(wait_text("Start recording", 30), "try again")
        js("window.__wl.mode = 'wrong'")
        time.sleep(1.0)
        key("r")
        check(wait_text("Liveness passed", 30), "liveness passed again")
        key("r")
        check(wait_text("Access denied", 90), "different gesture: access denied")
        shot("05_denied")

        # a video cannot follow the instruction
        key("t")
        check(wait_text("Start recording", 30), "try again")
        js("window.__wl.answer = false")
        time.sleep(1.0)
        key("r")
        check(wait_text("Liveness check failed", 40), "unanswered challenge: liveness check failed")
        shot("06_liveness_failed")

        # profiles persist in IndexedDB across a reload
        d.get(base + "#/users")
        d.refresh()
        check(boot(), "app boots again after reload")
        check(wait_text("alice", 30), "profile still there after reload (IndexedDB)")
        log = d.execute_async_script(
            "const done = arguments[0]; import('/js/api.js').then(({api}) => api.activity(10)).then(done)")
        check(len(log["entries"]) == 3 and [e["granted"] for e in log["entries"]] == [False, False, True],
              f"sign-in log kept: {[e['granted'] for e in log['entries']]}")
        shot("07_users")
        d.get(base + "#/security")
        check(wait_text("Profiles are stored unencrypted in the browser", 30), "security page (mobile texts)")
        shot("08_security")
        d.get(base + "#/settings")
        check(wait_text("Developer mode", 30), "settings page")
        for label in ("Back", "Front (selfie)"):
            btn = d.find_element(By.XPATH, f"//button[contains(., '{label}')]")
            tap(btn)
            time.sleep(1.0)
            check(btn.get_attribute("aria-pressed") == "true" and "Cannot read properties" not in body(),
                  f"settings: camera '{label}' selected without an error")
        nav = js("const n = document.querySelector('.sidebar').getBoundingClientRect();"
                 "return {top: n.top, bottom: n.bottom, vh: window.innerHeight,"
                 " labels: [...document.querySelectorAll('#nav a span')].map(s => s.offsetParent !== null)}")
        check(nav["bottom"] >= nav["vh"] - 1 and nav["top"] > nav["vh"] / 2 and all(nav["labels"]),
              "navigation is a bottom tab bar with labels")
        check(js("const b = document.querySelector('.mobile-brand'); return !!b && b.offsetParent !== null"),
              "top bar shows the WaveLock brand")
        shot("09_settings")

        problems = [e["message"][:200] for e in d.get_log("browser")
                    if e["level"] == "SEVERE" or "Content Security Policy" in e["message"]]
        check(not problems, "no console errors or CSP violations" + (f": {problems[:3]}" if problems else ""))
    finally:
        d.quit()
        if srv:
            srv.shutdown()
    print("ALL PASSED" if not failures else f"{len(failures)} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
