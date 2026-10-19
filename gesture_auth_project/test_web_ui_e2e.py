"""
Optional browser test of the WaveLock web UI (headless Chrome + Selenium).

Drives the real UI against the server with the SIMULATED camera from
test_web_backend.py: registration wizard, sign-in granted / denied, all
pages in light and dark mode, phone width, developer mode, benchmark and
face dialog. Fails on console / CSP errors, leaked scores in protected
mode and horizontal scrolling. Saves screenshots.

Usage:  python test_web_ui_e2e.py [screenshot_dir]
Needs:  pip install selenium  and Google Chrome (not required by WaveLock itself).
"""
import os, sys, time, json, shutil, tempfile

PROJ = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(tempfile.gettempdir(), "wavelock_ui_shots")
sys.path.insert(0, PROJ)
os.chdir(PROJ)
os.makedirs(OUT, exist_ok=True)

import test_wavelock as tw
import gesture_auth
from test_web_backend import make_app, WRONG_ORDER

# SIMULATED face stage: the drawn ellipse is not a real face, so the real
# SFace check would (correctly) reject it. Stand in a match for the UI test.
gesture_auth.verify_face_over_frames = lambda frames, emb: {
    "match": True, "score": 0.74, "confidence": 0.86, "reason": None,
    "frames_checked": 6, "faces_detected": 6}
try:
    from selenium import webdriver
except ImportError:
    print("SKIPPED: selenium is not installed (pip install selenium).")
    sys.exit(0)
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException

problems = []


def driver_for(width, height):
    opts = webdriver.ChromeOptions()
    opts.add_argument("--headless=new")
    opts.add_argument(f"--window-size={width},{height}")
    opts.add_argument("--force-device-scale-factor=1")
    opts.add_argument("--hide-scrollbars")
    opts.set_capability("goog:loggingPrefs", {"browser": "ALL"})
    return webdriver.Chrome(options=opts)


def console_errors(d, label):
    for entry in d.get_log("browser"):
        if entry["level"] in ("SEVERE",) or "Content Security Policy" in entry["message"]:
            # aborted MJPEG streams on navigation are expected
            if "/api/stream" in entry["message"] and "net::ERR" in entry["message"]:
                continue
            problems.append(f"[{label}] console: {entry['message'][:300]}")


def shot(d, name):
    time.sleep(0.4)
    d.save_screenshot(os.path.join(OUT, f"{name}.png"))


CURRENT_APP = {}


def click(d, el):
    """Scroll an element to the middle of the window first (fixed bars sit at the edges), then click."""
    d.execute_script("arguments[0].scrollIntoView({block: 'center'});", el)
    time.sleep(0.15)
    el.click()


def wait_text(d, text, timeout=40):
    try:
        WebDriverWait(d, timeout).until(lambda drv: text.lower() in drv.find_element(By.TAG_NAME, "body").text.lower())
    except Exception:
        d.save_screenshot(os.path.join(OUT, "TIMEOUT.png"))
        app = CURRENT_APP.get("app")
        if app:
            st = app.ok("GET", "/api/status")
            print("STATUS AT TIMEOUT:", json.dumps({k: st[k] for k in ("activity", "register")}, default=str)[:1500])
        raise


def set_theme(d, theme):
    d.execute_script("localStorage.setItem('wavelock-theme', arguments[0]);", theme)
    d.refresh()
    time.sleep(0.8)


def check_preview(d, label, timeout=15):
    """The camera preview must show a real picture, not a black box."""
    try:
        WebDriverWait(d, timeout).until(lambda drv: drv.execute_script(
            "const i = document.querySelector('.camera img.feed');"
            "return !!i && i.complete && i.naturalWidth > 0 && "
            "getComputedStyle(document.querySelector('.cam-boot')).display === 'none';"))
    except Exception:
        problems.append(f"[{label}] camera preview shows no picture")


def check_no_hscroll(d, label):
    w = d.execute_script("return [document.documentElement.scrollWidth, window.innerWidth]")
    if w[0] > w[1] + 1:
        problems.append(f"[{label}] horizontal scroll: page {w[0]}px > viewport {w[1]}px")


def run(debug):
    tmp = tempfile.mkdtemp(prefix="wavelock_e2e_")
    app = make_app(tmp, debug=debug)
    CURRENT_APP["app"] = app
    tag = "dev" if debug else "normal"
    d = driver_for(1440, 960)
    try:
        base = app.base
        d.get(base + "/#/dashboard")
        wait_text(d, "Recent activity")
        shot(d, f"{tag}_01_dashboard_empty")
        # theme switch in the masthead flips the theme and its state
        before = d.execute_script("return document.documentElement.dataset.theme")
        click(d, d.find_element(By.ID, "theme-toggle"))
        after = d.execute_script("return document.documentElement.dataset.theme")
        checked = d.find_element(By.ID, "theme-toggle").get_attribute("aria-checked")
        if before == after or checked != str(after == "dark").lower():
            problems.append(f"[{tag}] theme switch did not flip ({before} -> {after}, aria-checked {checked})")
        click(d, d.find_element(By.ID, "theme-toggle"))

        # ---------------- register through the wizard
        d.get(base + "/#/register")
        wait_text(d, "Create a profile")
        shot(d, f"{tag}_02_register_account")
        d.find_element(By.ID, "reg-user").send_keys("alice")
        click(d, d.find_element(By.CSS_SELECTOR, "form button[type=submit]"))
        wait_text(d, "Capture face")
        check_preview(d, f"{tag}/register-face")
        shot(d, f"{tag}_03_register_face")
        if debug:
            click(d, d.find_element(By.XPATH, "//button[contains(., 'Capture face')]"))
        else:                                                           # C = capture face
            WebDriverWait(d, 20).until(EC.element_to_be_clickable(
                (By.XPATH, "//button[contains(., 'Capture face') and not(@disabled)]")))
            d.find_element(By.TAG_NAME, "body").send_keys("c")
        wait_text(d, "Sample 1 of 5", 30)
        pool = iter(tw.genuine_set(count=12, seed=100))
        for n in range(1, 30):
            st = app.ok("GET", "/api/status")["register"]
            if st is None or st["status"] != "active":
                break
            if not st["awaiting_record"]:
                time.sleep(0.15)
                continue
            app.tracker.play(next(pool))
            # The page follows the server about every 200 ms: wait until the
            # enabled record button is on screen, as a person would.
            btn = WebDriverWait(d, 20).until(EC.element_to_be_clickable(
                (By.XPATH, "//button[contains(., 'Record sample') and not(@disabled)]")))
            if n == 2:
                d.find_element(By.TAG_NAME, "body").send_keys("r")       # keyboard shortcut
            else:
                # The page re-renders once when a sample finishes; if that
                # replaced the button between finding and clicking, find it again.
                for _attempt in range(5):
                    try:
                        click(d, btn)
                        break
                    except StaleElementReferenceException:
                        btn = WebDriverWait(d, 20).until(EC.element_to_be_clickable(
                            (By.XPATH, "//button[contains(., 'Record sample') and not(@disabled)]")))
            if n == 3:
                time.sleep(0.45)
                shot(d, f"{tag}_04_register_recording")
            WebDriverWait(d, 30).until(lambda drv: (lambda s: s is None or s["status"] != "active"
                                                    or s["awaiting_record"] or s["phase"] == "saving")(
                app.ok("GET", "/api/status")["register"]))
            if n == 3:
                shot(d, f"{tag}_05_register_after_sample")
        wait_text(d, "Registration complete for alice", 60)
        shot(d, f"{tag}_06_register_review")

        # ---------------- sign in: granted, then denied
        click(d, d.find_element(By.XPATH, "//a[contains(., 'Sign in now')]"))
        wait_text(d, "Your hand is your password")
        val = d.find_element(By.ID, "username").get_attribute("value")
        if val != "alice":
            problems.append(f"[{tag}] sign-in username not prefilled: {val!r}")
        shot(d, f"{tag}_07_signin_landing")
        click(d, d.find_element(By.CSS_SELECTOR, "form button[type=submit]"))
        wait_text(d, "Start recording")
        check_preview(d, f"{tag}/signin")
        shot(d, f"{tag}_08_signin_ready")
        app.tracker.play(tw.genuine_set(1, seed=31)[0])
        click(d, d.find_element(By.XPATH, "//button[contains(., 'Start recording')]"))
        time.sleep(0.5)
        shot(d, f"{tag}_09_signin_recording")
        wait_text(d, "Access granted", 40)
        shot(d, f"{tag}_10_signin_granted")
        click(d, d.find_element(By.XPATH, "//button[contains(., 'Sign in again')]"))
        wait_text(d, "Start recording")
        app.tracker.play(WRONG_ORDER)
        d.find_element(By.TAG_NAME, "body").send_keys(" ")                # Space shortcut
        wait_text(d, "Access denied", 40)
        body = d.find_element(By.TAG_NAME, "body").text
        if debug and "developer details" not in body.lower():
            problems.append("[dev] developer details missing after denial")
        if not debug and ("developer details" in body.lower() or "DTW" in body or "cosine" in body.lower()):
            problems.append("[normal] scores leaked into the protected sign-in screen")
        if not debug and "attempts left" not in body:
            problems.append("[normal] attempts-left message missing")
        shot(d, f"{tag}_11_signin_denied")

        # ---------------- keyboard shortcuts after a denial: ?, Esc, T, R
        body_el = d.find_element(By.TAG_NAME, "body")
        body_el.send_keys("?")
        wait_text(d, "Keyboard shortcuts", 10)
        listed = d.find_element(By.CSS_SELECTOR, ".modal").text
        if "Try again" not in listed or "Switch user" not in listed:
            problems.append(f"[{tag}] shortcut list incomplete: {listed!r}")
        shot(d, f"{tag}_11b_shortcuts")
        d.find_element(By.TAG_NAME, "body").send_keys(Keys.ESCAPE)
        WebDriverWait(d, 10).until(lambda drv: not drv.find_elements(By.CSS_SELECTOR, ".modal-backdrop"))
        d.find_element(By.TAG_NAME, "body").send_keys("t")              # T = try again
        wait_text(d, "Start recording", 20)
        app.tracker.play(WRONG_ORDER)
        WebDriverWait(d, 20).until(EC.element_to_be_clickable(
            (By.XPATH, "//button[contains(., 'Start recording') and not(@disabled)]")))
        d.find_element(By.TAG_NAME, "body").send_keys("r")              # R = record
        wait_text(d, "Access denied", 40)
        if debug:
            d.execute_script("window.scrollTo(0, document.body.scrollHeight)")
            shot(d, f"{tag}_12_signin_denied_details")

        # ---------------- other pages, both themes
        for theme in ("light", "dark"):
            set_theme(d, theme)
            for page, text in (("dashboard", "Recent activity"), ("users", "alice"),
                               ("security", "Protections in force"), ("settings", "Appearance")):
                d.get(base + f"/#/{page}")
                wait_text(d, text)
                shot(d, f"{tag}_20_{page}_{theme}")
                check_no_hscroll(d, f"{tag}/{page}/{theme}")
        if debug:
            d.get(base + "/#/security")
            wait_text(d, "SIMULATED benchmarks")
            click(d, d.find_element(By.XPATH, "//button[contains(., 'Run benchmark')]"))
            wait_text(d, "False Acceptance Rate", 600)
            d.execute_script("window.scrollTo(0, document.body.scrollHeight)")
            shot(d, f"{tag}_21_benchmark")

        # face enrolment dialog from Users
        d.get(base + "/#/users")
        wait_text(d, "Replace face")
        click(d, d.find_element(By.XPATH, "//button[contains(., 'Replace face')]"))
        wait_text(d, "Capture face")
        check_preview(d, f"{tag}/face-dialog")
        WebDriverWait(d, 20).until(EC.element_to_be_clickable(
            (By.XPATH, "//div[contains(@class,'modal')]//button[contains(., 'Capture face') and not(@disabled)]"))).click()
        wait_text(d, "Face enrolled", 30)
        shot(d, f"{tag}_22_face_dialog_done")
        d.find_element(By.XPATH, "//div[contains(@class,'modal')]//button[contains(., 'Close')]").click()
        console_errors(d, tag)
    finally:
        d.quit()

    # ---------------- phone width
    m = driver_for(390, 844)
    try:
        for theme in ("dark", "light"):
            m.get(app.base + "/#/dashboard")
            set_theme(m, theme)
            for page, text in (("dashboard", "Recent activity"), ("signin", "Your hand is your password"),
                               ("register", "Create a profile"), ("users", "alice")):
                m.get(app.base + f"/#/{page}")
                wait_text(m, text)
                shot(m, f"{tag}_30_mobile_{page}_{theme}")
                check_no_hscroll(m, f"{tag}/mobile/{page}/{theme}")
        console_errors(m, f"{tag}/mobile")
    finally:
        m.quit()
        app.close()
        shutil.rmtree(tmp, ignore_errors=True)


for debug in (False, True):
    try:
        run(debug)
    except Exception as e:
        import traceback
        traceback.print_exc()
        st = CURRENT_APP.get("app")
        if st:
            try:
                print("STATUS AT FAILURE:", json.dumps(st.ok("GET", "/api/status")["register"], default=str)[:800])
            except Exception:
                pass                                   # server already closed
        problems.append(f"[{'dev' if debug else 'normal'}] FAILED: {type(e).__name__}: {e}")

print("\nPROBLEMS:" if problems else "\nNo problems found.")
for p in problems:
    print("  -", p)
print("screenshots:", len(os.listdir(OUT)), "in", OUT)
sys.exit(1 if problems else 0)
