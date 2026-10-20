"""
Desktop check of the mobile app's Python side (mobile/py).

The mobile app runs mobile_core.py in the browser (Pyodide) with stand-ins
for cv2, mediapipe and dtaidistance (mobile/py/shims). This test runs it the
same way in CPython and compares every result with the desktop modules run
normally (real OpenCV / dtaidistance) in a second process on the same
synthetic gestures (test_wavelock.synth_raw). No real profile is read.

Run from gesture_auth_project/:
    python mobile/tests/test_mobile_core.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))
MOBILE_PY = os.path.join(PROJECT, "mobile", "py")
ASPECT = 640 / 480
NOW = 1_800_000_000.0


def make_inputs(path):
    """Synthetic recordings (desktop process, real modules)."""
    sys.path.insert(0, PROJECT)
    import numpy as np
    import test_wavelock as tw

    def pad(r, a, b):
        return np.concatenate([np.repeat(r[:1], a, axis=0), r, np.repeat(r[-1:], b, axis=0)])
    enrol = [pad(r, 12, 12) for r in tw.genuine_set()]
    probes = {
        "genuine": pad(tw.genuine_set(1, seed=31)[0], 12, 12),
        "genuine_late": pad(tw.genuine_set(1, seed=32)[0], 22, 2),
        "wrong_order": pad(tw.synth_raw(order=("ring", "pinky", "index", "middle"), seed=7), 12, 12),
    }
    np.savez(path, enrol=np.array(enrol, dtype=object), **probes, allow_pickle=True)


def reference(inputs, tmp, out):
    """The desktop path: WaveLockService methods with the real modules."""
    sys.path.insert(0, PROJECT)
    import types
    import numpy as np
    import gesture_compare as gc
    from gesture_capture import TARGET_FRAMES, validate_sample_quality
    from utils.normalize import normalize_gesture, aspect_for_method
    from web_backend.service import WaveLockService

    class Cam:
        def status(self):
            return {"state": "stopped"}

    data = np.load(inputs, allow_pickle=True)
    svc = WaveLockService(Cam(), templates_dir=tmp, debug=True)
    poses, raws, gates = [], [], []
    for raw in data["enrol"]:
        pose = normalize_gesture(gc.matching_recording(raw, gc.NEW_PROFILE_TEMPORAL_TRIM), TARGET_FRAMES,
                                 frame_aspect=aspect_for_method(gc.POSE_METHOD_LEGACY, ASPECT))
        ok, msg = validate_sample_quality(pose, poses)
        gates.append([bool(ok), msg])
        if ok:
            poses.append(pose)
            raws.append(np.asarray(raw, np.float64))
    r = {"username": "alice", "replacing": False, "pose_method": gc.POSE_METHOD_LEGACY,
         "check_level": "standard", "face_embedding": None}
    summary = svc._save_registration(r, poses, raws, ASPECT)
    summary.pop("location", None)
    with open(os.path.join(tmp, "alice", "config.json")) as f:
        config = json.load(f)
    results = {}
    for name in ("genuine", "genuine_late", "wrong_order"):
        profile = gc.load_user_profile("alice", tmp)
        rec = types.SimpleNamespace(landmarks=list(data[name]), clean_frames=[], frame_aspect=ASPECT)
        payload = svc._evaluate({"username": "alice", "profile": profile}, rec)
        payload.pop("time", None)
        results[name] = payload
    with open(os.path.join(tmp, "alice", "config.json")) as f:
        config_after = json.load(f)                  # after a grant: template aging
    json.dump({"gates": gates, "summary": summary, "results": results, "config": config,
               "config_after": config_after,
               "users": svc.users()}, open(out, "w"), default=str)


def without_times(config):
    """config.json without the aging timestamps (wall-clock, differ by run)."""
    c = json.loads(json.dumps(config))
    c.get("aging_anchor", {}).pop("created", None)
    for e in c.get("aging_log", []):
        e.pop("time", None)
    return c


def close(a, b, path="", tol=1e-9):
    """Recursive comparison; returns a list of differences."""
    if isinstance(a, dict) and isinstance(b, dict):
        diffs = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                diffs.append(f"{path}.{k}: missing on one side")
            else:
                diffs += close(a[k], b[k], f"{path}.{k}", tol)
        return diffs
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        return sum((close(x, y, f"{path}[{i}]", tol) for i, (x, y) in enumerate(zip(a, b))), [])
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return [] if abs(a - b) <= tol * max(1.0, abs(a), abs(b)) else [f"{path}: {a} != {b}"]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def mobile_checks(inputs, ref, tmp, check):
    """The mobile side, run through the stand-ins (CPython here, Pyodide in
    tests/pyodide_core.mjs). sys.path must already put the shims first."""
    import numpy as np
    import cv2
    import dtaidistance
    check("missing" in repr(cv2.FaceDetectorYN) and "shims" in dtaidistance.__file__,
          "cv2 / dtaidistance resolve to the mobile stand-ins")
    import mobile_core as mc
    import gesture_compare as gc
    check("shims" in gc.dtw_ndim.__file__, "gesture_compare uses the NumPy DTW stand-in")

    data = np.load(inputs, allow_pickle=True)
    mc.init(tmp, debug=True)
    check(mc.call("auth_begin", {"raw_username": "nobody"})["__error__"]["code"] == "not_registered",
          "errors come back as __error__ with the service's code")

    st = mc.call("reg_begin", {"raw_username": "Alice", "pose_method": "legacy"})
    check(st == {"username": "alice", "replacing": False, "pose_method": gc.POSE_METHOD_LEGACY},
          "reg_begin normalises the name like the web app")
    gates = []
    for raw in data["enrol"]:
        res = mc.call("reg_sample", {"landmarks": np.asarray(raw).tolist(), "frame_aspect": ASPECT})
        gates.append([res["ok"], res["message"]])
    check(gates == ref["gates"], "quality gate: same accept / reject and messages")
    summary = mc.call("reg_save", {})
    d = close(summary, ref["summary"])
    check(not d, "registration report identical to the desktop" + (f" {d[:3]}" if d else ""))
    config = json.load(open(os.path.join(tmp, "alice", "config.json")))
    d = close(config, ref["config"])
    check(not d, "saved config.json identical (thresholds, trim, kinematics)" + (f" {d[:3]}" if d else ""))

    mc.call("auth_begin", {"raw_username": "alice"})
    for name in ("genuine", "genuine_late", "wrong_order"):
        payload = mc.call("evaluate", {"username": "alice", "landmarks": np.asarray(data[name]).tolist(),
                                       "frame_aspect": ASPECT})
        payload.pop("time", None)
        d = close(payload, ref["results"][name])
        check(not d, f"sign-in '{name}' identical to the desktop (granted={payload['granted']})"
              + (f" {d[:3]}" if d else ""))
    check(ref["results"]["genuine"]["granted"] and not ref["results"]["wrong_order"]["granted"],
          "reference sanity: genuine granted, wrong finger order denied")
    after = json.load(open(os.path.join(tmp, "alice", "config.json")))
    d = close(without_times(after), without_times(ref["config_after"]))
    check(not d and "aging_log" in after,
          "after the sign-ins: same template aging (anchor, log, recalibration)" + (f" {d[:3]}" if d else ""))
    users_m = mc.call("users", {})
    for u in users_m + ref["users"]:
        u.pop("last_attempt", None)
    check(not close(users_m, ref["users"]), "users() identical")

    # attempt limit: five failures lock the account (utils.attempt_limit)
    for _ in range(5):
        mc.call("evaluate", {"username": "alice", "landmarks": np.asarray(data["wrong_order"]).tolist(),
                             "frame_aspect": ASPECT})
    check(mc.call("locked_for", {"username": "alice"}) > 0, "five failures lock the account")

    # liveness: the Python challenge, driven frame by frame
    import test_wavelock as tw
    pub = mc.call("challenge_new", {"username": "alice", "now": NOW})
    want = pub["steps"][0]["count"]
    t = NOW
    for k in range(30):
        t += 0.05
        st = mc.call("challenge_feed", {"hand": tw.finger_pose(want).tolist(), "frame_aspect": ASPECT,
                                        "now": t})
    check(st["public"]["index"] >= 1 or st["public"]["state"] == "passed",
          "challenge: holding the requested fingers completes the step")
    pub = mc.call("challenge_new", {"username": "alice", "now": NOW})
    five = np.array([0.5, 0.65, 0.0]) + tw._build_hand_frame({f: True for f in tw.FINGERS}) * 0.18
    five[:, 0] = 0.5 + tw._build_hand_frame({f: True for f in tw.FINGERS})[:, 0] * 0.18 / ASPECT
    for k in range(30):
        st = mc.call("challenge_feed", {"hand": five.tolist(), "frame_aspect": ASPECT, "now": NOW + 0.05 * k})
    check(st["public"]["index"] == 0 and (pub["steps"][0]["count"] != 5),
          "challenge: five fingers never answer a request for 1-4")
    st = mc.call("challenge_feed", {"hand": None, "now": NOW + 100})
    check(st["public"]["state"] == "failed" and st["public"]["reason"] == "liveness_timeout_fingers",
          "challenge times out like the desktop")
    res = mc.call("liveness_failed", {"username": "alice", "reason": "liveness_timeout_fingers"})
    check(res["liveness_failed"] and res["granted"] is False, "liveness failure payload")

    # face: embeddings computed in the browser, decision in Python
    mc.call("reg_begin", {"raw_username": "bob"})
    for raw in data["enrol"]:
        mc.call("reg_sample", {"landmarks": np.asarray(raw).tolist(), "frame_aspect": ASPECT})
    rng = np.random.RandomState(4)
    me = rng.randn(128)
    ok = mc.call("reg_face", {"embeddings": [(me + rng.randn(128) * 0.05).tolist() for _ in range(5)]})
    bad = mc.call("reg_begin", {"raw_username": "carl"}) and mc.call(
        "reg_face", {"embeddings": [rng.randn(128).tolist() for _ in range(5)]})
    check(ok["ok"] and not bad["ok"], "face enrolment: consistent frames accepted, mixed refused")
    mc.call("reg_cancel", {})
    mc.call("reg_begin", {"raw_username": "bob"})
    for raw in data["enrol"]:
        mc.call("reg_sample", {"landmarks": np.asarray(raw).tolist(), "frame_aspect": ASPECT})
    mc.call("reg_face", {"embeddings": [(me + rng.randn(128) * 0.05).tolist() for _ in range(5)]})
    s = mc.call("reg_save", {})
    check(s["face"] == "enrolled", "registration with a face anchor")
    mc.call("auth_begin", {"raw_username": "bob"})
    idx = mc.call("face_sample_indices", {"n_frames": 90})
    check(idx == [0, 17, 35, 53, 71, 89], "face frames sampled like the desktop")
    same = [(me + rng.randn(128) * 0.1).tolist() for _ in idx]
    other = [rng.randn(128).tolist() for _ in idx]
    g = mc.call("evaluate", {"username": "bob", "landmarks": np.asarray(data["genuine"]).tolist(),
                             "frame_aspect": ASPECT, "face_samples": same})
    x = mc.call("evaluate", {"username": "bob", "landmarks": np.asarray(data["genuine"]).tolist(),
                             "frame_aspect": ASPECT, "face_samples": other})
    n = mc.call("evaluate", {"username": "bob", "landmarks": np.asarray(data["genuine"]).tolist(),
                             "frame_aspect": ASPECT, "face_samples": [None] * 6})
    check(g["granted"] and g["details"]["face"]["match"], "own face + gesture granted")
    check(not x["granted"] and x["details"]["failure_reason"] == "impostor_face_identity",
          "another face denied (impostor_face_identity)")
    check(not n["granted"] and n["details"]["failure_reason"] == "no_face_detected",
          "no face denied (no_face_detected)")
    b = mc.call("challenge_bind", {"username": "bob", "embedding": me.tolist()})
    c = mc.call("challenge_bind", {"username": "bob", "embedding": rng.randn(128).tolist()})
    check(b["passed"] and c["reason"] == "liveness_face_mismatch", "liveness bound to the enrolled face")

    # owner check: replacing an existing profile needs a recent sign-in
    e = mc.call("reg_begin", {"raw_username": "alice", "overwrite": True})
    check(e.get("__error__", {}).get("code") == "owner_signin_required", "owner sign-in required to replace")
    e = mc.call("face_enroll_check", {"raw_username": "alice"})
    check(e.get("__error__", {}).get("code") == "owner_signin_required", "owner sign-in required for a face")


def main():
    work = tempfile.mkdtemp(prefix="wl_mobile_core_")
    failures = []

    def check(cond, label):
        print(("PASS  " if cond else "FAIL  ") + label)
        if not cond:
            failures.append(label)
    try:
        inputs = os.path.join(work, "inputs.npz")
        ref_json = os.path.join(work, "reference.json")
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        for args in (["--inputs", inputs], ["--reference", inputs, os.path.join(work, "desktop"), ref_json]):
            out = subprocess.run([sys.executable, __file__] + args, cwd=PROJECT, env=env,
                                 capture_output=True, text=True, encoding="utf-8", errors="replace")
            if out.returncode:
                print(out.stdout[-2000:], out.stderr[-3000:])
                raise SystemExit("desktop reference failed")
        ref = json.load(open(ref_json))

        # ---------------------------------------------------------------- mobile side (shims)
        sys.path[:0] = [os.path.join(MOBILE_PY, "shims"), MOBILE_PY, PROJECT]
        mobile_checks(inputs, ref, os.path.join(work, "mobile"), check)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"{'ALL PASSED' if not failures else f'{len(failures)} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["--inputs"]:
        make_inputs(sys.argv[2])
    elif sys.argv[1:2] == ["--reference"]:
        reference(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        sys.exit(main())
