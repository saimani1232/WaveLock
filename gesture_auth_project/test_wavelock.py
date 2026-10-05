"""
WaveLock self-checks — run with:  python test_wavelock.py [name ...]

No camera needed: gestures are synthesised as raw MediaPipe-style landmark
sequences (a hand raising fingers in order while the wrist follows a path),
so registration, authentication, aging and the evaluation scripts can be
exercised end to end in a temporary templates directory.
"""

import os
import sys
import shutil
import tempfile
import subprocess

import numpy as np

from cohort_library import _build_hand_frame
from utils.normalize import normalize_gesture, extract_wrist_trajectory
from utils.kinematics import (
    prepare_kinematic_sequence, extract_jerk_metric, KINEMATIC_METHOD_SMOOTHED,
)
from utils.face_auth import classify_face_score
import gesture_compare as gc

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
ASPECT = 640 / 480


def synth_raw(order=("index", "middle", "pinky", "ring"), n_frames=80,
              path="circle", seed=0, noise=0.002, warp=0.0, radius=0.08,
              hand_scale=0.18):
    """Raw landmarks (n_frames, 21, 3) in MediaPipe image coordinates."""
    rng = np.random.RandomState(seed)
    keys = [{f: False for f in FINGERS}]
    for finger in order:
        nxt = dict(keys[-1])
        nxt[finger] = True
        keys.append(nxt)
    key_frames = [_build_hand_frame(k) for k in keys]

    t = np.linspace(0.0, 1.0, n_frames)
    if warp:
        t = np.clip(t + warp * np.sin(np.pi * t), 0.0, 1.0)
    frames = []
    for ti in t:
        pos = ti * (len(key_frames) - 1)
        i = min(int(pos), len(key_frames) - 2)
        a = pos - i
        a = a * a * (3 - 2 * a)                    # smoothstep between poses
        pose = (1 - a) * key_frames[i] + a * key_frames[i + 1]

        if path == "circle":
            wrist = np.array([0.5 + radius * np.cos(2 * np.pi * ti) / ASPECT,
                              0.65 + radius * np.sin(2 * np.pi * ti), 0.0])
        elif path == "line":
            wrist = np.array([0.35 + 0.3 * ti / ASPECT, 0.65, 0.0])
        else:                                      # static
            wrist = np.array([0.5, 0.65, 0.0])

        frame = wrist + pose * hand_scale
        frame[:, 0] = wrist[0] + pose[:, 0] * hand_scale / ASPECT
        frames.append(frame)
    raw = np.array(frames)
    return raw + rng.normal(0, noise, raw.shape)


def genuine_set(count=5, seed=100, **kw):
    raws = []
    rng = np.random.RandomState(seed)
    for k in range(count):
        raws.append(synth_raw(n_frames=int(rng.randint(70, 91)), seed=seed + k,
                              warp=rng.uniform(-0.05, 0.05), **kw))
    return raws


def register(tmp, username="alice", raws=None, face=None):
    raws = raws if raws is not None else genuine_set()
    poses = [normalize_gesture(r) for r in raws]
    gc.save_registration(username, poses, templates_dir=tmp, raw_samples=raws,
                         frame_aspect=ASPECT, face_embedding=face)
    return gc.load_user_profile(username, templates_dir=tmp)


def attempt(profile, raw, face_match=None, face_conf=1.0, face_details=None):
    pose = normalize_gesture(raw)
    traj = (extract_wrist_trajectory(raw, ASPECT)
            if profile["trajectory_templates"] else None)
    return gc.authenticate_with_details(
        pose, profile["templates"], profile["threshold"],
        profile["finger_state_threshold"], profile["transition_threshold"],
        profile["segment_threshold"], profile["anthropometric_profile"],
        profile["kinematic_profile"], face_match=face_match,
        face_confidence=face_conf, face_details=face_details,
        live_trajectory=traj, stored_trajectories=profile["trajectory_templates"],
        trajectory_threshold=profile["trajectory_threshold"],
        live_kinematic_sequence=prepare_kinematic_sequence(raw),
    )


class TempDir:
    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix="wavelock_test_")
        return self.path

    def __exit__(self, *exc):
        shutil.rmtree(self.path, ignore_errors=True)


# ---------------------------------------------------------------- tests

def test_usernames():
    assert gc.normalize_username("  Sai Mani ") == "sai_mani"
    assert gc.normalize_username("Bob-2") == "bob-2"
    for bad in ["", "   ", "../evil", "a/b", "a\\b", ".hidden", "a..b", "x" * 65]:
        try:
            gc.normalize_username(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted unsafe username {bad!r}")


def test_registration_roundtrip_and_listing():
    with TempDir() as tmp:
        profile = register(tmp)
        assert profile["raw_templates"] is not None and len(profile["raw_templates"]) == 5
        assert profile["trajectory_threshold"] >= gc.MIN_TRAJECTORY_THRESHOLD
        assert profile["kinematic_profile"]["method"] == KINEMATIC_METHOD_SMOOTHED
        assert profile["face_embedding"] is None
        # staging / junk directories must never appear as users
        os.makedirs(os.path.join(tmp, ".alice.partial"))
        os.makedirs(os.path.join(tmp, "Bad Name"))
        np.save(os.path.join(tmp, "Bad Name", "gesture_1.npy"), np.zeros((60, 21, 3)))
        assert gc.list_registered_users(tmp) == ["alice"]


def test_genuine_accepted():
    with TempDir() as tmp:
        profile = register(tmp)
        accepted = 0
        for k, raw in enumerate(genuine_set(count=10, seed=500)):
            granted, _, _, d = attempt(profile, raw)
            accepted += granted
            if not granted:
                print(f"    genuine #{k} rejected: {d['failure_reason']}")
        assert accepted >= 9, f"genuine acceptance too low: {accepted}/10"


def test_wrong_finger_order_rejected():
    with TempDir() as tmp:
        profile = register(tmp)
        raw = synth_raw(order=("ring", "pinky", "index", "middle"), seed=7)
        granted, _, _, d = attempt(profile, raw)
        assert not granted and "transition_order" in d["failure_reason"], d["failure_reason"]


def test_trajectory_gate():
    with TempDir() as tmp:
        profile = register(tmp)                           # enrolled: circular path
        same_pose_other_path = synth_raw(path="line", seed=8)
        granted, _, _, d = attempt(profile, same_pose_other_path)
        assert not granted and "trajectory" in d["failure_reason"], d["failure_reason"]
        still = synth_raw(path="static", seed=9)
        granted, _, _, d = attempt(profile, still)
        assert not granted and "trajectory" in d["failure_reason"], d["failure_reason"]


def test_dtw_gate_is_enforced():
    # Old rule: transition + fused>=0.55 only. With the DTW score at 0 and
    # perfect finger scores the fused score is exactly 0.55 and access was
    # granted although Gate 1 failed.
    with TempDir() as tmp:
        profile = register(tmp)
        raw = genuine_set(count=1, seed=900)[0]
        pose = normalize_gesture(raw)
        template = pose.copy()
        template[:, 1:, :] += 1e-3        # DTW > 0, identical finger states
        granted, _, _, d = gc.authenticate_with_details(
            pose, [template], 1e-9, 0.3, 0.4, 0.5,
            profile["anthropometric_profile"], profile["kinematic_profile"],
            live_kinematic_sequence=prepare_kinematic_sequence(raw),
        )
        assert d["score_finger"] == 1.0 and d["score_segment"] == 1.0
        assert not granted and "distance" in d["failure_reason"], d["failure_reason"]


def test_face_reasons_and_soft_consensus():
    with TempDir() as tmp:
        profile = register(tmp)
        raw = genuine_set(count=1, seed=901)[0]
        _, _, _, d = attempt(profile, raw, face_match=False, face_conf=0.0,
                             face_details={"score": 0.0, "reason": "no_face_detected"})
        assert d["failure_reason"] == "no_face_detected", d["failure_reason"]
        _, _, _, d = attempt(profile, raw, face_match=False, face_conf=0.45,
                             face_details={"score": 0.47, "reason": "lookalike_sibling_detected"})
        assert d["failure_reason"] == "lookalike_sibling_detected"
        # numpy bools must behave like Python bools
        granted, _, _, _ = attempt(profile, raw, face_match=np.bool_(True), face_conf=0.9,
                                   face_details={"score": 0.75, "reason": None})
        assert granted

        # Soft consensus must use the RAW cosine (0.62 < 0.70) not the mapped
        # confidence (~0.75 at cosine 0.62). Build a borderline hand: two
        # ratios deviate by 2.3x tolerance, three not at all -> fails the
        # 4-of-5 rule while mean deviation (0.92 tol) and confidence (~0.43)
        # stay inside the soft-consensus window.
        from utils.anthropometrics import extract_gesture_anthropometric_signature
        live_sig = extract_gesture_anthropometric_signature(normalize_gesture(raw))
        tol = profile["anthropometric_profile"]["tolerance"]
        dev = np.array([2.3 * tol, 2.3 * tol, 0.0, 0.0, 0.0])
        borderline = dict(profile, anthropometric_profile={
            "baseline": list(live_sig / (1.0 + dev)), "tolerance": tol,
            "method": "metacarpal_invariant_v2"})
        is_match, conf, _ = classify_face_score(0.62)
        assert is_match and conf >= 0.70
        _, _, _, d = attempt(borderline, raw, face_match=True, face_conf=conf,
                             face_details={"score": 0.62, "reason": None})
        assert d["anthropometric_match"] is False, "test hand did not fail anthropometry"
        assert d["kinematic_match"] and d["anthropometric_confidence"] >= 0.35, d["anthropometric_details"]
        assert not d["soft_consensus_used"]
        # ...while a genuinely strong face (cosine 0.75) is allowed to use it.
        _, conf75, _ = classify_face_score(0.75)
        granted, _, _, d = attempt(borderline, raw, face_match=True, face_conf=conf75,
                                   face_details={"score": 0.75, "reason": None})
        assert d["soft_consensus_used"] and granted, d["failure_reason"]


def test_face_classification_tiers():
    assert classify_face_score(0.80)[0] is True
    m, _, r = classify_face_score(0.45)
    assert not m and r == "lookalike_sibling_detected"
    m, _, r = classify_face_score(0.10)
    assert not m and r == "impostor_face_identity"


def test_kinematic_frame_rate_stability():
    # Same motion recorded at different frame counts must give similar jerk.
    def jerk(n):
        return np.mean([extract_jerk_metric(prepare_kinematic_sequence(
            synth_raw(n_frames=n, seed=s, noise=0.003))) for s in range(6)])
    values = [jerk(n) for n in (45, 60, 75, 90)]
    assert max(values) / min(values) < 1.75, values


def test_legacy_profile_still_works():
    # A user registered before raw recordings existed: pose templates only.
    with TempDir() as tmp:
        raws = genuine_set()
        poses = [normalize_gesture(r) for r in raws]
        gc.save_registration("legacy", poses, templates_dir=tmp)
        profile = gc.load_user_profile("legacy", templates_dir=tmp)
        assert profile["raw_templates"] is None
        assert profile["trajectory_templates"] is None
        assert "method" not in profile["kinematic_profile"]
        granted, _, _, d = attempt(profile, genuine_set(count=1, seed=777)[0])
        assert granted, d["failure_reason"]
        assert d["kinematic_method"] == "legacy_template_v1"


def test_smoothed_profile_requires_live_sequence():
    with TempDir() as tmp:
        profile = register(tmp)
        try:
            gc.authenticate_with_details(normalize_gesture(genuine_set(1)[0]),
                                         profile["templates"], profile["threshold"],
                                         kinematic_profile=profile["kinematic_profile"])
        except ValueError:
            return
        raise AssertionError("smoothed profile silently used a legacy live signature")


def test_face_preserved_and_replaced():
    with TempDir() as tmp:
        face_a = np.random.RandomState(1).randn(128).astype(np.float32)
        face_a /= np.linalg.norm(face_a)
        register(tmp, face=face_a)
        profile = register(tmp)                           # re-register, no face given
        assert np.allclose(profile["face_embedding"], face_a)
        face_b = -face_a
        profile = register(tmp, face=face_b)
        assert np.allclose(profile["face_embedding"], face_b)
        assert gc._read_config("alice", tmp)["face_enrolled"] is True


def test_failed_save_keeps_old_profile():
    with TempDir() as tmp:
        register(tmp)
        before = sorted(os.listdir(os.path.join(tmp, "alice")))
        original = gc.build_profile_config
        gc.build_profile_config = lambda *a, **k: {"threshold": object()}  # not JSON-serialisable
        try:
            try:
                register(tmp)
            except TypeError:
                pass
            else:
                raise AssertionError("expected the save to fail")
        finally:
            gc.build_profile_config = original
        assert sorted(os.listdir(os.path.join(tmp, "alice"))) == before
        assert not os.path.exists(os.path.join(tmp, ".alice.partial"))
        gc.load_user_profile("alice", templates_dir=tmp)


def test_corrupted_config_is_reported_not_overwritten():
    with TempDir() as tmp:
        register(tmp)
        path = os.path.join(tmp, "alice", "config.json")
        with open(path, "w") as f:
            f.write("{not json")
        try:
            gc.load_user_profile("alice", templates_dir=tmp)
        except ValueError as e:
            assert "corrupted" in str(e)
        else:
            raise AssertionError("corrupted config accepted")
        with open(path) as f:
            assert f.read() == "{not json"


def test_template_aging_keeps_everything_aligned():
    with TempDir() as tmp:
        profile = register(tmp)
        live_raw = genuine_set(count=1, seed=4242, noise=0.0005)[0]
        live_pose = normalize_gesture(live_raw)
        # Raw-mode users must not age without the live raw recording.
        assert gc.update_templates_if_high_confidence(
            "alice", live_pose, 0.0, profile["threshold"], profile["templates"],
            templates_dir=tmp) is None
        result = gc.update_templates_if_high_confidence(
            "alice", live_pose, 0.0, profile["threshold"], profile["templates"],
            templates_dir=tmp, live_raw=live_raw)
        assert result is not None, "aging did not trigger for a near-ideal attempt"
        idx = result["replaced_template"]
        reloaded = gc.load_user_profile("alice", templates_dir=tmp)
        assert np.allclose(reloaded["templates"][idx - 1], live_pose)
        assert np.allclose(reloaded["raw_templates"][idx - 1], live_raw)
        assert reloaded["kinematic_profile"]["method"] == KINEMATIC_METHOD_SMOOTHED
        assert gc._read_config("alice", tmp)["last_template_update"] == idx
        # Impostor-grade confidence must never age.
        assert gc.update_templates_if_high_confidence(
            "alice", live_pose, 0.0, profile["threshold"], profile["templates"],
            templates_dir=tmp, live_raw=live_raw, anthropometric_confidence=0.5) is None


def test_evaluation_scripts_run():
    with TempDir() as tmp:
        register(tmp, username="evaluser")
        here = os.path.dirname(os.path.abspath(__file__))
        for script in ("evaluate_biometric_security.py", "evaluate_multimodal_security.py"):
            out = subprocess.run(
                [sys.executable, os.path.join(here, script), "--user", "evaluser",
                 "--templates-dir", tmp],
                capture_output=True, text=True, cwd=here, timeout=600,
            )
            assert out.returncode == 0, f"{script} failed:\n{out.stdout}\n{out.stderr}"
            assert "False Acceptance Rate" in out.stdout, out.stdout
        out = subprocess.run(
            [sys.executable, os.path.join(here, "evaluate_biometric_security.py"),
             "--user", "nobody", "--templates-dir", tmp],
            capture_output=True, text=True, cwd=here, timeout=120,
        )
        assert out.returncode != 0 and "not registered" in out.stdout + out.stderr


def test_ui_evaluate_attempt_and_dashboard():
    import gesture_auth as ga
    with TempDir() as tmp:
        profile = register(tmp)
        good = ga.evaluate_attempt(list(genuine_set(1, seed=31)[0]), [], ASPECT, profile)
        assert good["granted"], good["reason"]
        short = ga.evaluate_attempt(list(genuine_set(1)[0][:5]), [], ASPECT, profile)
        assert not short["granted"] and short["reason"] == "too_few_frames"
        assert short["distance"] is None and short["face_match"] is None
        cam = np.zeros((480, 640, 3), np.uint8)
        for res in (good, short):
            ga.build_dashboard(cam, state="GRANTED" if res["granted"] else "DENIED",
                               username="alice", threshold=profile["threshold"],
                               num_templates=5, face_info=ga.face_info_from_result(res, True),
                               gesture_info=ga.gesture_info_from_result(res),
                               hand_info=ga.hand_info_from_result(res),
                               result_info={"granted": res["granted"],
                                            "fused_score": res["fused_score"] or 0.0,
                                            "reason": res["reason"]})
        for fail in ("no_face_detected", "unstable_face_match", "lookalike_sibling_detected",
                     "impostor_face_identity"):
            ga.build_dashboard(cam, face_info={"match": False, "score": 0.2, "fail": fail,
                                               "enrolled": True})


def test_models_and_mediapipe_load():
    import cv2
    from utils.face_auth import detect_primary_face, verify_face_over_frames
    from gesture_capture import setup_mediapipe
    blank = np.zeros((480, 640, 3), np.uint8)
    assert detect_primary_face(blank) == (None, None)
    res = verify_face_over_frames([blank] * 3, np.ones(128, np.float32))
    assert res["reason"] == "no_face_detected" and res["match"] is False
    hands, *_ = setup_mediapipe()
    try:
        out = hands.process(cv2.cvtColor(blank, cv2.COLOR_BGR2RGB))
        assert out.multi_hand_landmarks is None
    finally:
        hands.close()


ALL = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    wanted = sys.argv[1:]
    tests = [t for t in ALL if not wanted or any(w in t.__name__ for w in wanted)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:                            # report and keep going
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
