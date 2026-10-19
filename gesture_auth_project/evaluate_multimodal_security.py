"""
Multimodal Biometric Benchmark (N = 180 synthetic trials)

Compares 4 architectural configurations across 6 threat cohorts:
  1. Macro Gesture Only (DTW + Finger State + Transition + Segment gates)
  2. Gesture + Hand Biometrics (anthropometrics + kinematics)
  3. Face Only (SFace cosine vs the operational threshold)
  4. Full Multimodal Cascade (face anchor + gesture + hand biometrics)

Cohorts (30 trials each):
  1. Genuine user            - genuine face,    genuine gesture variations
  2. Zero-effort impostor     - stranger face,   random cohort gestures
  3. Shoulder-surfer          - stranger face,   copied gesture, distinct hand
  4. Similar hand (sibling)   - stranger face,   copied gesture, hand +-5-10%
  5. 2D photo + copied gesture- genuine face photo, copied gesture, attacker's hand
  6. Sibling look-alike       - face in the 0.40-0.53 look-alike band,
                                copied gesture, similar hand

IMPORTANT: every trial is synthetic. Face embeddings are random unit vectors
placed at a chosen cosine from the enrolled vector (not SFace outputs): the
cosine ranges below are ASSUMPTIONS, not measurements. Gestures are
perturbations of the enrolled templates; the copied-gesture cohorts are a
PERFECT-REPLAY attacker (exact enrolled motion), far stronger than a human
observer. Results show how the decision logic behaves, not real-world error
rates. Liveness challenges are not part of this benchmark (they run before
evaluate_attempt in the live system), so cohort 5 depends on the hand checks.

Semester-2 corrections: rates per attack with 95 % Clopper-Pearson intervals
(no pooled FAR / HTER); genuine probes are checked against held-out profiles
(the template a probe was made from is left out); the similar-hand pause
survives the noise (see evaluate_biometric_security.py).

Usage:
    python evaluate_multimodal_security.py [--user NAME] [--templates-dir DIR]
"""

import numpy as np

from gesture_compare import authenticate_with_details
from cohort_library import generate_cohort_library
from evaluate_biometric_security import (
    load_benchmark_user,
    held_out_profile,
    rate_text,
    generate_genuine_variations,
    generate_distinct_hand_impostors,
    generate_similar_hand_impostors,
)
from utils.face_auth import classify_face_score

N_TRIALS = 30

# ASSUMED cosine ranges per face cohort (not SFace measurements). Genuine
# live faces are assumed to score 0.62-0.85 against their enrollment; a
# printed photo of the user slightly lower; siblings in the look-alike band
# [0.40, 0.53).
GENUINE_COSINE = (0.62, 0.85)
PHOTO_COSINE = (0.58, 0.78)
LOOKALIKE_COSINE = (0.41, 0.52)


def random_unit_embedding(rng, dim=128):
    v = rng.randn(dim)
    return v / np.linalg.norm(v)


def embedding_at_cosine(base, cosine, rng):
    """Unit vector whose cosine similarity with unit vector `base` is exactly `cosine`."""
    noise = rng.randn(base.shape[0])
    orth = noise - np.dot(noise, base) * base
    orth /= np.linalg.norm(orth)
    return cosine * base + np.sqrt(max(0.0, 1.0 - cosine ** 2)) * orth


def face_cohort(base, n, rng, cosine_range=None):
    if cosine_range is None:                     # unrelated strangers (~0 cosine)
        return [random_unit_embedding(rng) for _ in range(n)]
    lo, hi = cosine_range
    return [embedding_at_cosine(base, rng.uniform(lo, hi), rng) for _ in range(n)]


def evaluate_multimodal_trial(face_emb_live, hand_sample, enrolled_face, user):
    """Evaluate one attempt under all 4 configurations (user = profile dict)."""
    face_sim = float(np.dot(enrolled_face, face_emb_live))
    face_passed, face_conf, face_reason = classify_face_score(face_sim)

    granted_full, _, _, details = authenticate_with_details(
        hand_sample, user["templates"], user["threshold"],
        user["fst"], user["tt"], user["st"],
        user["anthro_prof"], user["kin_prof"],
        face_match=face_passed,
        face_confidence=face_conf,
        face_details={"score": face_sim, "reason": face_reason},
    )

    macro_passed = details["passes_macro"]
    return {
        "macro_passed": macro_passed,
        "gesture_bio_passed": macro_passed and details["passes_biometric"],
        "face_passed": face_passed,
        "multimodal_passed": granted_full,
        "failure_reason": details.get("failure_reason"),
        "face_sim": face_sim,
    }


def main():
    user = load_benchmark_user(
        description="WaveLock multimodal synthetic benchmark (180 trials)"
    )
    templates = user["templates"]

    print("=" * 88)
    print(f"  WAVELOCK MULTIMODAL BIOMETRIC BENCHMARK (N = {6 * N_TRIALS} synthetic trials)")
    print("=" * 88)
    print(f"Target User: '{user['username']}' ({len(templates)} templates)")
    print("NOTE: SIMULATED faces (assumed cosines) and gestures; copied gestures = perfect replay;")
    print("      liveness challenges and the wrist-path gate are not exercised.")
    print()

    rng = np.random.RandomState(2026)
    master_face = random_unit_embedding(rng)

    cohort_lib = generate_cohort_library()
    random_gestures = (cohort_lib * (N_TRIALS // len(cohort_lib) + 1))[:N_TRIALS]

    cohorts = {
        "1. Genuine User (Face + Gesture)": (
            face_cohort(master_face, N_TRIALS, rng, GENUINE_COSINE),
            generate_genuine_variations(templates, count=N_TRIALS, seed=5555), True),
        "2. Zero-Effort Impostor": (
            face_cohort(master_face, N_TRIALS, rng),
            random_gestures, False),
        "3. Perfect replay, distinct hand": (
            face_cohort(master_face, N_TRIALS, rng),
            generate_distinct_hand_impostors(templates, count=N_TRIALS, seed=6666), False),
        "4. Perfect replay, similar hand": (
            face_cohort(master_face, N_TRIALS, rng),
            generate_similar_hand_impostors(templates, count=N_TRIALS, seed=7777), False),
        "5. Photo + perfect replay": (
            face_cohort(master_face, N_TRIALS, rng, PHOTO_COSINE),
            generate_distinct_hand_impostors(templates, count=N_TRIALS, seed=9999), False),
        "6. Look-alike + perfect replay": (
            face_cohort(master_face, N_TRIALS, rng, LOOKALIKE_COSINE),
            generate_similar_hand_impostors(templates, count=N_TRIALS, seed=8888), False),
    }

    results = {}
    for name, (faces, hands, is_genuine) in cohorts.items():
        counts = {"macro": 0, "hand_bio": 0, "face": 0, "multi": 0}
        sims = []
        for k, (f, h) in enumerate(zip(faces, hands)):
            # genuine variation k comes from template k % n: hold that one out
            prof = held_out_profile(user, k % len(templates)) if is_genuine else user
            res = evaluate_multimodal_trial(f, h, master_face, prof)
            counts["macro"] += res["macro_passed"]
            counts["hand_bio"] += res["gesture_bio_passed"]
            counts["face"] += res["face_passed"]
            counts["multi"] += res["multimodal_passed"]
            sims.append(res["face_sim"])
        results[name] = {"total": len(faces), "is_genuine": is_genuine,
                         "sim_range": (min(sims), max(sims)), **counts}

    print(f"{'Cohort (SIMULATED)':<34} | {'Face cos':<11} | {'Macro Only':<20} | "
          f"{'Hand Bio':<20} | {'Face Only':<20} | {'Multimodal':<20}")
    print("-" * 140)
    for name, d in results.items():
        t = d["total"]
        cells = [rate_text(d[k], t) for k in ("macro", "hand_bio", "face", "multi")]
        lo, hi = d["sim_range"]
        print(f"{name:<34} | {lo:5.2f}-{hi:4.2f} | " + " | ".join(f"{c:<20}" for c in cells))
    print("=" * 140)

    gen = results["1. Genuine User (Face + Gesture)"]
    print("\n  ACCEPTANCE RATES PER ATTACK (SIMULATED; each attack reported separately):")
    print(f"  Genuine false rejection rate (held-out probes), multimodal: "
          f"{rate_text(gen['total'] - gen['multi'], gen['total'])}")
    for name, d in results.items():
        if d["is_genuine"]:
            continue
        print(f"  {name:<34} accepted (multimodal): {rate_text(d['multi'], d['total'])}")
    print("  Face cosines are assumed; copied gestures are a perfect replay. Validate with")
    print("  evaluate_real_recordings.py (real recordings, zero-effort impostors).")
    print("=" * 140)


if __name__ == "__main__":
    main()
