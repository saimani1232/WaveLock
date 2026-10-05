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
placed at a chosen cosine from the enrolled vector (not SFace outputs), and
gestures are perturbations of the enrolled templates. Results show how the
decision logic behaves, not real-world FAR/FRR; there is also no liveness
detection, so cohort 5's outcome depends entirely on the hand biometrics.

Usage:
    python evaluate_multimodal_security.py [--user NAME] [--templates-dir DIR]
"""

import numpy as np

from gesture_compare import authenticate_with_details
from cohort_library import generate_cohort_library
from evaluate_biometric_security import (
    load_benchmark_user,
    generate_genuine_variations,
    generate_distinct_hand_impostors,
    generate_similar_hand_impostors,
)
from utils.face_auth import classify_face_score

N_TRIALS = 30

# Cosine ranges per face cohort. Genuine live faces typically score
# 0.62-0.85 against their enrollment; a printed photo of the user slightly
# lower; siblings fall in the look-alike band [0.40, 0.53).
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
    """Evaluate one attempt under all 4 configurations."""
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
    print("NOTE: synthetic faces/gestures; no liveness detection; wrist-path gate not exercised.")
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
        "3. Shoulder-Surfer (Distinct Hand)": (
            face_cohort(master_face, N_TRIALS, rng),
            generate_distinct_hand_impostors(templates, count=N_TRIALS, seed=6666), False),
        "4. Similar Hand (Stranger Face)": (
            face_cohort(master_face, N_TRIALS, rng),
            generate_similar_hand_impostors(templates, count=N_TRIALS, seed=7777), False),
        "5. 2D Photo + Copied Gesture": (
            face_cohort(master_face, N_TRIALS, rng, PHOTO_COSINE),
            generate_distinct_hand_impostors(templates, count=N_TRIALS, seed=9999), False),
        "6. Sibling Look-alike Attack": (
            face_cohort(master_face, N_TRIALS, rng, LOOKALIKE_COSINE),
            generate_similar_hand_impostors(templates, count=N_TRIALS, seed=8888), False),
    }

    results = {}
    for name, (faces, hands, is_genuine) in cohorts.items():
        counts = {"macro": 0, "hand_bio": 0, "face": 0, "multi": 0}
        sims = []
        for f, h in zip(faces, hands):
            res = evaluate_multimodal_trial(f, h, master_face, user)
            counts["macro"] += res["macro_passed"]
            counts["hand_bio"] += res["gesture_bio_passed"]
            counts["face"] += res["face_passed"]
            counts["multi"] += res["multimodal_passed"]
            sims.append(res["face_sim"])
        results[name] = {"total": len(faces), "is_genuine": is_genuine,
                         "sim_range": (min(sims), max(sims)), **counts}

    print(f"{'Threat Scenario / Cohort':<36} | {'Face cos':<11} | {'Macro Only':<11} | "
          f"{'Hand Bio':<11} | {'Face Only':<11} | {'Multimodal':<11}")
    print("-" * 110)
    for name, d in results.items():
        t = d["total"]
        cells = [f"{d[k]}/{t} ({d[k] / t:.0%})" for k in ("macro", "hand_bio", "face", "multi")]
        lo, hi = d["sim_range"]
        print(f"{name:<36} | {lo:.2f}-{hi:.2f}   | " + " | ".join(f"{c:<11}" for c in cells))
    print("=" * 110)

    gen = results["1. Genuine User (Face + Gesture)"]
    impostors = [d for d in results.values() if not d["is_genuine"]]
    imp_total = sum(d["total"] for d in impostors)

    print(f"\n  METRICS SUMMARY ({sum(d['total'] for d in results.values())} trials, "
          f"{imp_total} impostor attempts):")
    for label, key in (("Macro Gesture Only", "macro"),
                       ("Gesture + Hand Biometrics", "hand_bio"),
                       ("Face Recognition Alone", "face"),
                       ("Full Multimodal Cascade", "multi")):
        frr = (gen["total"] - gen[key]) / gen["total"]
        far = sum(d[key] for d in impostors) / imp_total
        print(f"  {label}:")
        print(f"    - False Rejection Rate (FRR):  {frr:.1%}")
        print(f"    - False Acceptance Rate (FAR): {far:.1%}")
        print(f"    - Half Total Error Rate (HTER): {(far + frr) / 2:.1%}")
    print("=" * 110)


if __name__ == "__main__":
    main()
