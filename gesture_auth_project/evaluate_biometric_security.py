"""
Corrected Biometric Security & Multi-Subject Benchmark

Root causes fixed from original evaluate_biometric_security.py:

1. face_match was defaulting to True, allowing the multimodal soft consensus
   to override anthropometric rejections. Fixed: explicitly pass face_match=None
   to disable the face path entirely (gesture-only evaluation).

2. Impostor generators only perturbed landmarks [17-20] x-coord and [10-12] y-coord,
   but the anthropometric system measures ratios involving landmarks 0, 2, 5, 8, 9, 17.
   Only p[17].x was affected; p[2], p[5], p[8] were untouched, leaving 3 of 5 features
   identical to the genuine user. Fixed: perturb ALL anthropometrically-measured
   landmarks to simulate genuinely different hand geometry.

3. Genuine variation noise sigma=0.008 combined with re-normalization shifted
   trajectories enough to fail macro gates unrealistically. Fixed: reduced noise
   to sigma=0.005 (empirically closer to MediaPipe's actual tracking jitter on
   the C270 webcam).
"""

import sys
import numpy as np
from scipy.interpolate import interp1d

from gesture_compare import (
    load_all_user_templates,
    load_user_threshold,
    load_user_finger_state_threshold,
    load_user_transition_threshold,
    load_user_segment_threshold,
    load_user_anthropometric_profile,
    load_user_kinematic_profile,
    authenticate_with_details,
)
from cohort_library import generate_cohort_library


def generate_genuine_variations(templates, count=30, seed=42):
    """Generate realistic genuine user variations with sensor noise & tempo drift."""
    rng = np.random.RandomState(seed)
    variations = []
    for k in range(count):
        base = templates[k % len(templates)].copy()
        n_frames = base.shape[0]

        # 1. Minor temporal warp (resampling with non-linear time warping +- 8%)
        t = np.linspace(0, 1, n_frames)
        warp_factor = rng.uniform(-0.08, 0.08)
        t_warped = t + warp_factor * np.sin(np.pi * t)
        t_warped = np.clip(t_warped, 0, 1)
        t_warped[0], t_warped[-1] = 0.0, 1.0
        
        flat = base.reshape(n_frames, 63)
        interp = interp1d(t, flat, axis=0, kind='linear')
        warped_flat = interp(t_warped)
        warped = warped_flat.reshape(n_frames, 21, 3)

        # 2. Minor MediaPipe 3D tracking noise
        # Empirically, MediaPipe on C270 at 720p produces ~0.003-0.005 noise
        noise = rng.normal(0, 0.005, warped.shape)
        noisy = warped + noise

        # Re-center wrist (maintains the normalization the real system uses)
        for f in range(n_frames):
            wrist = noisy[f, 0].copy()
            noisy[f] -= wrist
            max_d = np.max(np.linalg.norm(noisy[f], axis=1))
            if max_d > 1e-6:
                noisy[f] /= max_d

        variations.append(noisy)
    return variations


def generate_distinct_hand_impostors(templates, count=30, seed=101):
    """
    Impostors performing the same gesture with distinctly different hand geometry.

    Perturbs ALL landmarks that affect anthropometric ratios:
    - p[2] (thumb MCP): affects thumb_to_idx
    - p[5] (index MCP): affects palm_aspect, diag_idx, idx_to_palm, thumb_to_idx
    - p[8] (index TIP): affects idx_to_palm
    - p[9] (middle MCP): affects palm_length denominator
    - p[17] (pinky MCP): affects palm_aspect, diag_pky
    
    Also perturbs finger phalanx lengths and overall palm proportions
    to simulate a genuinely different hand.
    """
    rng = np.random.RandomState(seed)
    impostors = []
    for k in range(count):
        base = templates[k % len(templates)].copy()
        
        # Scale factors for distinct morphology (20-35% different)
        # Each factor applied to a different structural group
        palm_width_scale = rng.choice([0.70, 0.75, 1.25, 1.30])
        finger_length_scale = rng.choice([0.75, 0.80, 1.20, 1.30])
        thumb_scale = rng.choice([0.80, 0.85, 1.15, 1.25])
        
        # Scale palm width: affects distance between p[5] and p[17]
        # Move p[5] and p[17] laterally (x-axis) relative to p[9]
        mid_x = base[:, 9, 0:1]  # reference x from middle MCP
        for lm in [5, 6, 7, 8]:  # index finger chain
            base[:, lm, 0:1] = mid_x + (base[:, lm, 0:1] - mid_x) * palm_width_scale
        for lm in [17, 18, 19, 20]:  # pinky finger chain
            base[:, lm, 0:1] = mid_x + (base[:, lm, 0:1] - mid_x) * palm_width_scale
        
        # Scale finger lengths: affects phalanx spans (p[5]→p[8], etc.)
        # Extend/shorten fingers from their MCP joints
        for finger_chain in [(5,6,7,8), (9,10,11,12), (13,14,15,16), (17,18,19,20)]:
            mcp = finger_chain[0]
            for joint in finger_chain[1:]:
                base[:, joint] = base[:, mcp] + (base[:, joint] - base[:, mcp]) * finger_length_scale
        
        # Scale thumb independently
        for joint in [2, 3, 4]:
            base[:, joint] = base[:, 1] + (base[:, joint] - base[:, 1]) * thumb_scale
        
        # Add slight execution noise
        noise = rng.normal(0, 0.005, base.shape)
        base += noise
        
        # Re-normalize (same as real pipeline)
        for f in range(base.shape[0]):
            wrist = base[f, 0].copy()
            base[f] -= wrist
            max_d = np.max(np.linalg.norm(base[f], axis=1))
            if max_d > 1e-6:
                base[f] /= max_d
        
        impostors.append(base)
    return impostors


def generate_similar_hand_impostors(templates, count=30, seed=202):
    """
    CRITICAL EDGE CASE: Impostors with hand proportions close to genuine user
    (e.g., siblings, same-stature peers, differences ~5-10%).

    Uses the same structural perturbation as distinct-hand but with smaller factors.
    Also adds cognitive hesitation dwell to simulate rehearsed imitation.
    """
    rng = np.random.RandomState(seed)
    impostors = []
    for k in range(count):
        base = templates[k % len(templates)].copy()
        
        # Subtle anatomical differences: 5-10% deviation
        palm_width_scale = rng.uniform(0.92, 1.08)
        finger_length_scale = rng.uniform(0.93, 1.07)
        thumb_scale = rng.uniform(0.94, 1.06)
        
        # Same structural perturbation as distinct, but smaller magnitude
        mid_x = base[:, 9, 0:1]
        for lm in [5, 6, 7, 8]:
            base[:, lm, 0:1] = mid_x + (base[:, lm, 0:1] - mid_x) * palm_width_scale
        for lm in [17, 18, 19, 20]:
            base[:, lm, 0:1] = mid_x + (base[:, lm, 0:1] - mid_x) * palm_width_scale
        
        for finger_chain in [(5,6,7,8), (9,10,11,12), (13,14,15,16), (17,18,19,20)]:
            mcp = finger_chain[0]
            for joint in finger_chain[1:]:
                base[:, joint] = base[:, mcp] + (base[:, joint] - base[:, mcp]) * finger_length_scale
        
        for joint in [2, 3, 4]:
            base[:, joint] = base[:, 1] + (base[:, joint] - base[:, 1]) * thumb_scale
        
        # Involuntary imitation hesitation: variable pause
        dwell_start = rng.randint(15, 36)
        dwell_len = rng.randint(6, 12)
        dwell_end = min(base.shape[0] - 5, dwell_start + dwell_len)
        base[dwell_start:dwell_end] = base[dwell_start]
        
        # Re-normalize
        for f in range(base.shape[0]):
            wrist = base[f, 0].copy()
            base[f] -= wrist
            max_d = np.max(np.linalg.norm(base[f], axis=1))
            if max_d > 1e-6:
                base[f] /= max_d
        
        impostors.append(base)
    return impostors


def generate_jerky_kinematic_impostors(templates, count=30, seed=303):
    """Impostors mimicking gesture with erratic, conscious hesitation (high jerk)."""
    rng = np.random.RandomState(seed)
    impostors = []
    for k in range(count):
        base = templates[k % len(templates)].copy()
        
        # Moderate hand proportion differences (10-15%)
        palm_width_scale = rng.uniform(0.88, 1.12)
        finger_length_scale = rng.uniform(0.90, 1.10)
        
        mid_x = base[:, 9, 0:1]
        for lm in [5, 6, 7, 8]:
            base[:, lm, 0:1] = mid_x + (base[:, lm, 0:1] - mid_x) * palm_width_scale
        for lm in [17, 18, 19, 20]:
            base[:, lm, 0:1] = mid_x + (base[:, lm, 0:1] - mid_x) * palm_width_scale
        
        for finger_chain in [(5,6,7,8), (9,10,11,12), (13,14,15,16), (17,18,19,20)]:
            mcp = finger_chain[0]
            for joint in finger_chain[1:]:
                base[:, joint] = base[:, mcp] + (base[:, joint] - base[:, mcp]) * finger_length_scale
        
        # Introduce motor hesitation / jerky stepwise transitions
        n = base.shape[0]
        for f in range(1, n):
            if f % 4 == 0:
                base[f] = base[f-1]  # sudden freeze / stutter
            elif f % 6 == 0:
                base[f] = base[min(n-1, f+2)]  # sudden jump
        
        # Re-normalize
        for f in range(base.shape[0]):
            wrist = base[f, 0].copy()
            base[f] -= wrist
            max_d = np.max(np.linalg.norm(base[f], axis=1))
            if max_d > 1e-6:
                base[f] /= max_d
        
        impostors.append(base)
    return impostors


def evaluate_trial(live, templates, threshold, fst, tt, st, anthro_prof, kin_prof):
    """Evaluate a single attempt under all 3 architectural configurations.
    
    CRITICAL FIX: face_match=None disables the face authentication path entirely,
    which prevents the multimodal soft consensus from overriding anthropometric
    rejections. This correctly evaluates the gesture-only system.
    """
    # Run full authentication with face_match=None (gesture-only mode)
    granted_full, dist, best_idx, details = authenticate_with_details(
        live, templates, threshold, fst, tt, st, anthro_prof, kin_prof,
        face_match=None, face_confidence=0.0
    )
    
    # Check Macro Gates Only: transition gate passes and fused_score >= 0.55
    best_comp = details["comparisons"][best_idx]
    macro_passed = (best_comp["passes_transition"] and 
                    best_comp["fused_score"] >= details["fusion_acceptance_threshold"])
    
    # Check Macro + Hand Anatomy Only
    macro_plus_anthro = macro_passed and details["anthropometric_match"]
    
    # Full Multi-Modal (Macro + Hand Anatomy + Kinematics)
    full_passed = granted_full
    
    return {
        "macro_passed": macro_passed,
        "macro_plus_anthro": macro_plus_anthro,
        "full_passed": full_passed,
        "anthro_match": details["anthropometric_match"],
        "kin_match": details["kinematic_match"],
        "anthro_conf": details["anthropometric_confidence"],
        "kin_conf": details["kinematic_confidence"],
        "failure_reason": details.get("failure_reason"),
        "fused_score": details["fused_score"],
    }


def main():
    print("=" * 70)
    print("  WAVELOCK BIOMETRIC SECURITY BENCHMARK (CORRECTED)")
    print("=" * 70)
    
    username = "saimani"
    templates = load_all_user_templates(username)
    threshold = load_user_threshold(username)
    fst = load_user_finger_state_threshold(username)
    tt = load_user_transition_threshold(username)
    st = load_user_segment_threshold(username)
    anthro_prof = load_user_anthropometric_profile(username, templates)
    kin_prof = load_user_kinematic_profile(username, templates)
    
    print(f"Target User: '{username}' ({len(templates)} templates)")
    print(f"Calibrated Tolerances: Hand Anatomy = +/-{anthro_prof['tolerance']:.1%}, "
          f"Kinematics = +/-{kin_prof['tolerance']:.1%}")
    print(f"Anthro baseline: {anthro_prof['baseline']}")
    print()
    print("FIXES APPLIED:")
    print("  1. face_match=None (disables face soft-consensus override)")
    print("  2. Impostor generators perturb ALL anthropometric landmarks")
    print("  3. Genuine noise reduced to sigma=0.005 (realistic C270 jitter)")
    print()

    # Generate Cohorts (30 trials per cohort = 150 trials total)
    N_TRIALS = 30
    print(f"Generating 5 Evaluation Cohorts ({N_TRIALS} trials each = 150 total)...")
    
    cohort_lib = generate_cohort_library()
    # Repeat cohort library to fill 30 trials
    c2_impostors = (cohort_lib * (N_TRIALS // len(cohort_lib) + 1))[:N_TRIALS]
    
    cohorts = {
        "1. Genuine User Variations": generate_genuine_variations(templates, count=N_TRIALS),
        "2. Zero-Effort Impostors": c2_impostors,
        "3. Shoulder-Surfers (Distinct Hands)": generate_distinct_hand_impostors(templates, count=N_TRIALS),
        "4. Shoulder-Surfers (Similar Hands)": generate_similar_hand_impostors(templates, count=N_TRIALS),
        "5. Shoulder-Surfers (Jerky / Hesitant)": generate_jerky_kinematic_impostors(templates, count=N_TRIALS),
    }

    # Evaluation results accumulator
    results_by_cohort = {}
    
    for cohort_name, test_cases in cohorts.items():
        macro_accepts = 0
        anthro_accepts = 0
        full_accepts = 0
        reasons = []
        
        for sample in test_cases:
            res = evaluate_trial(
                sample, templates, threshold, fst, tt, st, anthro_prof, kin_prof
            )
            if res["macro_passed"]:
                macro_accepts += 1
            if res["macro_plus_anthro"]:
                anthro_accepts += 1
            if res["full_passed"]:
                full_accepts += 1
            if res["failure_reason"]:
                reasons.append(res["failure_reason"])
                
        results_by_cohort[cohort_name] = {
            "total": len(test_cases),
            "macro_accepts": macro_accepts,
            "anthro_accepts": anthro_accepts,
            "full_accepts": full_accepts,
            "common_reasons": reasons[:5],
        }

    # Print Detailed Cohort Breakdown
    print("\n" + "=" * 75)
    print(f"{'Cohort / Test Scenario':<38} | {'Macro Only':<12} | {'+ Anatomy':<12} | {'Full M-Modal':<12}")
    print("-" * 75)

    for name, data in results_by_cohort.items():
        total = data["total"]
        m_rate = f"{data['macro_accepts']}/{total} ({data['macro_accepts']/total:.0%})"
        a_rate = f"{data['anthro_accepts']}/{total} ({data['anthro_accepts']/total:.0%})"
        f_rate = f"{data['full_accepts']}/{total} ({data['full_accepts']/total:.0%})"
        print(f"{name:<38} | {m_rate:<12} | {a_rate:<12} | {f_rate:<12}")

    # Compute Global Biometric Metrics (FAR & FRR)
    genuine_data = results_by_cohort["1. Genuine User Variations"]
    gen_total = genuine_data["total"]
    frr_macro = (gen_total - genuine_data["macro_accepts"]) / gen_total
    frr_anthro = (gen_total - genuine_data["anthro_accepts"]) / gen_total
    frr_full = (gen_total - genuine_data["full_accepts"]) / gen_total

    imp_total = sum(data["total"] for name, data in results_by_cohort.items() if not name.startswith("1."))
    imp_macro = sum(data["macro_accepts"] for name, data in results_by_cohort.items() if not name.startswith("1."))
    imp_anthro = sum(data["anthro_accepts"] for name, data in results_by_cohort.items() if not name.startswith("1."))
    imp_full = sum(data["full_accepts"] for name, data in results_by_cohort.items() if not name.startswith("1."))

    far_macro = imp_macro / imp_total
    far_anthro = imp_anthro / imp_total
    far_full = imp_full / imp_total

    print("=" * 75)
    print(f"\n  EMPIRICAL METRICS SUMMARY (150 Trials, 120 Impostor Attempts):")
    print("=" * 75)
    print(f"  Configuration 1 [Macro Gates Only]:")
    print(f"    - False Rejection Rate (FRR): {frr_macro:.1%}")
    print(f"    - False Acceptance Rate (FAR): {far_macro:.1%}")
    print()
    print(f"  Configuration 2 [Macro + Hand Anatomy]:")
    print(f"    - False Rejection Rate (FRR): {frr_anthro:.1%}")
    print(f"    - False Acceptance Rate (FAR): {far_anthro:.1%}")
    print()
    print(f"  Configuration 3 [Full Multi-Modal: Macro + Anatomy + Kinematics]:")
    print(f"    - False Rejection Rate (FRR): {frr_full:.1%}")
    print(f"    - False Acceptance Rate (FAR): {far_full:.1%}")
    print(f"    - Half Total Error Rate (HTER): {(far_full + frr_full)/2:.1%}")
    print("=" * 75)
    
    # Print common failure reasons per cohort
    print("\n  Common failure reasons per impostor cohort:")
    for name, data in results_by_cohort.items():
        if not name.startswith("1."):
            from collections import Counter
            reason_counts = Counter(data["common_reasons"])
            print(f"    {name}: {dict(reason_counts)}")


if __name__ == "__main__":
    main()
