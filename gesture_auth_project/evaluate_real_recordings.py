"""
WaveLock evaluation on the REAL enrolment recordings (MEASURED, read-only).

Every registered profile keeps its five raw recordings (gesture_N_raw.npy).
They are the only real data available, so they are used as follows:

  Genuine (leave-one-out): each recording of user u is authenticated against
      a profile calibrated from u's OTHER four recordings.
  Zero-effort impostor: each recording of user v is authenticated against
      the profile of every other user u (all five recordings of u).

Every trial runs the production decision, gesture_auth.evaluate_attempt(),
on a temporary profile built by gesture_compare.save_registration(); nothing
in templates/ is written. The face stage is off (no face frames are stored
with the recordings), so these numbers describe the gesture + hand layers.

Also reported:
  - ablation on real trials: DTW only, gesture gates, + hand anatomy,
    + kinematics (= full gesture decision);
  - base-paper method (Shin et al. 2022, DTW-KNN): closed-set k-nearest-
    neighbour identification over all enrolled recordings; a claim is
    accepted when the k nearest templates vote for the claimed user;
  - equal error rate (EER) of the gesture fused score S_macro and of the
    DTW ratio (best DTW / user threshold), for reference only (the real
    decision is a conjunction of gates, not one score);
  - 95 % Clopper-Pearson intervals for every rate.

Limits (always printed): same-session recordings (genuine variation across
days is not covered); zero-effort impostors only (no imitation, photo or
replay attacks); few users.

Usage:
    python evaluate_real_recordings.py [--users a,b,c] [--variant NAME ...]
                                       [--templates-dir DIR] [--out results.json]
Variants (matcher methods to compare on the same data): see VARIANTS.
"""

import argparse
import json
import os
import sys
import tempfile
from collections import Counter

import numpy as np
from scipy.stats import beta

from gesture_compare import (
    TEMPLATES_DIR,
    list_registered_users,
    load_all_user_templates,
    load_user_raw_templates,
    save_registration,
    load_user_profile,
    compute_dtw_distance,
    _read_config,
    matching_recording,
    TEMPORAL_TRIM_METHOD,
)
from gesture_auth import evaluate_attempt
from gesture_capture import TARGET_FRAMES
from utils.normalize import normalize_gesture, aspect_for_method, POSE_METHOD_LEGACY, POSE_METHOD_ASPECT

# Profiles confirmed by the team as five different people (2026-10-16).
CONFIRMED_DISTINCT = ["lib", "res", "saimani", "test", "umesh"]

# name -> keyword arguments for save_registration (matcher method tags).
# Later semester-2 methods add their own entries here.
VARIANTS = {
    "current": {"pose_method": POSE_METHOD_LEGACY},
    "aspect": {"pose_method": POSE_METHOD_ASPECT},
    "trim": {"pose_method": POSE_METHOD_LEGACY, "temporal_trim": TEMPORAL_TRIM_METHOD},
    "relaxed": {"pose_method": POSE_METHOD_LEGACY, "check_level": "relaxed"},
    "trim_relaxed": {"pose_method": POSE_METHOD_LEGACY, "temporal_trim": TEMPORAL_TRIM_METHOD,
                     "check_level": "relaxed"},
}


# ---------------------------------------------------------------- statistics
def clopper_pearson(k, n, alpha=0.05):
    """Exact two-sided 95 % interval for k successes out of n (in %)."""
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (100 * lo, 100 * hi)


def rate(k, n):
    lo, hi = clopper_pearson(k, n)
    return {"k": int(k), "n": int(n), "pct": round(100 * k / n, 1) if n else None,
            "ci95": [round(lo, 1), round(hi, 1)]}


def eer(genuine_scores, impostor_scores, higher_is_genuine=True):
    """Equal error rate (%) and the threshold where FAR and FRR cross."""
    sign = 1.0 if higher_is_genuine else -1.0
    g = sign * np.asarray(genuine_scores, float)
    i = sign * np.asarray(impostor_scores, float)
    best = None                                   # (|FRR - FAR|, EER, threshold)
    for t in np.unique(np.concatenate([g, i])):
        frr = float(np.mean(g < t))               # genuine below the threshold: rejected
        far = float(np.mean(i >= t))              # impostor at or above: accepted
        if best is None or abs(frr - far) < best[0]:
            best = (abs(frr - far), (frr + far) / 2, sign * t)
    return round(100 * best[1], 1), float(best[2])


# ---------------------------------------------------------------- data
def load_recordings(user, templates_dir):
    n = len(load_all_user_templates(user, templates_dir))
    raws = load_user_raw_templates(user, n, templates_dir)
    if raws is None:
        return None
    aspect = float(_read_config(user, templates_dir).get("frame_aspect") or 4 / 3)
    return {"raws": raws, "aspect": aspect}


def build_profile(tmp, name, raws, aspect, variant):
    opts = dict(VARIANTS[variant])
    pose_method = opts.get("pose_method", POSE_METHOD_LEGACY)
    poses = [normalize_gesture(matching_recording(r, opts.get("temporal_trim")), TARGET_FRAMES,
                               frame_aspect=aspect_for_method(pose_method, aspect))
             for r in raws]
    save_registration(name, poses, templates_dir=tmp, raw_samples=raws, frame_aspect=aspect, **opts)
    profile = load_user_profile(name, tmp)
    profile["face_embedding"] = None            # no face frames exist for these recordings
    return profile


def run_trial(profile, raw, aspect):
    res = evaluate_attempt(list(raw), [], aspect, profile)
    d = res["details"] or {}
    comps = d.get("comparisons") or []
    dmin = min(c["distance"] for c in comps) if comps else float("inf")
    return {
        "granted": bool(res["granted"]),
        "reason": res["reason"],
        "dtw_only": bool(dmin <= profile["threshold"]),
        "macro": bool(d.get("passes_macro")),
        "anthro": bool(d.get("anthropometric_match")),
        "kin": bool(d.get("kinematic_match")),
        "fused": float(d.get("fused_score") or 0.0),
        "dtw_ratio": float(dmin / profile["threshold"]) if profile["threshold"] else float("inf"),
    }


# ---------------------------------------------------------------- base paper
def dtw_knn(gallery, probe_pose, exclude, k=3):
    """Labels of the k nearest gallery templates (DTW on pose sequences)."""
    dists = sorted((compute_dtw_distance(probe_pose, t), lab) for lab, idx, t in gallery
                   if (lab, idx) != exclude)
    return Counter(lab for _, lab in dists[:k]).most_common(1)[0][0]


# ---------------------------------------------------------------- main run
def evaluate(users, templates_dir, variant="current", log=print):
    data = {}
    for u in users:
        rec = load_recordings(u, templates_dir)
        if rec is None:
            log(f"  skipped {u}: no raw recordings (registered before they were stored)")
            continue
        data[u] = rec
    users = list(data)
    rows = []
    with tempfile.TemporaryDirectory(prefix="wavelock_eval_") as tmp:
        for u in users:
            raws, asp = data[u]["raws"], data[u]["aspect"]
            for i in range(len(raws)):
                keep = [r for k, r in enumerate(raws) if k != i]
                p = build_profile(tmp, f"{u}-loo{i}", keep, asp, variant)
                rows.append(dict(kind="genuine", target=u, probe=u, rec=i + 1, **run_trial(p, raws[i], asp)))
            p_full = build_profile(tmp, f"{u}-full", raws, asp, variant)
            for v in users:
                if v == u:
                    continue
                for j, raw in enumerate(data[v]["raws"]):
                    rows.append(dict(kind="impostor", target=u, probe=v, rec=j + 1,
                                     **run_trial(p_full, raw, data[v]["aspect"])))

    # base paper: DTW-KNN closed-set identification (variant-independent poses)
    gallery = []
    for u in users:
        for idx, raw in enumerate(data[u]["raws"]):
            gallery.append((u, idx, normalize_gesture(raw, TARGET_FRAMES)))
    for row in rows:
        probe_raw = data[row["probe"]]["raws"][row["rec"] - 1]
        label = dtw_knn(gallery, normalize_gesture(probe_raw, TARGET_FRAMES),
                        exclude=(row["probe"], row["rec"] - 1))
        row["dtw_knn"] = bool(label == row["target"])
    return users, rows


def summarise(rows, users):
    out = {}
    for kind in ("genuine", "impostor"):
        r = [x for x in rows if x["kind"] == kind]
        out[kind] = {key: rate(sum(x[key] for x in r), len(r))
                     for key in ("granted", "dtw_only", "macro", "dtw_knn")}
        out[kind]["macro+anthro"] = rate(sum(x["macro"] and x["anthro"] for x in r), len(r))
        out[kind]["macro+kin"] = rate(sum(x["macro"] and x["kin"] for x in r), len(r))
    g = [x for x in rows if x["kind"] == "genuine"]
    im = [x for x in rows if x["kind"] == "impostor"]
    out["eer_fused_score"] = eer([x["fused"] for x in g], [x["fused"] for x in im], True)
    out["eer_dtw_ratio"] = eer([x["dtw_ratio"] for x in g], [x["dtw_ratio"] for x in im], False)
    out["per_user"] = {u: {
        "genuine_granted": rate(sum(x["granted"] for x in g if x["target"] == u), sum(1 for x in g if x["target"] == u)),
        "impostor_granted": rate(sum(x["granted"] for x in im if x["target"] == u), sum(1 for x in im if x["target"] == u)),
    } for u in users}
    out["rejection_reasons"] = Counter(p for x in g if not x["granted"] for p in (x["reason"] or "").split("_and_"))
    return out


def fmt(r):
    return f"{r['k']:3d}/{r['n']:<3d} {r['pct']:5.1f} %  [{r['ci95'][0]:4.1f}, {r['ci95'][1]:5.1f}]"


def report(name, s):
    print(f"\n=== {name}")
    print(f"{'layer':<34}{'genuine accepted':<34}{'zero-effort impostor accepted'}")
    for key, label in (("dtw_only", "DTW only (AirAuth-style)"), ("macro", "gesture gates"),
                       ("macro+anthro", "gesture gates + hand anatomy"),
                       ("macro+kin", "gesture gates + kinematics"),
                       ("granted", "full gesture decision (WaveLock)"),
                       ("dtw_knn", "DTW-KNN, k=3 (Shin et al. 2022)")):
        print(f"{label:<34}{fmt(s['genuine'][key]):<34}{fmt(s['impostor'][key])}")
    print(f"EER: fused gesture score {s['eer_fused_score'][0]} %, DTW ratio {s['eer_dtw_ratio'][0]} % (reference only)")
    print("genuine rejections by failing check:", dict(s["rejection_reasons"]) or "none")
    for u, v in s["per_user"].items():
        print(f"  {u:<9} genuine {fmt(v['genuine_granted'])}   impostor {fmt(v['impostor_granted'])}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="WaveLock evaluation on the real enrolment recordings")
    ap.add_argument("--users", default=None, help="comma-separated users (default: all with raw recordings)")
    ap.add_argument("--variant", action="append", choices=sorted(VARIANTS), default=None,
                    help="matcher method(s) to evaluate (default: current)")
    ap.add_argument("--templates-dir", default=TEMPLATES_DIR)
    ap.add_argument("--out", default=None, help="write all trials and summaries as JSON")
    args = ap.parse_args(argv)

    users = args.users.split(",") if args.users else list_registered_users(args.templates_dir)
    variants = args.variant or ["current"]
    print("MEASURED on real enrolment recordings (same session, face stage off, zero-effort impostors only).")
    print(f"Users: {', '.join(users)}" + ("" if set(users) <= set(CONFIRMED_DISTINCT) else
          "   (only lib, res, saimani, test, umesh are confirmed to be different people)"))
    results = {}
    for v in variants:
        used, rows = evaluate(users, args.templates_dir, v)
        s = summarise(rows, used)
        report(f"variant: {v}", s)
        results[v] = {"users": used, "summary": s, "trials": rows}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1, default=lambda o: dict(o) if isinstance(o, Counter) else str(o))
        print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
