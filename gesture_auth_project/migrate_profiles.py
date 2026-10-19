"""
Migrate registered profiles to the aspect-corrected pose method.

Profiles created before the fix were normalised with x and y in different
units (see utils/normalize.py, POSE_METHOD_ASPECT). Every profile that kept
its raw recordings (gesture_N_raw.npy) can be rebuilt from them without
re-registering: the pose templates and all thresholds are recomputed and the
enrolled face is kept. A copy of the old profile is saved first in
templates/.backup_<user>_<timestamp>/ (ignored as a user by the system).

Profiles without raw recordings cannot be migrated; they keep working with
the legacy method until the user registers again.

Usage:
    python migrate_profiles.py [--user NAME] [--dry-run] [--templates-dir DIR]

Temporal trim (semester 2, see gesture_compare.TEMPORAL_TRIM_METHOD):
    python migrate_profiles.py --trim [--user NAME] [--dry-run]
recalibrates a profile from its raw recordings so still frames at the start
and end are ignored (same pose method; old profile backed up first).

Encryption at rest (utils/secure_store.py, Windows DPAPI):
    python migrate_profiles.py --encrypt [--user NAME]   encrypt existing .npy files in place
    python migrate_profiles.py --decrypt [--user NAME]   back to plain files (e.g. before moving
                                                         the profiles to another computer)
Every file is re-read and compared before it replaces the original. Encrypted
files can only be read by this Windows account on this computer.
"""

import os
import sys
import shutil
import argparse

import numpy as np
from datetime import datetime

from gesture_compare import (
    TEMPLATES_DIR,
    list_registered_users,
    normalize_username,
    load_all_user_templates,
    load_user_raw_templates,
    build_profile_config,
    save_registration,
    _read_config,
    matching_recording,
    TEMPORAL_TRIM_METHOD,
)
from gesture_capture import TARGET_FRAMES
from utils.normalize import normalize_gesture, aspect_for_method, POSE_METHOD_ASPECT, POSE_METHOD_LEGACY
from utils import secure_store

SHOWN_KEYS = ("threshold", "finger_state_threshold", "transition_threshold",
              "segment_threshold", "trajectory_threshold")


def migrate_user(username, templates_dir=TEMPLATES_DIR, dry_run=False):
    """
    Rebuild one profile with POSE_METHOD_ASPECT.

    Returns:
        dict with "status" ("migrated", "would_migrate", "already_current"
        or "no_raw_recordings"), the old and new threshold values, and the
        backup folder when one was written.
    """
    username = normalize_username(username)
    config = _read_config(username, templates_dir)
    old_method = config.get("pose_method", POSE_METHOD_LEGACY)
    if old_method == POSE_METHOD_ASPECT:
        return {"user": username, "status": "already_current"}

    n_templates = len(load_all_user_templates(username, templates_dir))
    raws = load_user_raw_templates(username, n_templates, templates_dir)
    if raws is None:
        return {"user": username, "status": "no_raw_recordings"}

    aspect = float(config.get("frame_aspect") or 1.0)
    trim = config.get("temporal_trim")
    poses = [normalize_gesture(matching_recording(r, trim), TARGET_FRAMES, frame_aspect=aspect)
             for r in raws]
    new_config = build_profile_config(poses, raws, aspect, POSE_METHOD_ASPECT, trim)

    old_anthro = (config.get("anthropometric_profile") or {}).get("tolerance")
    report = {
        "user": username,
        "old": {k: config.get(k) for k in SHOWN_KEYS},
        "new": {k: new_config.get(k) for k in SHOWN_KEYS},
        "old_anthro_tolerance": old_anthro,
        "new_anthro_tolerance": new_config["anthropometric_profile"]["tolerance"],
        "backup": None,
    }
    if dry_run:
        report["status"] = "would_migrate"
        return report

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = os.path.join(templates_dir, f".backup_{username}_{stamp}")
    shutil.copytree(os.path.join(templates_dir, username), backup)
    # face_embedding=None keeps the enrolled face (see save_registration).
    save_registration(username, poses, templates_dir=templates_dir,
                      raw_samples=raws, frame_aspect=aspect,
                      pose_method=POSE_METHOD_ASPECT, temporal_trim=trim)
    report["status"] = "migrated"
    report["backup"] = backup
    return report


def retrim_user(username, templates_dir=TEMPLATES_DIR, dry_run=False):
    """
    Recalibrate one profile with TEMPORAL_TRIM_METHOD from its raw recordings
    (pose method and face kept). Template-aging anchor and log start afresh.

    Returns:
        dict like migrate_user() ("retrimmed", "would_retrim",
        "already_current" or "no_raw_recordings").
    """
    username = normalize_username(username)
    config = _read_config(username, templates_dir)
    if config.get("temporal_trim") == TEMPORAL_TRIM_METHOD:
        return {"user": username, "status": "already_current"}
    n_templates = len(load_all_user_templates(username, templates_dir))
    raws = load_user_raw_templates(username, n_templates, templates_dir)
    if raws is None:
        return {"user": username, "status": "no_raw_recordings"}

    pose_method = config.get("pose_method", POSE_METHOD_LEGACY)
    aspect = float(config.get("frame_aspect") or 1.0)
    pose_aspect = aspect_for_method(pose_method, aspect)
    poses = [normalize_gesture(matching_recording(r, TEMPORAL_TRIM_METHOD), TARGET_FRAMES,
                               frame_aspect=pose_aspect) for r in raws]
    new_config = build_profile_config(poses, raws, aspect, pose_method, TEMPORAL_TRIM_METHOD)
    report = {
        "user": username,
        "old": {k: config.get(k) for k in SHOWN_KEYS},
        "new": {k: new_config.get(k) for k in SHOWN_KEYS},
        "old_anthro_tolerance": (config.get("anthropometric_profile") or {}).get("tolerance"),
        "new_anthro_tolerance": new_config["anthropometric_profile"]["tolerance"],
        "backup": None,
    }
    if dry_run:
        report["status"] = "would_retrim"
        return report
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = os.path.join(templates_dir, f".backup_{username}_{stamp}")
    shutil.copytree(os.path.join(templates_dir, username), backup)
    save_registration(username, poses, templates_dir=templates_dir,
                      raw_samples=raws, frame_aspect=aspect,
                      pose_method=pose_method, temporal_trim=TEMPORAL_TRIM_METHOD)
    report["status"] = "retrimmed"
    report["backup"] = backup
    return report


def profile_arrays(username, templates_dir=TEMPLATES_DIR):
    """Every biometric .npy file of a profile (templates, raw recordings, face)."""
    user_dir = os.path.join(templates_dir, normalize_username(username))
    return sorted(os.path.join(user_dir, f) for f in os.listdir(user_dir)
                  if f.endswith(".npy") and not f.startswith("."))


def encrypt_user(username, templates_dir=TEMPLATES_DIR, decrypt=False):
    """Encrypt (or decrypt) a profile's arrays in place. Returns files changed."""
    if not decrypt and not secure_store.available():
        raise ValueError("Windows DPAPI is not available on this system.")
    changed = 0
    for path in profile_arrays(username, templates_dir):
        if decrypt:
            if secure_store.is_encrypted(path):
                array = secure_store.load_array(path)
                tmp = path + ".tmp_plain"
                secure_store.save_array(tmp, array, encrypt=False)
                if not np.array_equal(secure_store.load_array(tmp), array):
                    os.remove(tmp)
                    raise ValueError(f"round-trip check failed for {path}")
                os.replace(tmp, path)
                changed += 1
        elif secure_store.encrypt_file_in_place(path):
            changed += 1
    return changed


def main(argv=None):
    parser = argparse.ArgumentParser(description="Migrate WaveLock profiles to the aspect-corrected pose method")
    parser.add_argument("--user", "-u", default=None, help="Only this user (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Show the new thresholds without writing")
    parser.add_argument("--templates-dir", default=TEMPLATES_DIR, help="Templates root directory")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--encrypt", action="store_true", help="Encrypt the stored arrays (Windows DPAPI)")
    mode.add_argument("--decrypt", action="store_true", help="Store the arrays as plain .npy again")
    mode.add_argument("--trim", action="store_true",
                      help="Recalibrate with the temporal trim (still frames at start/end ignored)")
    args = parser.parse_args(argv)

    users = list_registered_users(args.templates_dir)
    if args.user:
        try:
            users = [normalize_username(args.user)]
        except ValueError as e:
            print(f"ERROR: {e}")
            return 1
    if not users:
        print(f"No registered users in {args.templates_dir}.")
        return 1

    if args.encrypt or args.decrypt:
        failed = 0
        for user in users:
            try:
                n = 0 if args.dry_run else encrypt_user(user, args.templates_dir, decrypt=args.decrypt)
            except (OSError, ValueError) as e:
                failed += 1
                print(f"  {user}: ERROR {e}")
                continue
            total = len(profile_arrays(user, args.templates_dir))
            enc = sum(secure_store.is_encrypted(p) for p in profile_arrays(user, args.templates_dir))
            print(f"  {user}: {n} file(s) changed; {enc}/{total} encrypted")
        return 1 if failed else 0

    failed = 0
    for user in users:
        try:
            r = (retrim_user if args.trim else migrate_user)(user, args.templates_dir, args.dry_run)
        except (OSError, ValueError, FileNotFoundError) as e:
            failed += 1
            print(f"  {user}: ERROR {e}")
            continue
        print(f"  {user}: {r['status']}")
        if "new" in r:
            for k in SHOWN_KEYS:
                print(f"      {k:<24} {r['old'][k]!s:>8} -> {r['new'][k]!s:>8}")
            print(f"      {'anthropometric tolerance':<24} {r['old_anthro_tolerance']!s:>8} -> "
                  f"{r['new_anthro_tolerance']!s:>8}")
        if r.get("backup"):
            print(f"      backup: {r['backup']}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
