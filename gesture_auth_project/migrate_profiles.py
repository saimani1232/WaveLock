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
"""

import os
import sys
import shutil
import argparse
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
)
from gesture_capture import TARGET_FRAMES
from utils.normalize import normalize_gesture, POSE_METHOD_ASPECT, POSE_METHOD_LEGACY

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
    poses = [normalize_gesture(r, TARGET_FRAMES, frame_aspect=aspect) for r in raws]
    new_config = build_profile_config(poses, raws, aspect, POSE_METHOD_ASPECT)

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
                      pose_method=POSE_METHOD_ASPECT)
    report["status"] = "migrated"
    report["backup"] = backup
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Migrate WaveLock profiles to the aspect-corrected pose method")
    parser.add_argument("--user", "-u", default=None, help="Only this user (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Show the new thresholds without writing")
    parser.add_argument("--templates-dir", default=TEMPLATES_DIR, help="Templates root directory")
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

    failed = 0
    for user in users:
        try:
            r = migrate_user(user, args.templates_dir, args.dry_run)
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
