"""
Failed-attempt limiting for WaveLock logins.

After MAX_CONSECUTIVE_FAILURES failed attempts in a row the account is
locked for LOCKOUT_BASE_SEC seconds; every further lockout doubles the wait
(capped at LOCKOUT_MAX_SEC). A granted login resets the counter.

State lives in templates/.attempts/<user>.json, outside the user's profile
folder, so it survives re-registration and restarting the program. The
folder name starts with a dot, so list_registered_users() ignores it. Each
attempt is also appended to templates/.attempts/<user>.log (one JSON object
per line) for later analysis.

Limits: the files are ordinary local files. Anyone who can delete them can
reset the counter; protecting them needs operating-system access control.
"""

import json
import os
import tempfile
import time
from datetime import datetime

MAX_CONSECUTIVE_FAILURES = 5
LOCKOUT_BASE_SEC = 30
LOCKOUT_MAX_SEC = 900
ATTEMPTS_DIRNAME = ".attempts"


def _paths(username, templates_dir):
    folder = os.path.join(templates_dir, ATTEMPTS_DIRNAME)
    return (folder,
            os.path.join(folder, f"{username}.json"),
            os.path.join(folder, f"{username}.log"))


def _fresh_state():
    return {"consecutive_failures": 0, "lockouts": 0, "locked_until": 0.0}


def load_attempt_state(username, templates_dir, now=None):
    """
    Read the attempt state. A damaged state file fails closed: it is read
    as a fresh lockout rather than as a clean record.
    """
    _, state_path, _ = _paths(username, templates_dir)
    if not os.path.exists(state_path):
        return _fresh_state()
    try:
        with open(state_path, "r") as f:
            state = json.load(f)
        if not isinstance(state, dict):
            raise ValueError("not a JSON object")
        return {
            "consecutive_failures": int(state.get("consecutive_failures", 0)),
            "lockouts": int(state.get("lockouts", 0)),
            "locked_until": float(state.get("locked_until", 0.0)),
        }
    except (ValueError, TypeError, OSError):
        now = time.time() if now is None else now
        return {"consecutive_failures": 0, "lockouts": 1,
                "locked_until": now + LOCKOUT_BASE_SEC}


def _save_state(username, templates_dir, state):
    folder, state_path, _ = _paths(username, templates_dir)
    os.makedirs(folder, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=folder, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, state_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def seconds_locked(username, templates_dir, now=None):
    """Seconds until the next attempt is allowed (0.0 when not locked)."""
    now = time.time() if now is None else now
    state = load_attempt_state(username, templates_dir, now)
    return max(0.0, state["locked_until"] - now)


def record_attempt(username, granted, templates_dir, reason=None, now=None):
    """
    Update the counter after one evaluated attempt and append it to the log.

    Returns:
        dict: the new state plus "attempts_left" before the next lockout
        and "locked_for" (seconds, 0.0 when not locked).
    """
    now = time.time() if now is None else now
    state = load_attempt_state(username, templates_dir, now)

    if granted:
        state = _fresh_state()
    else:
        state["consecutive_failures"] += 1
        if state["consecutive_failures"] >= MAX_CONSECUTIVE_FAILURES:
            state["lockouts"] += 1
            wait = min(LOCKOUT_MAX_SEC,
                       LOCKOUT_BASE_SEC * 2 ** (state["lockouts"] - 1))
            state["locked_until"] = now + wait
            state["consecutive_failures"] = 0
    _save_state(username, templates_dir, state)

    _, _, log_path = _paths(username, templates_dir)
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "time": datetime.fromtimestamp(now).isoformat(timespec="seconds"),
            "granted": bool(granted),
            "reason": reason,
        }) + "\n")

    return dict(state,
                attempts_left=MAX_CONSECUTIVE_FAILURES - state["consecutive_failures"],
                locked_for=max(0.0, state["locked_until"] - now))
