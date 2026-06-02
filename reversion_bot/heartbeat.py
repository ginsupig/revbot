"""Liveness heartbeat for unattended (headless) running.

``main.py`` writes a small JSON each poll cycle; ``health_check.py`` reads it
and tells you whether the bot is alive and recent. This is how you confirm at
a glance that a scheduled, logged-off bot is actually running — without having
to tail logs or attach to a session.

The age/liveness helpers are pure so they unit-test without touching the clock.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Optional


def write_heartbeat(path: str, status: Optional[dict] = None) -> dict:
    """Atomically write a heartbeat JSON (epoch + ISO timestamp, pid, status)."""
    payload = {
        "ts_epoch": time.time(),
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
    }
    if status:
        payload.update(status)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)  # atomic on the same filesystem
    return payload


def read_heartbeat(path: str) -> Optional[dict]:
    """Return the heartbeat dict, or None if missing/unreadable."""
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def heartbeat_age_seconds(hb: Optional[dict], now: Optional[float] = None) -> Optional[float]:
    """Seconds since the heartbeat was written, or None if missing/unparseable."""
    if not hb:
        return None
    ts = hb.get("ts_epoch")
    if ts is None:
        return None
    try:
        ts = float(ts)
    except (TypeError, ValueError):
        return None
    now = time.time() if now is None else now
    return max(0.0, now - ts)


def is_alive(hb: Optional[dict], max_age_seconds: float, now: Optional[float] = None) -> bool:
    """True if the heartbeat exists and is fresher than ``max_age_seconds``."""
    age = heartbeat_age_seconds(hb, now=now)
    return age is not None and age <= max_age_seconds
