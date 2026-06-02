"""Is the (headless) bot alive? Reads the heartbeat main.py writes each cycle.

Exit code 0 = alive (heartbeat fresh), 1 = stale or missing. Lets you confirm a
scheduled, logged-off bot is running, or wire it into external monitoring.

Usage:
    python health_check.py                 # default: state/heartbeat.json, 180s
    python health_check.py --max-age 300
    python health_check.py --json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from reversion_bot.heartbeat import read_heartbeat, heartbeat_age_seconds, is_alive


def main() -> None:
    parser = argparse.ArgumentParser(description="revbot liveness health check")
    parser.add_argument("--path", default=os.path.join("state", "heartbeat.json"),
                        help="heartbeat file (default state/heartbeat.json)")
    parser.add_argument("--max-age", type=float, default=180.0,
                        help="max heartbeat age in seconds before 'stale' (default 180)")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args()

    hb = read_heartbeat(args.path)
    age = heartbeat_age_seconds(hb)
    alive = is_alive(hb, args.max_age)

    if args.json:
        print(json.dumps({"alive": alive, "age_seconds": age, "heartbeat": hb}, indent=2))
    elif hb is None:
        print(f"DOWN: no heartbeat at {args.path} (bot never started, or wrong path)")
    elif alive:
        print(f"ALIVE: last beat {age:.0f}s ago | cycle={hb.get('cycle')} "
              f"market_open={hb.get('market_open')} monitoring={hb.get('monitoring')} "
              f"mode={hb.get('mode')} pid={hb.get('pid')}")
    else:
        print(f"STALE: last beat {age:.0f}s ago (> {args.max_age:.0f}s) — bot likely dead/hung. "
              f"pid={hb.get('pid')}")

    sys.exit(0 if alive else 1)


if __name__ == "__main__":
    main()
