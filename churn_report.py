"""Per-symbol churn / trade-economics report from your Alpaca fills.

Shows, for each symbol, how much edge you actually capture *per round trip*
(in bps of trip notional) and flags names where that edge is below your assumed
round-trip cost — i.e. where trading is feeding the spread, not your account.

Usage:
    python churn_report.py                 # last 14 days, 5 bps assumed cost
    python churn_report.py --days 30 --cost-bps 8
    python churn_report.py --json

Reads Alpaca creds from the environment / .env (APCA_API_KEY_ID, etc.).
"""

from __future__ import annotations

import argparse
import json

from pnl_report import _load_env, _make_client, fetch_fills
from reversion_bot.churn import build_churn_report, format_churn_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpaca churn / trade-economics report")
    parser.add_argument("--days", type=int, default=14, help="lookback window in days (default 14)")
    parser.add_argument("--cost-bps", type=float, default=5.0,
                        help="assumed round-trip cost in bps, for comparison (default 5)")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = parser.parse_args()

    _load_env()
    client = _make_client()
    fills = fetch_fills(client, args.days)
    report = build_churn_report(fills, cost_bps=args.cost_bps)

    if args.json:
        rows = [vars(r) for r in report["rows"]]
        print(json.dumps({**report, "rows": rows, "days": args.days}, indent=2))
    else:
        print(f"Lookback: {args.days} days")
        print(format_churn_report(report))


if __name__ == "__main__":
    main()
