"""Realized-PnL snapshot over the last 7 and 14 days, in one run.

Thin convenience wrapper around pnl_report.py / trade_report.py: fetches your
Alpaca fills once per window and prints a per-symbol realized-PnL table for
each. Equivalent to running:
    python pnl_report.py --days 7
    python pnl_report.py --days 14

Usage:
    python pnl_7_14.py                 # 7- and 14-day windows
    python pnl_7_14.py --days 7 30     # any windows you want
    python pnl_7_14.py --sort pnl

Reads Alpaca creds from the environment / .env (APCA_API_KEY_ID, etc.).
"""

from __future__ import annotations

import argparse

from pnl_report import _load_env, _make_client, fetch_fills
from reversion_bot.trade_report import SORT_KEYS, build_report, format_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Realized PnL over 7- and 14-day windows")
    parser.add_argument("--days", type=int, nargs="+", default=[7, 14],
                        help="windows in days (default: 7 14)")
    parser.add_argument("--sort", choices=sorted(SORT_KEYS), default="pnl",
                        help="sort rows by this column (default pnl)")
    args = parser.parse_args()

    _load_env()
    client = _make_client()

    for days in args.days:
        fills = fetch_fills(client, days)
        report = build_report(fills, sort_by=args.sort)
        print(format_report(report, days))
        print()  # blank line between windows


if __name__ == "__main__":
    main()
