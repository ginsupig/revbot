"""Pull executed trades from Alpaca over a lookback window and report realized
PnL plus each symbol's share of traded notional.

Usage:
    python pnl_report.py              # last 60 days (default)
    python pnl_report.py --days 30
    python pnl_report.py --json       # machine-readable output

Reads Alpaca creds from the environment / .env:
    APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from typing import List

from reversion_bot.trade_report import (
    SORT_KEYS,
    build_daily_timeline,
    build_report,
    build_scoreboard,
    build_trades,
    format_csv,
    format_daily_csv,
    format_daily_timeline,
    format_report,
    format_scoreboard,
    format_trades,
)


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass


def _make_client():
    # Shared preflight: validates creds, prints the endpoint, and on an auth
    # failure prints a clear paper/live hint instead of a raw traceback.
    from reversion_bot.report_client import make_client
    return make_client()


def _attr(obj, name, default=None):
    """Read a field whether the activity is an SDK object or a raw dict."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def fetch_fills(client, days: int) -> List[dict]:
    """Fetch FILL activities over the lookback window, normalized for the report.

    Paginates with page_token so windows larger than one page are covered.
    """
    after = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    fills: List[dict] = []
    page_token = None
    page_size = 100

    while True:
        batch = client.get_activities(
            activity_types="FILL",
            after=after,
            direction="asc",
            page_size=page_size,
            page_token=page_token,
        )
        if not batch:
            break
        for a in batch:
            qty = _attr(a, "qty")
            price = _attr(a, "price")
            symbol = _attr(a, "symbol")
            side = _attr(a, "side")
            if qty is None or price is None or not symbol or not side:
                continue
            fills.append(
                {
                    "symbol": str(symbol),
                    "side": str(side).lower(),
                    "qty": abs(float(qty)),
                    "price": float(price),
                    "time": str(_attr(a, "transaction_time", "")),
                }
            )
        if len(batch) < page_size:
            break
        page_token = _attr(batch[-1], "id")
        if not page_token:
            break

    return fills


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpaca realized-PnL / symbol-weight report")
    parser.add_argument("--days", type=int, default=60, help="lookback window in days (default 60)")
    parser.add_argument(
        "--sort", choices=sorted(SORT_KEYS), default="notional",
        help="sort rows by this column (default notional)",
    )
    parser.add_argument(
        "--scoreboard", action="store_true",
        help="per-day rolling scoreboard (consistency + cut-candidate flags) "
             "instead of the aggregate summary table",
    )
    parser.add_argument(
        "--daily", action="store_true",
        help="per-day timeline: each day's total realized PnL and the tickers traded",
    )
    parser.add_argument(
        "--trades", nargs="?", const="", metavar="SYMBOL",
        help="per-round-trip drill-down (entry/exit/hold/PnL), worst first. "
             "Pass a symbol (e.g. --trades WDC) to isolate one name, or bare "
             "--trades for every round-trip in the window.",
    )
    out_group = parser.add_mutually_exclusive_group()
    out_group.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    out_group.add_argument("--csv", action="store_true", help="emit CSV instead of a table")
    args = parser.parse_args()

    _load_env()
    client = _make_client()
    fills = fetch_fills(client, args.days)

    if args.trades is not None:
        report = build_trades(fills, symbol=args.trades or None)
        if args.json:
            print(json.dumps({**report, "days": args.days}, indent=2))
        else:
            print(format_trades(report, args.days))
        return

    if args.daily:
        report = build_daily_timeline(fills)
        if args.json:
            print(json.dumps({**report, "days": args.days}, indent=2))
        elif args.csv:
            print(format_daily_csv(report), end="")
        else:
            print(format_daily_timeline(report, args.days))
        return

    if args.scoreboard:
        report = build_scoreboard(fills)
        if args.json:
            rows = [vars(r) for r in report["rows"]]
            print(json.dumps({**report, "rows": rows, "days": args.days}, indent=2))
        else:
            print(format_scoreboard(report, args.days))
        return

    report = build_report(fills, sort_by=args.sort)

    if args.json:
        rows = [vars(r) for r in report["rows"]]
        out = {**report, "rows": rows, "days": args.days}
        print(json.dumps(out, indent=2))
    elif args.csv:
        print(format_csv(report), end="")
    else:
        print(format_report(report, args.days))


if __name__ == "__main__":
    main()
