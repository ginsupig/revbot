"""Report broker fees over a lookback window.

pnl_report.py shows only price-based realized PnL from FILL activities and
omits fees entirely, so its totals are GROSS. Alpaca is commission-free on US
equities, but regulatory pass-throughs (SEC Section 31, FINRA TAF, etc.) post
as separate non-trade activities on the sell side. This script sums those so
you can see net-of-fee performance.

Usage:
    python fees_report.py --days 1
    python fees_report.py --days 7 --json
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List


def _attr(obj, name, default=None):
    """Read a field whether the activity is an SDK object or a raw dict."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def make_client():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    from alpaca_trade_api.rest import REST

    key = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")
    base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
    if not key or not secret:
        raise SystemExit(
            "Missing Alpaca creds. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY "
            "(see .env.example)."
        )
    return REST(key, secret, base_url)


def fetch_non_trade_activities(client, days: int) -> List[dict]:
    """All non-FILL account activities over the window (fees, journals, etc.).

    We fetch without an activity_types filter and drop FILLs, so fee codes are
    captured whatever Alpaca labels them (SEC/TAF/REG/FEE), rather than guessing
    the exact type strings. Paginates via the last activity id.
    """
    after = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    out: List[dict] = []
    page_token = None
    page_size = 100
    while True:
        batch = client.get_activities(
            after=after, direction="asc", page_size=page_size, page_token=page_token
        )
        if not batch:
            break
        for a in batch:
            atype = str(_attr(a, "activity_type") or _attr(a, "type") or "UNKNOWN")
            if atype.upper() == "FILL":
                continue
            out.append(
                {
                    "activity_type": atype,
                    "net_amount": _attr(a, "net_amount"),
                    "symbol": _attr(a, "symbol"),
                    "date": str(_attr(a, "date", "") or _attr(a, "transaction_time", "")),
                }
            )
        if len(batch) < page_size:
            break
        page_token = _attr(batch[-1], "id")
        if not page_token:
            break
    return out


def summarize_fees(activities: List[dict]) -> dict:
    """Group non-trade activities by type and sum their net amounts.

    Pure (no network) so it unit-tests in isolation. Returns per-type counts +
    totals and an overall fees figure (the sum of negative net amounts, i.e.
    debits — that's what fees are).
    """
    by_type: Dict[str, dict] = defaultdict(lambda: {"count": 0, "total": 0.0})
    total_fees = 0.0
    for a in activities:
        atype = str(a.get("activity_type") or "UNKNOWN")
        raw = a.get("net_amount")
        try:
            net = float(raw) if raw is not None else 0.0
        except (TypeError, ValueError):
            net = 0.0
        by_type[atype]["count"] += 1
        by_type[atype]["total"] += net
        if net < 0:
            total_fees += net
    return {
        "by_type": {k: v for k, v in sorted(by_type.items())},
        "total_fees": round(total_fees, 4),
    }


def format_report(report: dict, days: int) -> str:
    lines = [f"Fees report — last {days} day(s)"]
    by_type = report["by_type"]
    if not by_type:
        lines.append("  (no non-trade activities — $0 in fees/adjustments)")
        return "\n".join(lines)
    lines.append(f"  {'TYPE':<10}{'COUNT':>7}{'NET AMOUNT':>16}")
    lines.append(f"  {'-'*10}{'-'*7:>7}{'-'*16:>16}")
    for atype, agg in by_type.items():
        lines.append(f"  {atype:<10}{agg['count']:>7}{agg['total']:>16,.4f}")
    lines.append(f"\n  total fees (debits): {report['total_fees']:,.4f}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpaca fees / non-trade activity report")
    parser.add_argument("--days", type=int, default=1, help="lookback window in days (default 1)")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = parser.parse_args()

    client = make_client()
    activities = fetch_non_trade_activities(client, args.days)
    report = summarize_fees(activities)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_report(report, args.days))


if __name__ == "__main__":
    main()
