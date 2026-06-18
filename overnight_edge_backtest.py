"""Overnight-edge backtest: where does the return live -- day or night?

Every strategy this cycle held INTRADAY. But historically most US equity return
accrues OVERNIGHT (close-to-open), not during the session -- the documented
"overnight drift." This decomposes each name's daily return into its intraday and
overnight pieces and tests whether buying into the close (esp. after weakness) and
holding overnight pays, on the same two-window, cost-swept discipline. No signal
to fit -- it's a holding-period anomaly, not a tuned signal.

Arms (long, one trade/day/name -- no occupancy issues):
  intraday        : buy at OPEN[d], sell at CLOSE[d]           (the session)
  overnight       : buy at CLOSE[d-1], sell at OPEN[d]         (the night)
  overnight+down  : overnight, only after a DOWN day           (buy weakness, hold)
  overnight+os    : overnight, only when prior close is oversold (RSI<thr)

Cost is per round-trip (you cross the spread at both ends; overnight legs are at
the close/open auctions where spreads can be wider, so sweep it). Two windows,
long-only. Research-only -- a real overnight book needs gap-risk sizing.

Usage:
    python overnight_edge_backtest.py --days 365
    python overnight_edge_backtest.py --universe neutral --days 365
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from reversion_bot.indicators import calculate_rsi
from run_real_backtest import fetch_alpaca_bars

CORE = [
    "ASTS", "MU", "AMD", "NVDA", "SMCI", "META", "AAPL", "APP", "ARM", "CRWD",
    "WDC", "MSFT", "LRCX", "JPM", "FCX", "OXY", "LLY", "CLF", "HIMS", "TEM",
]
NEUTRAL = [
    "AAPL", "MSFT", "INTC", "CSCO", "IBM", "ORCL", "QCOM", "TXN",
    "JPM", "BAC", "WFC", "GS", "C", "AXP", "JNJ", "PFE", "MRK", "ABBV", "UNH",
    "BMY", "PG", "KO", "PEP", "WMT", "COST", "MO", "HD", "MCD", "NKE", "SBUX",
    "DIS", "TGT", "XOM", "CVX", "COP", "CAT", "BA", "GE", "HON", "UPS",
    "VZ", "T", "CMCSA",
]
UNIVERSES = {"core": CORE, "neutral": NEUTRAL}

RSI_OVERSOLD = 40.0
ARMS = ("intraday", "overnight", "overnight+down", "overnight+os")


def _returns(bars):
    """Per-arm daily long returns (pre-cost) for one symbol."""
    o = bars["open"].astype(float).to_numpy()
    c = bars["close"].astype(float).to_numpy()
    rsi = calculate_rsi(bars).to_numpy()
    n = len(c)
    out = {a: [] for a in ARMS}
    for d in range(2, n):
        intraday = c[d] / o[d] - 1.0
        overnight = o[d] / c[d - 1] - 1.0          # held from prior close to this open
        out["intraday"].append(intraday)
        out["overnight"].append(overnight)
        if c[d - 1] < c[d - 2]:                     # prior day closed down
            out["overnight+down"].append(overnight)
        if not np.isnan(rsi[d - 1]) and rsi[d - 1] < RSI_OVERSOLD:
            out["overnight+os"].append(overnight)
    return out


def _pf(arr):
    w, l = arr[arr > 0].sum(), -arr[arr < 0].sum()
    return float(w / l) if l > 0 else (10.0 if w > 0 else 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Overnight-edge backtest (research-only)")
    parser.add_argument("symbols", nargs="*", help="symbols (overrides --universe)")
    parser.add_argument("--universe", choices=list(UNIVERSES), default="core")
    parser.add_argument("--days", type=int, default=365, help="window length in days (default 365)")
    parser.add_argument("--end", default=None, help="in-sample end YYYY-MM-DD (default today)")
    parser.add_argument("--costs", type=float, nargs="*", default=[0, 4, 8, 12])
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    symbols = [s.strip().upper() for s in args.symbols] if args.symbols else UNIVERSES[args.universe]
    end = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.now()
    fmt = "%Y-%m-%d"
    windows = {
        "in-sample": ((end - timedelta(days=args.days)).strftime(fmt), end.strftime(fmt)),
        "oos":       ((end - timedelta(days=2 * args.days)).strftime(fmt),
                      (end - timedelta(days=args.days)).strftime(fmt)),
    }
    costs = [c / 10000.0 for c in args.costs]
    uni = "custom" if args.symbols else args.universe
    print(f"Overnight edge (DAILY, long) | universe={uni}({len(symbols)})")
    print(f"  in-sample {windows['in-sample'][0]}->{windows['in-sample'][1]} | "
          f"oos {windows['oos'][0]}->{windows['oos'][1]}")

    pooled = {(a, w): [] for a in ARMS for w in windows}
    for sym in symbols:
        for w, (a, b) in windows.items():
            try:
                bars = fetch_alpaca_bars(sym, a, b, "1Day")
            except Exception as e:
                print(f"  # {sym} [{w}]: {e}")
                continue
            if bars is None or len(bars) < 30:
                continue
            r = _returns(bars)
            for arm in ARMS:
                pooled[(arm, w)] += r[arm]

    pf_hdr = "".join(f"PF@{int(c)}".rjust(8) for c in args.costs)
    print(f"\n  {'VARIANT':<15}{'WINDOW':<11}{'N':>6}{'EXP':>10}{'WIN':>7}  {pf_hdr}")
    print("  " + "-" * (49 + 8 * len(args.costs)))
    for arm in ARMS:
        for w in ("in-sample", "oos"):
            arr = np.asarray(pooled[(arm, w)], dtype=float)
            if arr.size == 0:
                print(f"  {arm:<15}{w:<11}{0:>6}  (no data)")
                continue
            pfs = "".join(f"{_pf(arr - c):>8.2f}" for c in costs)
            print(f"  {arm:<15}{w:<11}{arr.size:>6}{arr.mean()*100:>9.3f}%{(arr>0).mean()*100:>6.1f}%  {pfs}")
        print()

    print("  Look for an arm with EXP>0 and PF>1 in BOTH windows that survives cost.")
    print("  If overnight (or overnight+down) beats intraday consistently, the return")
    print("  lives at night -- a holding-period edge, not a signal. Needs gap-risk sizing.")


if __name__ == "__main__":
    main()
