"""Conditioned overnight backtest: does HOW a stock closed predict its overnight gap?

The plain overnight test ("buy everything at the close") is a wash. The real
hypothesis: the buy/sell-into-close decision should depend on the day's price
action. The motivating case: a stock that ran UP then REVERTED hard into the
close (closed weak, near its low) is short-term oversold and may GAP BACK UP
overnight -- the intraday fade overshot. This buckets each day's overnight gap
(close[d] -> open[d+1]) by intraday-close FEATURES and reports which predict a
persistent overnight edge.

Each arm = the overnight return on days where the close-feature was true:
  all          : every day (baseline)
  weak_close   : closed in the bottom third of the day's range (sold into close)
  strong_close : closed in the top third (strong close)
  down_day     : closed below the open
  fade_runup   : ran up >= --runup intraday but CLOSED BELOW OPEN (the WDC fade)
  oversold     : RSI(14) < 35 at the close
  below_band   : close below the lower Bollinger band (20,2)

Positive EXP / PF>1 in BOTH windows that survives cost = a buy-into-close signal;
a clearly negative arm = a sell/avoid-into-close signal. One trade/day/name,
two windows, cost sweep, long-only, daily bars. Research-only -- needs gap-risk
sizing. (Entry assumes a market-on-close fill, exit market-on-open.)

Usage:
    python overnight_signal_backtest.py --days 365
    python overnight_signal_backtest.py --universe neutral --runup 0.02 --days 365
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from reversion_bot.indicators import calculate_rsi, calculate_bollinger_bands
from run_real_backtest import fetch_alpaca_bars

CORE = [
    "ASTS", "MU", "AMD", "NVDA", "SMCI", "META", "AAPL", "APP", "ARM", "CRWD",
    "WDC", "MSFT", "LRCX", "JPM", "FCX", "OXY", "LLY", "CLF", "HIMS", "TEM",
]
NEUTRAL = [
    "AAPL", "MSFT", "INTC", "CSCO", "IBM", "ORCL", "QCOM", "TXN", "JPM", "BAC",
    "WFC", "GS", "C", "AXP", "JNJ", "PFE", "MRK", "ABBV", "UNH", "BMY", "PG",
    "KO", "PEP", "WMT", "COST", "MO", "HD", "MCD", "NKE", "SBUX", "DIS", "TGT",
    "XOM", "CVX", "COP", "CAT", "BA", "GE", "HON", "UPS", "VZ", "T", "CMCSA",
]
UNIVERSES = {"core": CORE, "neutral": NEUTRAL}

ARMS = ("all", "weak_close", "strong_close", "down_day", "fade_runup", "oversold", "below_band")


def _overnight_by_feature(bars, runup, rsi_thr):
    """Per-arm overnight returns (close[d] -> open[d+1]) bucketed by close-day features."""
    o = bars["open"].astype(float).to_numpy()
    h = bars["high"].astype(float).to_numpy()
    l = bars["low"].astype(float).to_numpy()
    c = bars["close"].astype(float).to_numpy()
    rsi = calculate_rsi(bars).to_numpy()
    _, _, lb = calculate_bollinger_bands(bars)
    lb = lb.to_numpy()
    n = len(c)
    out = {a: [] for a in ARMS}
    for d in range(20, n - 1):                 # warm bands; need d+1 for the open
        rng = h[d] - l[d]
        if rng <= 0 or c[d] <= 0:
            continue
        overnight = o[d + 1] / c[d] - 1.0
        close_pos = (c[d] - l[d]) / rng
        intraday = c[d] / o[d] - 1.0
        ran_up = h[d] / o[d] - 1.0

        out["all"].append(overnight)
        if close_pos < 0.33:
            out["weak_close"].append(overnight)
        if close_pos > 0.66:
            out["strong_close"].append(overnight)
        if intraday < 0:
            out["down_day"].append(overnight)
        if ran_up >= runup and c[d] < o[d]:     # ran up then faded into the close
            out["fade_runup"].append(overnight)
        if not np.isnan(rsi[d]) and rsi[d] < rsi_thr:
            out["oversold"].append(overnight)
        if not np.isnan(lb[d]) and c[d] < lb[d]:
            out["below_band"].append(overnight)
    return out


def _pf(arr):
    w, l = arr[arr > 0].sum(), -arr[arr < 0].sum()
    return float(w / l) if l > 0 else (10.0 if w > 0 else 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Conditioned overnight backtest (research-only)")
    parser.add_argument("symbols", nargs="*", help="symbols (overrides --universe)")
    parser.add_argument("--universe", choices=list(UNIVERSES), default="core")
    parser.add_argument("--days", type=int, default=365, help="window length in days (default 365)")
    parser.add_argument("--end", default=None, help="in-sample end YYYY-MM-DD (default today)")
    parser.add_argument("--runup", type=float, default=0.02, help="fade_runup: intraday high vs open (default 0.02)")
    parser.add_argument("--rsi", type=float, default=35.0, help="oversold RSI threshold (default 35)")
    parser.add_argument("--costs", type=float, nargs="*", default=[0, 4, 8])
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
    print(f"Conditioned overnight (DAILY, long) | universe={uni}({len(symbols)}) | "
          f"runup>={args.runup:.0%}, RSI<{args.rsi:g}")
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
            r = _overnight_by_feature(bars, args.runup, args.rsi)
            for arm in ARMS:
                pooled[(arm, w)] += r[arm]

    pf_hdr = "".join(f"PF@{int(c)}".rjust(8) for c in args.costs)
    print(f"\n  {'CLOSE-FEATURE':<14}{'WINDOW':<11}{'N':>6}{'EXP':>10}{'WIN':>7}  {pf_hdr}")
    print("  " + "-" * (47 + 8 * len(args.costs)))
    for arm in ARMS:
        for w in ("in-sample", "oos"):
            arr = np.asarray(pooled[(arm, w)], dtype=float)
            if arr.size == 0:
                print(f"  {arm:<14}{w:<11}{0:>6}  (no data)")
                continue
            pfs = "".join(f"{_pf(arr - c):>8.2f}" for c in costs)
            print(f"  {arm:<14}{w:<11}{arr.size:>6}{arr.mean()*100:>9.3f}%{(arr>0).mean()*100:>6.1f}%  {pfs}")
        print()

    print("  A close-feature whose overnight EXP/PF beats 'all' and stays >1 in BOTH windows")
    print("  is a buy-into-close signal; a clearly negative one is sell/avoid. Watch N on the")
    print("  rarer features (fade_runup/oversold/below_band) — thin N = noise, not signal.")


if __name__ == "__main__":
    main()
