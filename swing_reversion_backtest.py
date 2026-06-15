"""Swing reversion backtest: does the SAME reversion signal pay better on DAILY bars?

Intraday 5-min reversion has a thin, cost-eaten edge (it competes with HFT).
The hypothesis: on a DAILY timeframe, oversold dips bounce over 1-N days, the
moves are bigger, and cost is a smaller fraction — a less-crowded edge. This runs
the SAME ReversionEngine LONG_REVERSION decision on daily bars, holds multi-day,
and applies the same one-position-at-a-time occupancy + cost discipline used for
the intraday work — so it's apples-to-apples except the timeframe.

Two exit configs (the validated ablation), two non-overlapping windows, a cost
sweep:
  baseline : stop 1.2 / target 2.0 ATR, no trail
  tuned    : stop 1.2 / target 3.0 ATR + 1.5 ATR trailing stop

Long-only (matches the validated live config). Research-only — NOTHING is wired;
swing would need its own overnight-gap risk model before any deploy. Compare the
gross expectancy and breakeven-cost here against the intraday numbers.

Usage:
    python swing_reversion_backtest.py --days 365
    python swing_reversion_backtest.py --days 365 --max-hold 10 --costs 0 2 4 6 8 10
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from reversion_bot.config import ReversionConfig
from reversion_bot.engine import ReversionEngine
from reversion_bot.indicators import calculate_atr
from run_real_backtest import fetch_alpaca_bars

CORE = [
    "ASTS", "MU", "AMD", "NVDA", "SMCI", "META", "AAPL", "APP", "ARM", "CRWD",
    "WDC", "MSFT", "LRCX", "JPM", "FCX", "OXY", "LLY", "CLF", "HIMS", "TEM",
]

ATR_FLOOR_PCT = 0.0035

# (stop_atr, target_atr, trail_atr or None)
CONFIGS = {
    "baseline": (1.20, 2.00, None),
    "tuned":    (1.20, 3.00, 1.50),
}


def _atr(atr_arr, i, price):
    a = float(atr_arr[i]) if not np.isnan(atr_arr[i]) else 0.0
    return max(a, price * ATR_FLOOR_PCT)


def _signal_bars(bars, min_history):
    engine = ReversionEngine(ReversionConfig(min_history=min_history, enable_shorts=False))
    enriched = engine.calculate_indicators(bars)
    idx = [i for i in range(len(enriched))
           if engine.get_decision(enriched.iloc[max(0, i - 1): i + 1]).signal == "LONG_REVERSION"]
    return idx, calculate_atr(bars).to_numpy()


def _sim_gross(signal_idx, high, low, close, atr_arr, stop_m, tp_m, trail_m,
               max_hold_days, cooldown_days):
    """One-position daily long sim returning (gross_returns, hold_days_list).

    Daily occupancy: enter at the signal day's close, exit on a subsequent DAY by
    ATR stop / target / trailing stop / time stop (max_hold_days). Holds across
    days, so returns include overnight gaps. Cost is applied later in the sweep.
    """
    sig = set(signal_idx)
    n = len(close)
    rets, holds = [], []
    in_pos = False
    entry = init_stop = target = highest = 0.0
    trail_dist = None
    held = 0
    last_exit = -10 ** 9
    for i in range(n):
        if in_pos:
            held += 1
            highest = max(highest, high[i])
            cur_stop = init_stop if trail_dist is None else max(init_stop, highest - trail_dist)
            r = None
            if low[i] <= cur_stop:
                r = cur_stop / entry - 1.0
            elif high[i] >= target:
                r = target / entry - 1.0
            elif held >= max_hold_days:
                r = close[i] / entry - 1.0
            if r is not None:
                rets.append(r)
                holds.append(held)
                in_pos = False
                last_exit = i
        if not in_pos and i in sig and i - last_exit > cooldown_days:
            entry = close[i]
            atr = _atr(atr_arr, i, entry)
            init_stop = entry - stop_m * atr
            target = entry + tp_m * atr
            trail_dist = (trail_m * atr) if trail_m is not None else None
            highest = high[i]
            in_pos = True
            held = 0
    return rets, holds


def _pf(arr):
    w, l = arr[arr > 0].sum(), -arr[arr < 0].sum()
    return float(w / l) if l > 0 else (10.0 if w > 0 else 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Swing (daily) reversion backtest (research-only)")
    parser.add_argument("symbols", nargs="*", help="symbols (default: core universe)")
    parser.add_argument("--days", type=int, default=365, help="window length in days (default 365)")
    parser.add_argument("--end", default=None, help="in-sample end YYYY-MM-DD (default today)")
    parser.add_argument("--max-hold", type=int, default=10, help="max hold in TRADING days (default 10)")
    parser.add_argument("--cooldown", type=int, default=1, help="re-entry cooldown in days (default 1)")
    parser.add_argument("--min-history", type=int, default=60, help="daily warmup bars (default 60)")
    parser.add_argument("--costs", type=float, nargs="*", default=[0, 2, 4, 6, 8, 10],
                        help="round-trip costs in bps to sweep")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    symbols = [s.strip().upper() for s in args.symbols] if args.symbols else CORE
    end = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.now()
    fmt = "%Y-%m-%d"
    windows = {
        "in-sample": ((end - timedelta(days=args.days)).strftime(fmt), end.strftime(fmt)),
        "oos":       ((end - timedelta(days=2 * args.days)).strftime(fmt),
                      (end - timedelta(days=args.days)).strftime(fmt)),
    }
    costs = [c / 10000.0 for c in args.costs]
    print(f"Swing reversion (DAILY, long-only) | universe={len(symbols)} | max-hold {args.max_hold}d")
    print(f"  in-sample {windows['in-sample'][0]}->{windows['in-sample'][1]} | "
          f"oos {windows['oos'][0]}->{windows['oos'][1]}")
    print("  baseline=tp2.0  tuned=tp3.0+trail1.5")

    data = {(c, w): ([], []) for c in CONFIGS for w in windows}
    for sym in symbols:
        for w, (a, b) in windows.items():
            try:
                bars = fetch_alpaca_bars(sym, a, b, "1Day")
            except Exception as e:
                print(f"  # {sym} [{w}]: {e}")
                continue
            if bars is None or len(bars) < args.min_history + 5:
                continue
            idx, atr_arr = _signal_bars(bars, args.min_history)
            high = bars["high"].astype(float).to_numpy()
            low = bars["low"].astype(float).to_numpy()
            close = bars["close"].astype(float).to_numpy()
            for cfg, (sm, tm, trm) in CONFIGS.items():
                r, h = _sim_gross(idx, high, low, close, atr_arr, sm, tm, trm,
                                  args.max_hold, args.cooldown)
                data[(cfg, w)][0].extend(r)
                data[(cfg, w)][1].extend(h)

    pf_hdr = "".join(f"PF@{int(c)}".rjust(8) for c in args.costs)
    print(f"\n  {'CONFIG':<10}{'WINDOW':<11}{'N':>5}{'grossEXP':>10}{'WIN':>7}{'HOLD':>6}  {pf_hdr}")
    print("  " + "-" * (49 + 8 * len(args.costs)))
    for cfg in CONFIGS:
        for w in ("in-sample", "oos"):
            rets, holds = data[(cfg, w)]
            arr = np.asarray(rets, dtype=float)
            if arr.size == 0:
                print(f"  {cfg:<10}{w:<11}{0:>5}  (no trades)")
                continue
            avg_hold = float(np.mean(holds)) if holds else 0.0
            pfs = "".join(f"{_pf(arr - c):>8.2f}" for c in costs)
            print(f"  {cfg:<10}{w:<11}{arr.size:>5}{arr.mean()*100:>9.3f}%{(arr>0).mean()*100:>6.1f}%"
                  f"{avg_hold:>5.1f}d  {pfs}")
        print()

    print("  Compare grossEXP and the breakeven cost vs the intraday tuned run")
    print("  (intraday tuned gross ~+0.08-0.12%/trade, breakeven ~8bps). Bigger daily")
    print("  moves should lift gross EXP a lot and push breakeven far higher IF the edge")
    print("  holds in BOTH windows. Research-only; swing needs a gap-risk model to deploy.")


if __name__ == "__main__":
    main()
