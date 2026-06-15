"""Exit ablation: which carries the edge -- the wider target or the trailing stop?

The execution-tuning run showed tuned (stop 1.2 / target 3.0 ATR + 1.5 ATR trail)
robustly profitable in both windows out to ~8bps, vs baseline (tp 2.0, no trail)
at ~3.5bps. But target and trail were changed together. This isolates them on the
realistic live-occupancy sim (one position + symbol cooldown + loss-aware re-entry
brake), so we know what to wire:

  baseline   : stop 1.2 / target 2.0, no trail   (the audit config)
  tp3_only   : stop 1.2 / target 3.0, no trail   (wider target alone)
  trail_only : stop 1.2 / target 2.0 + 1.5 trail (trailing alone)
  tuned      : stop 1.2 / target 3.0 + 1.5 trail (both)

For each config x window it prints gross expectancy and PF at a cost sweep, so the
component that moves breakeven is obvious. (Limit entry was dropped: it caused
adverse selection -- filled the losers, missed the bounces.) Research-only.

Usage:
    python execution_tuning_backtest.py --days 180
    python execution_tuning_backtest.py --costs 0 1 2 3 4 --days 180
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
HOLD = 78
COOLDOWN_BARS, LOSS_COOLDOWN_BARS = 6, 18

# (stop_atr, target_atr, trail_atr or None)
CONFIGS = {
    "baseline":   (1.20, 2.00, None),
    "tp3_only":   (1.20, 3.00, None),
    "trail_only": (1.20, 2.00, 1.50),
    "tuned":      (1.20, 3.00, 1.50),
}


def _atr(atr_arr, i, price):
    a = float(atr_arr[i]) if not np.isnan(atr_arr[i]) else 0.0
    return max(a, price * ATR_FLOOR_PCT)


def _signal_bars(bars):
    engine = ReversionEngine(ReversionConfig(min_history=60, enable_shorts=False))
    enriched = engine.calculate_indicators(bars)
    idx = [i for i in range(len(enriched))
           if engine.get_decision(enriched.iloc[max(0, i - 1): i + 1]).signal == "LONG_REVERSION"]
    return idx, calculate_atr(bars).to_numpy()


def _sim_gross(signal_idx, high, low, close, atr_arr, stop_m, tp_m, trail_m) -> list:
    """Live-occupancy long sim (one position + cooldowns) returning GROSS returns.
    Trailing stop ratchets the stop up to highest_high - trail_m*ATR (never down)."""
    sig = set(signal_idx)
    n = len(close)
    out = []
    in_pos = False
    entry = init_stop = target = highest = 0.0
    trail_dist = None
    held = 0
    last_exit = -10 ** 9
    last_loss = False
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
            elif held >= HOLD:
                r = close[i] / entry - 1.0
            if r is not None:
                out.append(r)
                in_pos = False
                last_exit = i
                last_loss = r < 0
        if not in_pos and i in sig:
            cd = LOSS_COOLDOWN_BARS if last_loss else COOLDOWN_BARS
            if i - last_exit > cd:
                entry = close[i]
                atr = _atr(atr_arr, i, entry)
                init_stop = entry - stop_m * atr
                target = entry + tp_m * atr
                trail_dist = (trail_m * atr) if trail_m is not None else None
                highest = high[i]
                in_pos = True
                held = 0
    return out


def _pf(arr):
    w, l = arr[arr > 0].sum(), -arr[arr < 0].sum()
    return float(w / l) if l > 0 else (10.0 if w > 0 else 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exit ablation backtest (research-only)")
    parser.add_argument("symbols", nargs="*", help="symbols (default: core universe)")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--end", default=None)
    parser.add_argument("--timeframe", default="5Min")
    parser.add_argument("--costs", type=float, nargs="*", default=[0, 2, 4, 6, 8],
                        help="round-trip costs in bps to sweep (default 0 2 4 6 8)")
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
    print(f"Exit ablation (live occupancy, long) | universe={len(symbols)} | {args.timeframe}")
    print(f"  in-sample {windows['in-sample'][0]}->{windows['in-sample'][1]} | "
          f"oos {windows['oos'][0]}->{windows['oos'][1]}")
    print("  baseline=tp2.0  tp3_only=tp3.0  trail_only=tp2.0+trail1.5  tuned=tp3.0+trail1.5")

    gross = {(c, w): [] for c in CONFIGS for w in windows}
    for sym in symbols:
        for w, (a, b) in windows.items():
            try:
                bars = fetch_alpaca_bars(sym, a, b, args.timeframe)
            except Exception as e:
                print(f"  # {sym} [{w}]: {e}")
                continue
            if bars is None or len(bars) < 60:
                continue
            idx, atr_arr = _signal_bars(bars)
            high = bars["high"].astype(float).to_numpy()
            low = bars["low"].astype(float).to_numpy()
            close = bars["close"].astype(float).to_numpy()
            for cfg, (sm, tm, trm) in CONFIGS.items():
                gross[(cfg, w)] += _sim_gross(idx, high, low, close, atr_arr, sm, tm, trm)

    pf_hdr = "".join(f"PF@{int(c)}".rjust(8) for c in args.costs)
    print(f"\n  {'CONFIG':<11}{'WINDOW':<11}{'N':>6}{'grossEXP':>10}{'WIN':>7}  {pf_hdr}")
    print("  " + "-" * (45 + 8 * len(args.costs)))
    for cfg in CONFIGS:
        for w in ("in-sample", "oos"):
            arr = np.asarray(gross[(cfg, w)], dtype=float)
            if arr.size == 0:
                print(f"  {cfg:<11}{w:<11}{0:>6}  (no trades)")
                continue
            pfs = "".join(f"{_pf(arr - c):>8.2f}" for c in costs)
            print(f"  {cfg:<11}{w:<11}{arr.size:>6}{arr.mean()*100:>9.3f}%{(arr>0).mean()*100:>6.1f}%  {pfs}")
        print()

    print("  Compare tp3_only and trail_only vs baseline and tuned: if tuned ~ tp3_only,")
    print("  the target carries it; if tuned ~ trail_only, the trail does; if both lift and")
    print("  tuned is best, they're complementary -> wire both.")


if __name__ == "__main__":
    main()
