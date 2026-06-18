"""Exit stop/target grid: do wider stops/targets fix "winners choked, losers stopped"?

The trailing stop is the validated edge (sl1.2/tp3.0/trail1.5). But every exit
backtest fixed the STOP at 1.2 and only varied target/trail. Hypothesis: a 1.2 ATR
stop knocks reversion entries out early on noise (losers stop too soon), and a
3.0 target still caps winners. This is a 2x2 on stop (1.2 vs 1.5) x target (3.0 vs
3.5), all with the validated 1.5 ATR trail, on the realistic live-occupancy sim
(one position + symbol cooldown + loss-aware re-entry brake):

  sl1.2_tp3.0 : current validated
  sl1.5_tp3.0 : wider stop only      (fewer premature stop-outs?)
  sl1.2_tp3.5 : wider target only    (winners run further?)
  sl1.5_tp3.5 : both wider           (the proposed config)

For each config x window it prints N / gross expectancy / win / PF at a cost
sweep. A wider stop trades a higher win rate (fewer noise stop-outs) for a bigger
average loss when it IS a knife — watch whether net EXP/PF improves. Research-only.

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

# (stop_atr, target_atr, trail_atr or None) — 2x2 on stop x target, all with the
# validated 1.5 ATR trail. Tests the "winners choked / losers stopped early"
# hypothesis: does a wider STOP (1.2 -> 1.5) cut premature stop-outs, and a wider
# TARGET (3.0 -> 3.5) let winners run, vs the current validated sl1.2/tp3.0?
CONFIGS = {
    "sl1.2_tp3.0": (1.20, 3.00, 1.50),   # current validated
    "sl1.5_tp3.0": (1.50, 3.00, 1.50),   # wider stop only
    "sl1.2_tp3.5": (1.20, 3.50, 1.50),   # wider target only
    "sl1.5_tp3.5": (1.50, 3.50, 1.50),   # proposed (both wider)
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
    print(f"Exit stop/target grid (live occupancy, long) | universe={len(symbols)} | {args.timeframe}")
    print(f"  in-sample {windows['in-sample'][0]}->{windows['in-sample'][1]} | "
          f"oos {windows['oos'][0]}->{windows['oos'][1]}")
    print("  all configs use the validated 1.5 ATR trailing stop; sl/tp are ATR multiples")

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
    print(f"\n  {'CONFIG':<13}{'WINDOW':<11}{'N':>6}{'grossEXP':>10}{'WIN':>7}  {pf_hdr}")
    print("  " + "-" * (45 + 8 * len(args.costs)))
    for cfg in CONFIGS:
        for w in ("in-sample", "oos"):
            arr = np.asarray(gross[(cfg, w)], dtype=float)
            if arr.size == 0:
                print(f"  {cfg:<13}{w:<11}{0:>6}  (no trades)")
                continue
            pfs = "".join(f"{_pf(arr - c):>8.2f}" for c in costs)
            print(f"  {cfg:<13}{w:<11}{arr.size:>6}{arr.mean()*100:>9.3f}%{(arr>0).mean()*100:>6.1f}%  {pfs}")
        print()

    print("  Read sl1.5_tp3.5 vs sl1.2_tp3.0 (current): a higher WIN rate on the wider")
    print("  stop means fewer premature stop-outs; what matters is whether net EXP/PF")
    print("  improves in BOTH windows. Compare the two sl rows at fixed tp to isolate the")
    print("  stop's effect. If a config wins both windows surviving cost, we wire it.")


if __name__ == "__main__":
    main()
