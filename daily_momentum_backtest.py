"""Daily momentum backtest: does buying STRENGTH persist on daily where reversion didn't?

The swing-reversion test showed daily is cost-robust (moves dwarf friction) but the
reversion *signal* doesn't persist — it flipped sign across two years. Momentum is
the natural counter-hypothesis: it historically persists on daily where short-horizon
reversion decays, and it wants the exact exit (trail winners, no fixed target) that
HURT reversion.

Signal: an N-day breakout — close makes a new ``--lookback``-day high (buy strength),
optionally only above a long trend SMA (``--trend-sma``). Exit: a chandelier-style
ATR trailing stop (no profit target — let trends run), max-hold backstop. One position
at a time, two non-overlapping windows, cost sweep. Long-only.

Momentum has a LOW win rate with a fat right tail, so judge it on PF / gross EXP, not
win rate. Research-only — NOTHING wired; this just answers "is there a persistent daily
edge once we leave the crowded reversion/HFT space?"

Usage:
    python daily_momentum_backtest.py --days 365
    python daily_momentum_backtest.py --days 365 --lookback 20 --trend-sma 50
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from reversion_bot.indicators import calculate_atr
from run_real_backtest import fetch_alpaca_bars

CORE = [
    "ASTS", "MU", "AMD", "NVDA", "SMCI", "META", "AAPL", "APP", "ARM", "CRWD",
    "WDC", "MSFT", "LRCX", "JPM", "FCX", "OXY", "LLY", "CLF", "HIMS", "TEM",
]

ATR_FLOOR_PCT = 0.0035
# config -> ATR trailing distance (chandelier). No fixed target: trends run until the trail.
CONFIGS = {"trail2.0": 2.0, "trail3.0": 3.0, "trail4.0": 4.0}


def _atr(atr_arr, i, price):
    a = float(atr_arr[i]) if not np.isnan(atr_arr[i]) else 0.0
    return max(a, price * ATR_FLOOR_PCT)


def _signal_bars(bars, lookback, trend_sma):
    """Indices where close makes a new ``lookback``-day high (prior highs, no
    look-ahead), optionally only when close is above its ``trend_sma`` SMA."""
    high = bars["high"].astype(float)
    close = bars["close"].astype(float)
    prior_high = high.rolling(lookback, min_periods=lookback).max().shift(1)
    breakout = close > prior_high
    if trend_sma > 0:
        sma = close.rolling(trend_sma, min_periods=trend_sma).mean()
        breakout = breakout & (close > sma)
    idx = [i for i, b in enumerate(breakout.fillna(False).to_numpy()) if b]
    return idx, calculate_atr(bars).to_numpy()


def _sim_gross(signal_idx, high, low, close, atr_arr, trail_m, max_hold_days, cooldown_days):
    """One-position daily long sim, chandelier trailing stop (no target). Returns
    (gross_returns, hold_days)."""
    sig = set(signal_idx)
    n = len(close)
    rets, holds = [], []
    in_pos = False
    entry = init_stop = trail_dist = highest = 0.0
    held = 0
    last_exit = -10 ** 9
    for i in range(n):
        if in_pos:
            held += 1
            highest = max(highest, high[i])
            cur_stop = max(init_stop, highest - trail_dist)
            r = None
            if low[i] <= cur_stop:
                r = cur_stop / entry - 1.0
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
            trail_dist = trail_m * atr
            init_stop = entry - trail_dist
            highest = high[i]
            in_pos = True
            held = 0
    return rets, holds


def _pf(arr):
    w, l = arr[arr > 0].sum(), -arr[arr < 0].sum()
    return float(w / l) if l > 0 else (10.0 if w > 0 else 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily momentum backtest (research-only)")
    parser.add_argument("symbols", nargs="*", help="symbols (default: core universe)")
    parser.add_argument("--days", type=int, default=365, help="window length in days (default 365)")
    parser.add_argument("--end", default=None, help="in-sample end YYYY-MM-DD (default today)")
    parser.add_argument("--lookback", type=int, default=20, help="breakout lookback in days (default 20)")
    parser.add_argument("--trend-sma", type=int, default=50, help="only buy above this SMA; 0=off (default 50)")
    parser.add_argument("--max-hold", type=int, default=60, help="max hold in trading days (default 60)")
    parser.add_argument("--cooldown", type=int, default=1, help="re-entry cooldown in days (default 1)")
    parser.add_argument("--costs", type=float, nargs="*", default=[0, 2, 4, 6, 8, 10])
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
    print(f"Daily momentum (long-only) | universe={len(symbols)} | "
          f"{args.lookback}d-breakout, SMA{args.trend_sma} filter, max-hold {args.max_hold}d")
    print(f"  in-sample {windows['in-sample'][0]}->{windows['in-sample'][1]} | "
          f"oos {windows['oos'][0]}->{windows['oos'][1]}   (low win-rate is normal — judge on PF)")

    data = {(c, w): ([], []) for c in CONFIGS for w in windows}
    for sym in symbols:
        for w, (a, b) in windows.items():
            try:
                bars = fetch_alpaca_bars(sym, a, b, "1Day")
            except Exception as e:
                print(f"  # {sym} [{w}]: {e}")
                continue
            if bars is None or len(bars) < max(args.trend_sma, args.lookback) + 5:
                continue
            idx, atr_arr = _signal_bars(bars, args.lookback, args.trend_sma)
            high = bars["high"].astype(float).to_numpy()
            low = bars["low"].astype(float).to_numpy()
            close = bars["close"].astype(float).to_numpy()
            for cfg, trail in CONFIGS.items():
                r, h = _sim_gross(idx, high, low, close, atr_arr, trail, args.max_hold, args.cooldown)
                data[(cfg, w)][0].extend(r)
                data[(cfg, w)][1].extend(h)

    pf_hdr = "".join(f"PF@{int(c)}".rjust(8) for c in args.costs)
    print(f"\n  {'CONFIG':<9}{'WINDOW':<11}{'N':>5}{'grossEXP':>10}{'WIN':>7}{'HOLD':>7}  {pf_hdr}")
    print("  " + "-" * (49 + 8 * len(args.costs)))
    for cfg in CONFIGS:
        for w in ("in-sample", "oos"):
            rets, holds = data[(cfg, w)]
            arr = np.asarray(rets, dtype=float)
            if arr.size == 0:
                print(f"  {cfg:<9}{w:<11}{0:>5}  (no trades)")
                continue
            avg_hold = float(np.mean(holds)) if holds else 0.0
            pfs = "".join(f"{_pf(arr - c):>8.2f}" for c in costs)
            print(f"  {cfg:<9}{w:<11}{arr.size:>5}{arr.mean()*100:>9.3f}%{(arr>0).mean()*100:>6.1f}%"
                  f"{avg_hold:>6.1f}d  {pfs}")
        print()

    print("  Persistent momentum edge = PF>1 in BOTH windows, ideally with cost barely")
    print("  biting (daily moves dwarf bps). If it holds where reversion flipped, daily")
    print("  momentum is the real direction. Research-only; needs a gap-risk model to deploy.")


if __name__ == "__main__":
    main()
