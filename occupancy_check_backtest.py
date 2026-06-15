"""Occupancy audit: does realistic one-position-at-a-time trading change the verdict?

Every harness this session counted EVERY signal bar as a trade (occupancy-free).
But reversion signals cluster -- a falling knife sits below the band for many
bars, minting many losing entries, while a quick bounce mints few -- so losers
are oversampled and the measured PF is biased pessimistic. The live bot never
trades this way: one position per symbol, a symbol cooldown, and a longer
loss-aware re-entry brake.

This re-runs the SAME reversion entries three ways on the core universe, both
windows, to size the bias:

  free  : occupancy-free (every signal bar -> a trade; the old method)
  1pos  : one position at a time + symbol cooldown (30min = 6 bars)
  live  : 1pos + a longer cooldown after a LOSING exit (90min = 18 bars) --
          mirrors symbol_cooldown_minutes(30) + loss_reentry_cooldown_minutes(90)

If reversion's PF crosses 1.0 going free -> live, the "everything loses" finding
was a counting artifact and the absolute conclusion must be revised (relative
A/B comparisons stay valid -- the bias was symmetric across arms).

Usage:
    python occupancy_check_backtest.py --days 180
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
COST_PCT = 0.0008
STOP_ATR, TARGET_ATR, HOLD = 1.20, 2.00, 78
COOLDOWN_BARS = 6        # 30 min on 5-min bars (symbol_cooldown_minutes)
LOSS_COOLDOWN_BARS = 18  # 90 min (loss_reentry_cooldown_minutes)


def _signal_bars(bars):
    """Indices where the engine prints LONG_REVERSION (and the ATR series)."""
    engine = ReversionEngine(ReversionConfig(min_history=60, enable_shorts=False))
    enriched = engine.calculate_indicators(bars)
    idx = [i for i in range(len(enriched))
           if engine.get_decision(enriched.iloc[max(0, i - 1): i + 1]).signal == "LONG_REVERSION"]
    return idx, calculate_atr(bars).to_numpy()


def _atr(atr_arr, i, price):
    a = float(atr_arr[i]) if not np.isnan(atr_arr[i]) else 0.0
    return max(a, price * ATR_FLOOR_PCT)


def _free(signal_idx, high, low, close, atr_arr):
    """Occupancy-free: every signal bar is an independent long trade."""
    out = []
    n = len(close)
    for i in signal_idx:
        entry = close[i]
        atr = _atr(atr_arr, i, entry)
        stop, target = entry - atr * STOP_ATR, entry + atr * TARGET_ATR
        end = min(i + HOLD, n - 1)
        r = (close[end] / entry - 1.0) - COST_PCT
        for j in range(i + 1, end + 1):
            if low[j] <= stop:
                r = (stop / entry - 1.0) - COST_PCT
                break
            if high[j] >= target:
                r = (target / entry - 1.0) - COST_PCT
                break
        out.append(r)
    return out


def _sequential(signal_idx, high, low, close, atr_arr, loss_aware):
    """One position at a time + cooldown; longer cooldown after a loss if loss_aware."""
    sig = set(signal_idx)
    out = []
    n = len(close)
    in_pos = False
    entry = stop = target = 0.0
    held = 0
    last_exit = -10 ** 9
    last_loss = False
    for i in range(n):
        if in_pos:
            held += 1
            r = None
            if low[i] <= stop:
                r = (stop / entry - 1.0) - COST_PCT
            elif high[i] >= target:
                r = (target / entry - 1.0) - COST_PCT
            elif held >= HOLD:
                r = (close[i] / entry - 1.0) - COST_PCT
            if r is not None:
                out.append(r)
                in_pos = False
                last_exit = i
                last_loss = r < 0
        if not in_pos and i in sig:
            cd = LOSS_COOLDOWN_BARS if (loss_aware and last_loss) else COOLDOWN_BARS
            if i - last_exit > cd:
                entry = close[i]
                atr = _atr(atr_arr, i, entry)
                stop, target = entry - atr * STOP_ATR, entry + atr * TARGET_ATR
                in_pos = True
                held = 0
    return out


def _stats(returns) -> dict:
    arr = np.asarray(returns, dtype=float)
    if arr.size == 0:
        return dict(n=0, net=0.0, expectancy=0.0, pf=0.0, win=0.0)
    wins, losses = arr[arr > 0].sum(), -arr[arr < 0].sum()
    return dict(
        n=int(arr.size), net=float(arr.sum()), expectancy=float(arr.mean()),
        pf=float(wins / losses) if losses > 0 else (10.0 if wins > 0 else 0.0),
        win=float((arr > 0).mean()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Occupancy audit (research-only)")
    parser.add_argument("symbols", nargs="*", help="symbols (default: core universe)")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--end", default=None)
    parser.add_argument("--timeframe", default="5Min")
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
    print(f"Occupancy audit (reversion, long) | universe={len(symbols)} | {args.timeframe} | "
          f"in-sample {windows['in-sample'][0]}->{windows['in-sample'][1]} | "
          f"oos {windows['oos'][0]}->{windows['oos'][1]}")

    pooled = {(m, w): [] for m in ("free", "1pos", "live") for w in windows}
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
            pooled[("free", w)] += _free(idx, high, low, close, atr_arr)
            pooled[("1pos", w)] += _sequential(idx, high, low, close, atr_arr, loss_aware=False)
            pooled[("live", w)] += _sequential(idx, high, low, close, atr_arr, loss_aware=True)

    print(f"\n  {'MODE':<7}{'WINDOW':<11}{'N':>7}{'NET':>10}{'EXP':>9}{'PF':>7}{'WIN':>8}")
    print("  " + "-" * 55)
    for m in ("free", "1pos", "live"):
        for w in ("in-sample", "oos"):
            s = _stats(pooled[(m, w)])
            print(f"  {m:<7}{w:<11}{s['n']:>7}{s['net']:>10.4f}{s['expectancy']*100:>8.3f}%"
                  f"{s['pf']:>7.2f}{s['win']*100:>7.1f}%")
        print()

    print("  If PF rises free -> 1pos -> live and crosses 1.0, the clustered losers were a")
    print("  counting artifact: the live bot (one position + cooldowns) trades far fewer,")
    print("  better entries than the per-signal harness measured.")


if __name__ == "__main__":
    main()
