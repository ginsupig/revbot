"""Signal comparison backtest: is a DIFFERENT signal profitable where reversion isn't?

Trimming the universe got mean-reversion to breakeven; swapping to a higher-vol
universe didn't help -- so the problem is the SIGNAL, not the names. This holds
the universe fixed (core) and swaps the entry signal, running each through the
SAME ATR-bracket sim over two non-overlapping windows so only the signal varies:

  reversion : engine LONG_REVERSION (the current strategy; long)
  trendfail : failed-breakout fade (trendfail.py); long after a downside fakeout,
              short after an upside fakeout
  exhaustion: bearish-exhaustion short (exhaustion.py); short on a new high with
              fading volume

Per-setup (occupancy-free) expectancy; bracket and cost match the other harnesses.
Research-only, nothing wired. A signal is interesting only if it clears PF>1 in
BOTH windows -- the bar reversion fails.

Usage:
    python signal_compare_backtest.py --days 180
    python signal_compare_backtest.py --signals reversion trendfail --days 180
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from reversion_bot.config import ReversionConfig
from reversion_bot.engine import ReversionEngine
from reversion_bot.indicators import calculate_atr
from reversion_bot.trendfail import trend_fail_continue_strategy
from reversion_bot.exhaustion import exhaustion_short_signal
from run_real_backtest import fetch_alpaca_bars

CORE = [
    "ASTS", "MU", "AMD", "NVDA", "SMCI", "META", "AAPL", "APP", "ARM", "CRWD",
    "WDC", "MSFT", "LRCX", "JPM", "FCX", "OXY", "LLY", "CLF", "HIMS", "TEM",
]

ATR_FLOOR_PCT = 0.0035
COST_PCT = 0.0008
STD = dict(stop=1.20, target=2.00, hold=78)
ALL_SIGNALS = ("reversion", "trendfail", "exhaustion")


def _outcome(high, low, close, i, entry, atr, side, br) -> float:
    """Round-trip return for a long/short entry under an ATR bracket + time stop.
    Stop checked before target within a bar (conservative)."""
    end = min(i + br["hold"], len(close) - 1)
    if side == "long":
        stop, target = entry - atr * br["stop"], entry + atr * br["target"]
        for j in range(i + 1, end + 1):
            if low[j] <= stop:
                return (stop / entry - 1.0) - COST_PCT
            if high[j] >= target:
                return (target / entry - 1.0) - COST_PCT
        return (close[end] / entry - 1.0) - COST_PCT
    else:  # short
        stop, target = entry + atr * br["stop"], entry - atr * br["target"]
        for j in range(i + 1, end + 1):
            if high[j] >= stop:
                return (entry - stop) / entry - COST_PCT
            if low[j] <= target:
                return (entry - target) / entry - COST_PCT
        return (entry - close[end]) / entry - COST_PCT


def _entries_for_signal(name, bars, enriched, engine):
    """Return a list of (bar_index, side) entries for the named signal."""
    n = len(bars)
    if name == "reversion":
        out = []
        for i in range(n):
            if engine.get_decision(enriched.iloc[max(0, i - 1): i + 1]).signal == "LONG_REVERSION":
                out.append((i, "long"))
        return out
    if name == "trendfail":
        sig = trend_fail_continue_strategy(bars)["signal"].to_numpy()
        return [(i, "long" if sig[i] == 1 else "short") for i in range(n) if sig[i] != 0]
    if name == "exhaustion":
        sig = exhaustion_short_signal(bars).to_numpy()
        return [(i, "short") for i in range(n) if sig[i]]
    raise ValueError(name)


def _symbol_returns(bars, signals) -> dict:
    engine = ReversionEngine(ReversionConfig(min_history=60, enable_shorts=False))
    enriched = engine.calculate_indicators(bars)
    high = bars["high"].astype(float).to_numpy()
    low = bars["low"].astype(float).to_numpy()
    close = bars["close"].astype(float).to_numpy()
    atr = calculate_atr(bars).to_numpy()

    out = {s: [] for s in signals}
    for name in signals:
        for i, side in _entries_for_signal(name, bars, enriched, engine):
            c = close[i]
            a = max(float(atr[i]) if not np.isnan(atr[i]) else 0.0, c * ATR_FLOOR_PCT)
            out[name].append(_outcome(high, low, close, i, c, a, side, STD))
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
    parser = argparse.ArgumentParser(description="Signal comparison backtest (research-only)")
    parser.add_argument("symbols", nargs="*", help="symbols (default: core universe)")
    parser.add_argument("--signals", nargs="*", default=list(ALL_SIGNALS), choices=ALL_SIGNALS)
    parser.add_argument("--days", type=int, default=180, help="window length in days (default 180)")
    parser.add_argument("--end", default=None, help="in-sample end YYYY-MM-DD (default today)")
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
    print(f"Signal comparison | universe={len(symbols)} | {args.timeframe} | "
          f"in-sample {windows['in-sample'][0]}->{windows['in-sample'][1]} | "
          f"oos {windows['oos'][0]}->{windows['oos'][1]}")

    pooled = {(s, w): [] for s in args.signals for w in windows}
    for sym in symbols:
        for w, (a, b) in windows.items():
            try:
                bars = fetch_alpaca_bars(sym, a, b, args.timeframe)
            except Exception as e:
                print(f"  # {sym} [{w}]: {e}")
                continue
            if bars is None or len(bars) < 60:
                continue
            rets = _symbol_returns(bars, args.signals)
            for s in args.signals:
                pooled[(s, w)] += rets[s]

    print(f"\n  {'SIGNAL':<11}{'WINDOW':<11}{'N':>7}{'NET':>10}{'EXP':>9}{'PF':>7}{'WIN':>8}")
    print("  " + "-" * 60)
    for s in args.signals:
        for w in ("in-sample", "oos"):
            st = _stats(pooled[(s, w)])
            print(f"  {s:<11}{w:<11}{st['n']:>7}{st['net']:>10.4f}{st['expectancy']*100:>8.3f}%"
                  f"{st['pf']:>7.2f}{st['win']*100:>7.1f}%")
        print()

    print("  A signal is worth pursuing only if PF>1 in BOTH windows (the bar reversion")
    print("  fails). Same universe/bracket/cost across all three, so only the signal varies.")


if __name__ == "__main__":
    main()
