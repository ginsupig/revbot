"""Universe comparison backtest: trim vs swap the universe.

The per-symbol report showed (a) the base book is net-negative, (b) ~12 names
lose in BOTH windows, and (c) the positive names don't persist across windows.
That kills "pick the winners," but leaves two structural moves to compare:

  core     -- the current 20-name universe (baseline)
  trimmed  -- option 1: core minus the names negative in BOTH windows
  highvol  -- option 2: a different universe of higher-idiosyncratic-vol /
              less-efficient names, where mean-reversion historically pays

For each universe it runs the SAME engine (fully filtered, standard bracket) over
TWO non-overlapping windows in one pass -- in-sample (the last --days) and the
immediately-prior window (OOS) -- and reports the pooled portfolio expectancy /
PF for each, so a universe is only credible if it holds up in BOTH. Per-setup
(occupancy-free) expectancy; research-only, nothing wired.

Usage:
    python universe_compare_backtest.py                 # all three, both windows
    python universe_compare_backtest.py --universe highvol
    python universe_compare_backtest.py --days 180
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from reversion_bot.config import ReversionConfig
from reversion_bot.engine import ReversionEngine
from run_real_backtest import fetch_alpaca_bars

CORE = [
    "ASTS", "MU", "AMD", "NVDA", "SMCI", "META", "AAPL", "APP", "ARM", "CRWD",
    "WDC", "MSFT", "LRCX", "JPM", "FCX", "OXY", "LLY", "CLF", "HIMS", "TEM",
]
# Negative-expectancy in BOTH windows (the per-symbol report): drop them.
PERSISTENT_LOSERS = {
    "META", "NVDA", "MSFT", "AAPL", "AMD", "MU", "LLY", "LRCX", "FCX", "SMCI",
    "APP", "HIMS",
}
TRIMMED = [s for s in CORE if s not in PERSISTENT_LOSERS]
# Higher idiosyncratic vol / less-efficient names — where reversion should pay
# more than on liquid mega-caps. Liquidity varies; the harness skips thin names.
HIGHVOL = [
    "IONQ", "RGTI", "SOUN", "BBAI", "LUNR", "RKLB", "ACHR", "JOBY", "PLTR",
    "AFRM", "UPST", "RIOT", "MARA", "CVNA", "RIVN", "LCID", "DKNG", "HOOD",
    "COIN", "AI",
]
UNIVERSES = {"core": CORE, "trimmed": TRIMMED, "highvol": HIGHVOL}

ATR_FLOOR_PCT = 0.0035
COST_PCT = 0.0008
STD = dict(stop=1.20, target=2.00, hold=78)


def _outcome(high, low, close, i, entry, atr, br) -> float:
    stop = entry - atr * br["stop"]
    target = entry + atr * br["target"]
    end = min(i + br["hold"], len(close) - 1)
    for j in range(i + 1, end + 1):
        if low[j] <= stop:
            return (stop / entry - 1.0) - COST_PCT
        if high[j] >= target:
            return (target / entry - 1.0) - COST_PCT
    return (close[end] / entry - 1.0) - COST_PCT


def _symbol_returns(bars) -> list:
    engine = ReversionEngine(ReversionConfig(min_history=60, enable_shorts=False))
    enriched = engine.calculate_indicators(bars)
    high = bars["high"].astype(float).to_numpy()
    low = bars["low"].astype(float).to_numpy()
    close = bars["close"].astype(float).to_numpy()
    atr_col = enriched["atr"].to_numpy()
    out = []
    for i in range(len(enriched)):
        if engine.get_decision(enriched.iloc[max(0, i - 1): i + 1]).signal != "LONG_REVERSION":
            continue
        c = close[i]
        atr = max(float(atr_col[i]) if not np.isnan(atr_col[i]) else 0.0, c * ATR_FLOOR_PCT)
        out.append(_outcome(high, low, close, i, c, atr, STD))
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


def _run_universe(symbols, windows, timeframe) -> dict:
    """Return {window_label: pooled_stats, 'names': n_traded} for one universe."""
    pooled = {label: [] for label in windows}
    names_traded = 0
    for sym in symbols:
        traded_any = False
        for label, (a, b) in windows.items():
            try:
                bars = fetch_alpaca_bars(sym, a, b, timeframe)
            except Exception as e:
                print(f"  # {sym} [{label}]: {e}")
                continue
            if bars is None or len(bars) < 60:
                continue
            rets = _symbol_returns(bars)
            pooled[label] += rets
            traded_any = traded_any or bool(rets)
        names_traded += int(traded_any)
    out = {label: _stats(rets) for label, rets in pooled.items()}
    out["names"] = names_traded
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Universe comparison backtest (research-only)")
    parser.add_argument("--universe", choices=list(UNIVERSES) + ["all"], default="all")
    parser.add_argument("--days", type=int, default=180, help="window length in days (default 180)")
    parser.add_argument("--end", default=None, help="in-sample end YYYY-MM-DD (default today)")
    parser.add_argument("--timeframe", default="5Min")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    end = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.now()
    is_start = end - timedelta(days=args.days)
    oos_start = end - timedelta(days=2 * args.days)
    fmt = "%Y-%m-%d"
    windows = {
        "in-sample": (is_start.strftime(fmt), end.strftime(fmt)),
        "oos":       (oos_start.strftime(fmt), is_start.strftime(fmt)),
    }
    chosen = list(UNIVERSES) if args.universe == "all" else [args.universe]
    print(f"Universe comparison | {args.timeframe} | in-sample {windows['in-sample'][0]}->"
          f"{windows['in-sample'][1]} | oos {windows['oos'][0]}->{windows['oos'][1]}")
    print(f"\n  {'UNIVERSE':<10}{'WINDOW':<11}{'names':>6}{'N':>7}{'NET':>10}{'EXP':>9}{'PF':>7}{'WIN':>8}")
    print("  " + "-" * 68)

    summary = {}
    for uni in chosen:
        res = _run_universe(UNIVERSES[uni], windows, args.timeframe)
        summary[uni] = res
        for label in ("in-sample", "oos"):
            s = res[label]
            print(f"  {uni:<10}{label:<11}{res['names']:>6}{s['n']:>7}{s['net']:>10.4f}"
                  f"{s['expectancy']*100:>8.3f}%{s['pf']:>7.2f}{s['win']*100:>7.1f}%")
        print()

    print("  VERDICT KEY: a universe is credible only if EXP/PF improve over core in")
    print("  BOTH windows. Trimmed that just removes losers -> closer to breakeven;")
    print("  highvol that's positive in both -> a real universe change worth pursuing.")


if __name__ == "__main__":
    main()
