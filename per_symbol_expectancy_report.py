"""Per-symbol expectancy report: which names actually carry edge?

Every regime/entry filter we tested is loss-mitigation; the base book is
net-negative on this 20-name universe. Before adding anything, the higher-value
question is which names to DROP. This replays the real engine LONG_REVERSION
decision per bar (fully filtered: the live default config has the per-symbol
trend filter on), simulates the standard bracket, and reports each symbol's
per-trade expectancy / PF / win-rate so the dead weight is visible.

Run it on both an in-sample and an OOS window (--end); a name worth keeping
should carry positive expectancy in BOTH, not just one. Per-setup (occupancy-free)
expectancy; research-only, nothing wired. The cut decision goes in the universe
list / allowlist, not the engine.

Usage:
    python per_symbol_expectancy_report.py --days 180
    python per_symbol_expectancy_report.py --end 2025-12-15 --days 180
    python per_symbol_expectancy_report.py --csv > names.csv
"""
from __future__ import annotations

import argparse
import csv as _csv
import io
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from reversion_bot.config import ReversionConfig
from reversion_bot.engine import ReversionEngine
from run_real_backtest import fetch_alpaca_bars

CORE_UNIVERSE = [
    "ASTS", "MU", "AMD", "NVDA", "SMCI", "META", "AAPL", "APP", "ARM", "CRWD",
    "WDC", "MSFT", "LRCX", "JPM", "FCX", "OXY", "LLY", "CLF", "HIMS", "TEM",
]

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-symbol expectancy report (research-only)")
    parser.add_argument("symbols", nargs="*", help="symbols (default: core universe)")
    parser.add_argument("--days", type=int, default=180, help="lookback in days (default 180)")
    parser.add_argument("--end", default=None, help="window end YYYY-MM-DD (default today; OOS)")
    parser.add_argument("--timeframe", default="5Min")
    parser.add_argument("--csv", action="store_true", help="emit CSV instead of a table")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    symbols = [s.strip().upper() for s in args.symbols] if args.symbols else CORE_UNIVERSE
    end = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.now()
    start = (end - timedelta(days=args.days)).strftime("%Y-%m-%d")
    end_s = end.strftime("%Y-%m-%d")

    rows = []
    all_returns = []
    for sym in symbols:
        try:
            bars = fetch_alpaca_bars(sym, start, end_s, args.timeframe)
            if bars is None or len(bars) < 60:
                rows.append((sym, _stats([])))
                continue
            rets = _symbol_returns(bars)
            all_returns += rets
            rows.append((sym, _stats(rets)))
        except Exception as e:
            print(f"# {sym}: error {e}")
            rows.append((sym, _stats([])))

    rows.sort(key=lambda r: r[1]["expectancy"])     # worst first — the cut candidates

    if args.csv:
        buf = io.StringIO()
        w = _csv.writer(buf)
        w.writerow(["symbol", "n", "net", "expectancy", "pf", "win"])
        for sym, s in rows:
            w.writerow([sym, s["n"], f"{s['net']:.4f}", f"{s['expectancy']:.6f}",
                        f"{s['pf']:.4f}", f"{s['win']:.4f}"])
        print(buf.getvalue(), end="")
        return

    traded = [s for _, s in rows if s["n"] > 0]
    total = _stats(all_returns)
    print(f"Per-symbol expectancy | symbols={len(symbols)} | {start}->{end_s} | {args.timeframe}")
    print(f"  (engine LONG_REVERSION, fully filtered, standard bracket — pre market-regime gate)")
    print(f"\n  {'SYMBOL':<8}{'N':>6}{'NET':>10}{'EXP':>10}{'PF':>7}{'WIN':>8}  FLAG")
    print(f"  {'-'*8:<8}{'-'*5:>6}{'-'*9:>10}{'-'*9:>10}{'-'*6:>7}{'-'*7:>8}  ----")
    for sym, s in rows:
        flag = "" if s["n"] == 0 else ("KEEP" if s["expectancy"] > 0 else "cut?")
        note = "(no trades)" if s["n"] == 0 else ""
        print(f"  {sym:<8}{s['n']:>6}{s['net']:>10.4f}{s['expectancy']*100:>9.3f}%"
              f"{s['pf']:>7.2f}{s['win']*100:>7.1f}%  {flag}{note}")
    pos = sum(1 for _, s in rows if s["expectancy"] > 0 and s["n"] > 0)
    print(f"\n  {pos}/{len(traded)} traded names carry positive expectancy this window.")
    print(f"  universe N={total['n']}  net={total['net']:.4f}  "
          f"EXP={total['expectancy']*100:.3f}%  PF={total['pf']:.2f}")
    print("  A name is a real KEEP only if it's positive in BOTH this and the OOS window.")


if __name__ == "__main__":
    main()
