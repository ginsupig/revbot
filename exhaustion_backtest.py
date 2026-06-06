"""Walk-forward backtest for the bearish-exhaustion SHORT signal (read-only).

Validates the "higher highs on fading RVOL" edge per symbol across the universe
before it's ever wired into the live bot. Tunes the signal in-sample on each
fold and measures out-of-sample PF / Sharpe / MaxDD — same protocol as
walkforward_report.py, but driving short_exhaustion_strategy.

By default it runs your vetted CORE universe plus a few EDGE names — borderline
tickers worth stress-testing (defaults to the churn-adds previously dropped).
Edge names are tagged with * in the output so you can see whether the signal
survives on names you weren't sure about.

Usage:
    python exhaustion_backtest.py                      # CORE + EDGE, 60d, 5Min
    python exhaustion_backtest.py NVDA SMCI MSTR       # explicit symbols
    python exhaustion_backtest.py --days 90 --splits 4
    python exhaustion_backtest.py --no-edge            # core only
    python exhaustion_backtest.py --min-pf 1.10        # flag survivors

Reads Alpaca creds from the environment / .env (APCA_API_KEY_ID, etc.).
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta

import numpy as np
from dotenv import load_dotenv

from reversion_bot.walkforward import run_exhaustion_walkforward
from run_real_backtest import fetch_alpaca_bars

# Vetted core (kept in sync with the live TRADE_ALLOWLIST).
CORE_UNIVERSE = [
    "ASTS", "MU", "AMD", "NVDA", "SMCI", "META", "AAPL", "APP", "ARM", "CRWD",
    "MSFT", "LRCX", "JPM", "FCX", "OXY", "LLY", "CLF", "HIMS", "TEM",
]

# "On the edge" names — borderline tickers to stress-test the signal against.
# Defaults to the churn-adds previously dropped from the watchlist.
EDGE_NAMES = ["MSTR", "RKLB", "RDW", "QBTS"]

PARAM_GRID = {
    "rvol_max": [0.8, 1.0, 1.2],
    "hh_lookback": [10, 20],
    "require_divergence": [True, False],
    "target_atr_multiple": [2.0, 3.0],
}


def resolve_symbols(cli_symbols: list[str], include_edge: bool) -> tuple[list[str], set]:
    if cli_symbols:
        syms = [s.strip().upper() for s in cli_symbols if s.strip()]
        return syms, set()
    edge = set(EDGE_NAMES) if include_edge else set()
    return CORE_UNIVERSE + (EDGE_NAMES if include_edge else []), edge


def _trade_count(results) -> int:
    return int(sum(int((res["signal"] == -1).sum()) for res in results))


def main() -> None:
    parser = argparse.ArgumentParser(description="Exhaustion-short walk-forward backtest (read-only)")
    parser.add_argument("symbols", nargs="*", help="symbols to test (default: CORE + EDGE)")
    parser.add_argument("--days", type=int, default=60, help="lookback window in days (default 60)")
    parser.add_argument("--timeframe", default="5Min", help="bar timeframe (default 5Min)")
    parser.add_argument("--splits", type=int, default=3, help="walk-forward folds (default 3)")
    parser.add_argument("--no-edge", action="store_true", help="core universe only (skip EDGE names)")
    parser.add_argument("--min-pf", type=float, default=1.10, help="flag survivors at/above this OOS PF")
    args = parser.parse_args()

    load_dotenv()
    symbols, edge = resolve_symbols(args.symbols, include_edge=not args.no_edge)

    end_dt = datetime.now()
    start = (end_dt - timedelta(days=args.days)).strftime("%Y-%m-%d")
    end = end_dt.strftime("%Y-%m-%d")

    print(f"Exhaustion-short walk-forward | symbols={len(symbols)} "
          f"({len(edge)} edge*) | window={args.days}d ({start} → {end}) "
          f"| timeframe={args.timeframe} | folds={args.splits}")

    scores = {}  # symbol -> (avg_pf, avg_sharpe, avg_dd, trades, is_edge)
    for symbol in symbols:
        try:
            print(f"\n--- {symbol}{' *' if symbol in edge else ''} ---")
            bars = fetch_alpaca_bars(symbol, start, end, args.timeframe)
            if bars is None or len(bars) < 160:
                print(f"Skipping {symbol}: insufficient data ({0 if bars is None else len(bars)} bars).")
                continue
            print(f"Fetched {len(bars)} bars.")
            results, oos_metrics, best_params = run_exhaustion_walkforward(
                bars, param_grid=PARAM_GRID, n_splits=args.splits
            )
            if oos_metrics:
                scores[symbol] = (
                    float(np.mean([m["profit_factor"] for m in oos_metrics])),
                    float(np.mean([m["sharpe"] for m in oos_metrics])),
                    float(np.mean([m["max_drawdown"] for m in oos_metrics])),
                    _trade_count(results),
                    symbol in edge,
                )
                print(f"Best params: {best_params}")
        except Exception as e:
            print(f"Error backtesting {symbol}: {e}")

    if not scores:
        print("\nNo results.")
        return

    print("\n=== Out-of-sample summary (avg across folds; * = edge name) ===")
    print(f"  {'SYMBOL':<9}{'OOS PF':>9}{'SHARPE':>9}{'MAXDD':>10}{'TRADES':>8}  SURVIVES")
    print(f"  {'-'*9:<9}{'-'*8:>9}{'-'*8:>9}{'-'*9:>10}{'-'*7:>8}  {'-'*8}")
    for sym, (pf, sr, dd, n, is_edge) in sorted(scores.items(), key=lambda kv: kv[1][0], reverse=True):
        tag = f"{sym}*" if is_edge else sym
        survives = "yes" if pf >= args.min_pf else ""
        print(f"  {tag:<9}{pf:>9.2f}{sr:>9.2f}{dd:>10.4f}{n:>8}  {survives}")

    pfs = [v[0] for v in scores.values()]
    srs = [v[1] for v in scores.values()]
    survivors = [s for s, v in scores.items() if v[0] >= args.min_pf]
    print(f"\n  Median OOS PF: {np.median(pfs):.2f}   Median OOS Sharpe: {np.median(srs):.2f}")
    print(f"  Survivors (PF >= {args.min_pf:.2f}): {', '.join(sorted(survivors)) or 'none'}")


if __name__ == "__main__":
    main()
