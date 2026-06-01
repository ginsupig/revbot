"""Walk-forward backtest report (read-only).

Runs the same walk-forward backtest the autotuner uses — tune on each in-sample
fold, measure out-of-sample — and prints per-symbol and aggregate OOS metrics.
Unlike ``autotune_run.py`` this does NOT write anything to ``.env``; it's purely
for looking at how the strategy holds up out of sample.

Usage:
    python walkforward_report.py                 # TRADE_SYMBOL or leveraged default, 60d
    python walkforward_report.py TQQQ SOXL       # explicit symbols
    python walkforward_report.py --days 90 --splits 4
    python walkforward_report.py --timeframe 5Min TQQQ

Reads Alpaca creds from the environment / .env (APCA_API_KEY_ID, etc.).
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta

import numpy as np
from dotenv import load_dotenv

from reversion_bot.walkforward import run_walkforward_backtest
from run_real_backtest import fetch_alpaca_bars

# Same live-aligned grid as autotune_run.py: tune deep-oversold RI x VWAP
# extension (the dimensions that matter for fast leveraged ETFs).
PARAM_GRID = {
    "band_length":            [20],
    "ri_threshold":           [-1.0, -0.5, 0.0, 0.5],
    "use_vwap_filter":        [True, False],
    "max_vwap_extension_pct": [0.012, 0.020, 0.030],
    "rsi_max":                [40, 48, 55],
    "min_history":            [160],
}

DEFAULT_SYMBOLS = [
    "TQQQ", "SOXL", "TECL", "NVDL", "TSLL",
    "AAPU", "METU", "GGLL", "MSFU", "AMZU",
]


def resolve_symbols(cli_symbols: list[str]) -> list[str]:
    if cli_symbols:
        return [s.strip().upper() for s in cli_symbols if s.strip()]
    env_symbols = os.getenv("TRADE_SYMBOL", "")
    if env_symbols:
        return [s.strip().upper() for s in env_symbols.split(",") if s.strip()]
    return DEFAULT_SYMBOLS


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward backtest report (read-only)")
    parser.add_argument("symbols", nargs="*", help="symbols to test (default: TRADE_SYMBOL or leveraged set)")
    parser.add_argument("--days", type=int, default=60, help="lookback window in days (default 60)")
    parser.add_argument("--timeframe", default="5Min", help="bar timeframe (default 5Min)")
    parser.add_argument("--splits", type=int, default=3, help="walk-forward folds (default 3)")
    args = parser.parse_args()

    load_dotenv()
    symbols = resolve_symbols(args.symbols)

    end_dt = datetime.now()
    start = (end_dt - timedelta(days=args.days)).strftime("%Y-%m-%d")
    end = end_dt.strftime("%Y-%m-%d")

    print(f"Walk-forward backtest | symbols={len(symbols)} | window={args.days}d "
          f"({start} → {end}) | timeframe={args.timeframe} | folds={args.splits}")

    scores = {}  # symbol -> (avg_pf, avg_sharpe, avg_dd)
    for symbol in symbols:
        try:
            print(f"\n--- {symbol} ---")
            bars = fetch_alpaca_bars(symbol, start, end, args.timeframe)
            if bars is None or len(bars) < 160:
                print(f"Skipping {symbol}: insufficient data ({0 if bars is None else len(bars)} bars).")
                continue
            print(f"Fetched {len(bars)} bars.")

            _, oos_metrics, best_params = run_walkforward_backtest(
                bars, param_grid=PARAM_GRID, n_splits=args.splits
            )
            if oos_metrics:
                scores[symbol] = (
                    float(np.mean([m["profit_factor"] for m in oos_metrics])),
                    float(np.mean([m["sharpe"] for m in oos_metrics])),
                    float(np.mean([m["max_drawdown"] for m in oos_metrics])),
                )
                print(f"Best params: {best_params}")
        except Exception as e:
            print(f"Error backtesting {symbol}: {e}")

    if not scores:
        print("\nNo results.")
        return

    print("\n=== Out-of-sample summary (avg across folds) ===")
    print(f"  {'SYMBOL':<8}{'OOS PF':>9}{'SHARPE':>9}{'MAXDD':>10}")
    print(f"  {'-'*8:<8}{'-'*8:>9}{'-'*8:>9}{'-'*9:>10}")
    for sym, (pf, sr, dd) in sorted(scores.items(), key=lambda kv: kv[1][0], reverse=True):
        print(f"  {sym:<8}{pf:>9.2f}{sr:>9.2f}{dd:>10.4f}")

    pfs = [v[0] for v in scores.values()]
    srs = [v[1] for v in scores.values()]
    print(f"\n  Median OOS PF: {np.median(pfs):.2f}   Median OOS Sharpe: {np.median(srs):.2f}")


if __name__ == "__main__":
    main()
