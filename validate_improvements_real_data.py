"""Validate the trendfail fix + tune its parameters ON REAL MARKET DATA.

Run this from the repo root on your machine (where .env has Alpaca creds):

    python validate_improvements_real_data.py --symbol AAPL --start 2026-03-01 --end 2026-06-01

It will:
  1. Fetch real 5-min bars via your existing fetch_alpaca_bars.
  2. Run the OLD trendfail (extracted from git) and the NEW one on the same
     bars and print PF / Sharpe (correct intraday annualization) / total
     return / trade count side by side.
  3. Grid-scan the NEW signal's (confirmation_window, threshold) on a
     chronological train split and report each combo's OUT-OF-SAMPLE result
     on the held-out tail — pick parameters from the OOS column only.

Nothing here changes live behavior; it only reads data and prints.
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")

import numpy as np

from reversion_bot.analytics import profit_factor, sharpe_ratio
from reversion_bot.trendfail import trend_fail_continue_strategy as NEW
from run_real_backtest import fetch_alpaca_bars

PPY_5MIN = 78 * 252

# The last commit that still contained the vacuous-confirmation version.
OLD_TRENDFAIL_REF = "2707d0e:reversion_bot/trendfail.py"


def load_old_trendfail():
    src = subprocess.check_output(["git", "show", OLD_TRENDFAIL_REF], text=True)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        path = f.name
    spec = importlib.util.spec_from_file_location("old_trendfail", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.trend_fail_continue_strategy


def summarize(out) -> tuple[float, float, float, int]:
    return (
        profit_factor(out),
        sharpe_ratio(out, periods_per_year=PPY_5MIN),
        float(out["pnl"].sum()),
        int((out["pnl"] != 0).sum()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="AAPL")
    ap.add_argument("--start", default="2026-03-01")
    ap.add_argument("--end", default="2026-06-01")
    ap.add_argument("--timeframe", default="5Min")
    ap.add_argument("--train-frac", type=float, default=0.7)
    args = ap.parse_args()

    bars = fetch_alpaca_bars(args.symbol, args.start, args.end, args.timeframe)
    print(f"Fetched {len(bars)} {args.timeframe} bars for {args.symbol}")
    if len(bars) < 500:
        sys.exit("Not enough bars for a meaningful comparison.")

    OLD = load_old_trendfail()

    print("\n=== OLD vs NEW on the full sample ===")
    print(f"{'impl':<6}{'PF':>8}{'Sharpe':>10}{'totRet':>10}{'trades':>8}")
    for name, fn in (("OLD", OLD), ("NEW", NEW)):
        pf, sr, tot, n = summarize(fn(bars.copy(), window=20,
                                      confirmation_window=3, threshold=0.005))
        print(f"{name:<6}{pf:>8.2f}{sr:>10.2f}{tot:>10.4f}{n:>8d}")

    cut = int(len(bars) * args.train_frac)
    train, test = bars.iloc[:cut], bars.iloc[cut:]
    print(f"\n=== NEW: parameter scan | train={len(train)} bars, OOS test={len(test)} bars ===")
    print(f"{'cw':>4}{'th':>8} | {'train PF':>9}{'n':>5} | {'OOS PF':>8}{'OOS Sharpe':>11}{'OOS ret':>9}{'n':>5}")
    for cw in (3, 5, 8, 12):
        for th in (0.001, 0.003, 0.005):
            kw = dict(window=20, confirmation_window=cw, threshold=th)
            tr = summarize(NEW(train.copy(), **kw))
            te = summarize(NEW(test.copy(), **kw))
            print(f"{cw:>4}{th:>8} | {tr[0]:>9.2f}{tr[3]:>5d} | "
                  f"{te[0]:>8.2f}{te[1]:>11.2f}{te[2]:>9.4f}{te[3]:>5d}")

    print("\nPick parameters from the OOS columns, then set "
          "trendfail_confirmation_window / trendfail_threshold in ReversionConfig.")


if __name__ == "__main__":
    main()
