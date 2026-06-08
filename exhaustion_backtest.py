"""Walk-forward backtest for the bearish-exhaustion SHORT signal (read-only).

Validates the "higher highs on fading RVOL" edge per symbol across the universe
before it's ever wired into the live bot. Tunes the signal in-sample on each
fold and measures out-of-sample PF / Sharpe / MaxDD — same protocol as
walkforward_report.py, but driving short_exhaustion_strategy.

By default it runs your vetted CORE universe plus a few EDGE names — borderline
tickers worth stress-testing (defaults to the churn-adds previously dropped).
Edge names are tagged with * in the output.

Two guards against fooling yourself with a noisy single window:
    --stability   run TWO non-overlapping windows (recent + the prior period of
                  the same length) and flag only names that clear --min-pf in
                  BOTH. The cheapest defense against selection bias.
    --borrow-bps  add a per-trade short borrow/slippage haircut (bps) so the
                  backtest doesn't credit shorts with costs they won't get live.

Usage:
    python exhaustion_backtest.py                          # CORE + EDGE, 60d
    python exhaustion_backtest.py --stability --borrow-bps 10
    python exhaustion_backtest.py NVDA SMCI MSTR --days 90
    python exhaustion_backtest.py --no-edge --min-pf 1.15

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

# Ablation grid: trimmed on the original dimensions so the new quality filters
# (confirm_mode, cmf_max, min_extension_atr) can vary without exploding the
# search. Each new dimension carries an "off" value so the tuner can choose to
# ignore it — that's the ablation: if a filter never wins, it isn't helping.
PARAM_GRID = {
    "rvol_max": [0.8, 1.0],
    "hh_lookback": [20],
    "require_divergence": [True],
    "target_atr_multiple": [2.0, 3.0],
    "confirm_mode": ["none", "medium", "strong"],
    "cmf_max": [None, 0.0],
    "min_extension_atr": [0.0, 1.5],
}


def resolve_symbols(cli_symbols: list[str], include_edge: bool) -> tuple[list[str], set]:
    if cli_symbols:
        return [s.strip().upper() for s in cli_symbols if s.strip()], set()
    edge = set(EDGE_NAMES) if include_edge else set()
    return CORE_UNIVERSE + (EDGE_NAMES if include_edge else []), edge


def _trade_count(results) -> int:
    return int(sum(int((res["signal"] == -1).sum()) for res in results))


def backtest_symbol(symbol, start, end, timeframe, splits, fixed_kwargs):
    """Run the exhaustion walkforward for one symbol/window.

    Returns (pf, sharpe, dd, trades) averaged across OOS folds, or None if the
    symbol can't be backtested (insufficient data / fetch error).
    """
    try:
        bars = fetch_alpaca_bars(symbol, start, end, timeframe)
    except Exception as e:
        print(f"  {symbol}: fetch error ({e})")
        return None
    if bars is None or len(bars) < 160:
        print(f"  {symbol}: insufficient data ({0 if bars is None else len(bars)} bars)")
        return None
    results, oos, _best = run_exhaustion_walkforward(
        bars, param_grid=PARAM_GRID, n_splits=splits, verbose=False, fixed_kwargs=fixed_kwargs
    )
    if not oos:
        return None
    return (
        float(np.mean([m["profit_factor"] for m in oos])),
        float(np.mean([m["sharpe"] for m in oos])),
        float(np.mean([m["max_drawdown"] for m in oos])),
        _trade_count(results),
    )


def _window(days_back_start: int, days_back_end: int) -> tuple[str, str]:
    now = datetime.now()
    start = (now - timedelta(days=days_back_start)).strftime("%Y-%m-%d")
    end = (now - timedelta(days=days_back_end)).strftime("%Y-%m-%d")
    return start, end


def main() -> None:
    parser = argparse.ArgumentParser(description="Exhaustion-short walk-forward backtest (read-only)")
    parser.add_argument("symbols", nargs="*", help="symbols to test (default: CORE + EDGE)")
    parser.add_argument("--days", type=int, default=60, help="lookback window in days (default 60)")
    parser.add_argument("--timeframe", default="5Min", help="bar timeframe (default 5Min)")
    parser.add_argument("--splits", type=int, default=3, help="walk-forward folds (default 3)")
    parser.add_argument("--no-edge", action="store_true", help="core universe only (skip EDGE names)")
    parser.add_argument("--min-pf", type=float, default=1.10, help="survivor threshold on OOS PF")
    parser.add_argument("--borrow-bps", type=float, default=0.0,
                        help="extra per-trade short borrow/slippage haircut, in bps (default 0)")
    parser.add_argument("--stability", action="store_true",
                        help="run recent AND prior non-overlapping windows; flag names that survive both")
    args = parser.parse_args()

    load_dotenv()
    symbols, edge = resolve_symbols(args.symbols, include_edge=not args.no_edge)
    fixed_kwargs = {"borrow_cost_pct": args.borrow_bps / 10_000.0} if args.borrow_bps else None

    haircut = f" | borrow haircut={args.borrow_bps:.0f}bps" if args.borrow_bps else ""
    print(f"Exhaustion-short walk-forward | symbols={len(symbols)} ({len(edge)} edge*) "
          f"| window={args.days}d | timeframe={args.timeframe} | folds={args.splits}{haircut}")

    if args.stability:
        _run_stability(args, symbols, edge, fixed_kwargs)
    else:
        _run_single(args, symbols, edge, fixed_kwargs)


def _run_single(args, symbols, edge, fixed_kwargs) -> None:
    start, end = _window(args.days, 0)
    print(f"  window: {start} → {end}\n")
    scores = {}
    for sym in symbols:
        m = backtest_symbol(sym, start, end, args.timeframe, args.splits, fixed_kwargs)
        if m:
            scores[sym] = m
            print(f"  {sym}{'*' if sym in edge else ''}: PF={m[0]:.2f} Sharpe={m[1]:.2f} trades={m[3]}")
    if not scores:
        print("\nNo results.")
        return
    print("\n=== Out-of-sample summary (avg across folds; * = edge name) ===")
    print(f"  {'SYMBOL':<9}{'OOS PF':>9}{'SHARPE':>9}{'MAXDD':>10}{'TRADES':>8}  SURVIVES")
    print(f"  {'-'*9:<9}{'-'*8:>9}{'-'*8:>9}{'-'*9:>10}{'-'*7:>8}  {'-'*8}")
    for sym, (pf, sr, dd, n) in sorted(scores.items(), key=lambda kv: kv[1][0], reverse=True):
        tag = f"{sym}*" if sym in edge else sym
        print(f"  {tag:<9}{pf:>9.2f}{sr:>9.2f}{dd:>10.4f}{n:>8}  {'yes' if pf >= args.min_pf else ''}")
    pfs = [v[0] for v in scores.values()]
    survivors = sorted(s for s, v in scores.items() if v[0] >= args.min_pf)
    print(f"\n  Median OOS PF: {np.median(pfs):.2f}")
    print(f"  Survivors (PF >= {args.min_pf:.2f}): {', '.join(survivors) or 'none'}")


def _run_stability(args, symbols, edge, fixed_kwargs) -> None:
    recent = _window(args.days, 0)
    prior = _window(2 * args.days, args.days)   # the period BEFORE the recent one
    print(f"  recent: {recent[0]} → {recent[1]}   prior: {prior[0]} → {prior[1]}\n")

    rows = {}
    for sym in symbols:
        print(f"  {sym}{'*' if sym in edge else ''} ...")
        r = backtest_symbol(sym, recent[0], recent[1], args.timeframe, args.splits, fixed_kwargs)
        p = backtest_symbol(sym, prior[0], prior[1], args.timeframe, args.splits, fixed_kwargs)
        if r or p:
            rows[sym] = (r, p)
    if not rows:
        print("\nNo results.")
        return

    print("\n=== Window-stability (PF in each window; STABLE = clears min-pf in BOTH) ===")
    print(f"  {'SYMBOL':<9}{'PF recent':>11}{'PF prior':>10}{'STABLE':>9}")
    print(f"  {'-'*9:<9}{'-'*10:>11}{'-'*9:>10}{'-'*8:>9}")

    def pf_of(m):
        return m[0] if m else float("nan")

    # Sort by the worse of the two windows so the most robust names rise.
    def robustness(item):
        r, p = item[1]
        return min(pf_of(r), pf_of(p))

    stable = []
    for sym, (r, p) in sorted(rows.items(), key=robustness, reverse=True):
        pr, pp = pf_of(r), pf_of(p)
        both = (r is not None and p is not None and pr >= args.min_pf and pp >= args.min_pf)
        if both:
            stable.append(sym)
        tag = f"{sym}*" if sym in edge else sym
        rs = f"{pr:.2f}" if r else "  n/a"
        ps = f"{pp:.2f}" if p else "  n/a"
        print(f"  {tag:<9}{rs:>11}{ps:>10}{'STABLE' if both else '':>9}")

    print(f"\n  STABLE survivors (PF >= {args.min_pf:.2f} in BOTH windows): {', '.join(sorted(stable)) or 'none'}")
    print("  These are the only names with a defensible, regime-robust edge.")


if __name__ == "__main__":
    main()
