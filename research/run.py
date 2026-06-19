"""One-command gated sweep: fetch -> holdout split -> reference sweep -> verdict.

Ties the pipeline together so you never write glue: it fetches bars per symbol,
splits each into a research head and a sacred holdout tail, runs the flush-bounce
reference sweep on both, pools per config, and runs the whole thing through
research.pipeline.evaluate_sweep -- printing a DEPLOY-CANDIDATE / REJECT verdict
with the evidence (deflated Sharpe, fold robustness, MC drawdown, cost headroom,
holdout). Swap in the VectorBT engine for a far bigger grid; the gating is the same.

Usage (on a machine with Alpaca creds in .env):
    python -m research.run --universe watchlist --days 365
    python -m research.run MU ASTS WDC --flush 0.03 0.05 0.07 --rsi 30 35 40
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np

from .engines.reference import sweep
from .pipeline import evaluate_sweep, PipelineResult

UNIVERSES = {
    "watchlist": ["ASTS", "MU", "AMD", "NVDA", "SMCI", "META", "APP", "ARM", "CRWD",
                  "MSFT", "LRCX", "FCX", "OXY", "CLF", "HIMS", "WDC", "COIN", "ADI",
                  "AMAT", "TSLA", "IONQ"],
    "keepers": ["ARM", "CRWD", "MU", "LRCX"],
}


def run_gated_sweep(fetch_fn, symbols, start, end, timeframe,
                    flush_grid, rsi_grid, drop_window=3, holdout_frac=0.25,
                    min_bars=80) -> PipelineResult:
    """Fetch each symbol via ``fetch_fn(sym, start, end, timeframe)``, split a
    holdout tail, sweep both, pool per config, and gate. ``fetch_fn`` injected so
    the core is testable without a broker."""
    research = defaultdict(list)
    holdout = defaultdict(list)
    for sym in symbols:
        try:
            bars = fetch_fn(sym, start, end, timeframe)
        except Exception as e:
            print(f"  # {sym}: {e}")
            continue
        if bars is None or len(bars) < min_bars:
            continue
        cut = int(round(len(bars) * (1.0 - holdout_frac)))
        rb, hb = bars.iloc[:cut], bars.iloc[cut:]
        for cfg, r in sweep(rb, flush_grid, rsi_grid, drop_window).items():
            research[cfg].extend(r.tolist())
        for cfg, r in sweep(hb, flush_grid, rsi_grid, drop_window).items():
            holdout[cfg].extend(r.tolist())
    research = {k: np.asarray(v, float) for k, v in research.items()}
    holdout = {k: np.asarray(v, float) for k, v in holdout.items()}
    return evaluate_sweep(research, holdout)


def format_report(res: PipelineResult) -> str:
    if res.chosen is None:
        return "  no trades / empty sweep."
    lines = [
        f"  trials swept   : {res.n_trials}",
        f"  best config    : {res.chosen}",
        f"  research Sharpe: {res.research_sharpe:.3f}   deflated: {res.deflated_sharpe:+.3f}"
        f"  (gate: {'PASS' if res.passed_gate else 'fail'})",
        f"  fold robust    : {res.robust}  folds={[round(s,2) for s in res.fold_sharpes]}",
        f"  breakeven cost : {res.breakeven_bps:.1f} bps",
        f"  holdout        : n={res.holdout.get('n',0)} mean={res.holdout.get('mean',0)*100:.3f}% "
        f"PF={res.holdout.get('pf',0):.2f}",
        f"  MC reshuffle   : prob_neg={res.mc.get('prob_negative',0):.2f} "
        f"maxDD_p95={res.mc.get('maxdd_p95',0)*100:.1f}%",
        "",
        f"  VERDICT: {res.verdict}",
    ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Gated flush-bounce sweep (research.pipeline)")
    p.add_argument("symbols", nargs="*", help="symbols (overrides --universe)")
    p.add_argument("--universe", choices=list(UNIVERSES), default="watchlist")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--end", default=None)
    p.add_argument("--timeframe", default="5Min")
    p.add_argument("--flush", type=float, nargs="*", default=[0.03, 0.05, 0.07])
    p.add_argument("--rsi", type=float, nargs="*", default=[30, 35, 40])
    p.add_argument("--drop-window", type=int, default=3)
    p.add_argument("--holdout", type=float, default=0.25)
    args = p.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    from run_real_backtest import fetch_alpaca_bars   # guarded: needs alpaca + creds

    symbols = [s.upper() for s in args.symbols] if args.symbols else UNIVERSES[args.universe]
    end = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.now()
    start = (end - timedelta(days=args.days)).strftime("%Y-%m-%d")
    end_s = end.strftime("%Y-%m-%d")
    print(f"Gated flush-bounce sweep | {len(symbols)} symbols | {start}->{end_s} | "
          f"{args.timeframe} | flush={args.flush} rsi={args.rsi} | holdout={args.holdout:.0%}")
    res = run_gated_sweep(fetch_alpaca_bars, symbols, start, end_s, args.timeframe,
                          args.flush, args.rsi, args.drop_window, args.holdout)
    print("\n" + format_report(res))


if __name__ == "__main__":
    main()
