"""Volume-threshold SWEEP for the exhaustion exit-modification setup.

The single rvol>=2.5 cut gave a tantalizing but n=14 result. The right next step
is not to wire wide brackets — it's to find the rvol threshold that keeps a PF
lift while holding N>100 (a sweet spot, not the highest-PF-at-tiny-N point). This
sweeps rvol in {1.5, 2.0, 2.5, 3.0, 4.0} in a single data pass and reports, for
each, the exhaustion group's size, per-bracket expectancy/PF, and the
wide-vs-standard differential against non-exhaustion (the control from #71).

It runs the sweep twice:
    Table A: exhaustion = decel & wick & volume(>=rvol)
    Table B: exhaustion = decel & wick & volume(>=rvol) & range(>= --range-mult ATR)

so you can see whether RANGE EXPANSION (the big-candle leg of real capitulation)
sharpens the edge or just thins the sample.

Per-setup (occupancy-free) expectancy, same as exhaustion_exit_ab_backtest.py;
research-only, nothing wired into the engine. Read the sweet spot, then validate
that exact definition OOS (--end) before anything proceeds.

Usage:
    python exhaustion_volume_sweep_backtest.py --days 30
    python exhaustion_volume_sweep_backtest.py --range-mult 1.5 --days 180
    python exhaustion_volume_sweep_backtest.py --end 2026-05-24 --days 20
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from reversion_bot.config import ReversionConfig
from reversion_bot.engine import ReversionEngine
from reversion_bot.exhaustion import (
    selling_velocity_decelerating,
    lower_wick_fraction,
    close_position,
    relative_volume,
    range_atr_ratio,
)
from run_real_backtest import fetch_alpaca_bars

CORE_UNIVERSE = [
    "ASTS", "MU", "AMD", "NVDA", "SMCI", "META", "AAPL", "APP", "ARM", "CRWD",
    "WDC", "MSFT", "LRCX", "JPM", "FCX", "OXY", "LLY", "CLF", "HIMS", "TEM",
]

ATR_FLOOR_PCT = 0.0035
COST_PCT = 0.0008
STD = dict(stop=1.20, target=2.00, hold=78)
WIDE = dict(stop=1.50, target=2.50, hold=120)
DEFAULT_RVOLS = [1.5, 2.0, 2.5, 3.0, 4.0]

# Record layout: (ret_std, ret_wide, decel, wick, rvol_value, range_ratio)
R_STD, R_WIDE, R_DECEL, R_WICK, R_RVOL, R_RANGE = range(6)


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


def _simulate(bars, short, long, min_wick) -> list:
    engine = ReversionEngine(ReversionConfig(min_history=60, enable_shorts=False))
    enriched = engine.calculate_indicators(bars)

    decel = selling_velocity_decelerating(bars, short=short, long=long).to_numpy()
    wick = ((lower_wick_fraction(bars) >= min_wick) & (close_position(bars) >= 0.5)).to_numpy()
    rvol = relative_volume(bars["volume"], 20).to_numpy()
    rng_ratio = range_atr_ratio(bars, enriched["atr"]).to_numpy()

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
        out.append((
            _outcome(high, low, close, i, c, atr, STD),
            _outcome(high, low, close, i, c, atr, WIDE),
            bool(decel[i]), bool(wick[i]),
            float(rvol[i]) if not np.isnan(rvol[i]) else 0.0,
            float(rng_ratio[i]) if not np.isnan(rng_ratio[i]) else 0.0,
        ))
    return out


def _pf(a):
    w, l = a[a > 0].sum(), -a[a < 0].sum()
    return float(w / l) if l > 0 else (10.0 if w > 0 else 0.0)


def _delta_exp(rows):
    if not rows:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    std = np.array([r[R_STD] for r in rows])
    wide = np.array([r[R_WIDE] for r in rows])
    return (float(std.mean()), float(wide.mean()), float(wide.mean() - std.mean()),
            _pf(std), _pf(wide))


def _sweep(records, rvols, require_range, range_mult) -> None:
    for thr in rvols:
        def is_exh(r):
            ok = r[R_DECEL] and r[R_WICK] and r[R_RVOL] >= thr
            return ok and (r[R_RANGE] >= range_mult if require_range else True)

        exh = [r for r in records if is_exh(r)]
        non = [r for r in records if not is_exh(r)]
        e_std, e_wide, e_d, e_pf_std, e_pf_wide = _delta_exp(exh)
        _, _, n_d, _, _ = _delta_exp(non)
        diff = e_d - n_d
        print(f"  {thr:>5.1f}{len(exh):>7}{e_std*100:>10.3f}%{e_wide*100:>10.3f}%"
              f"{e_pf_std:>8.2f}{e_pf_wide:>9.2f}{diff*100:>14.3f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exhaustion volume-threshold sweep (research-only)")
    parser.add_argument("symbols", nargs="*", help="symbols (default: core universe)")
    parser.add_argument("--days", type=int, default=20, help="lookback in days (default 20)")
    parser.add_argument("--end", default=None, help="window end YYYY-MM-DD (default today; for OOS)")
    parser.add_argument("--timeframe", default="5Min")
    parser.add_argument("--short", type=int, default=3, help="deceleration short window")
    parser.add_argument("--long", type=int, default=10, help="deceleration long window")
    parser.add_argument("--min-wick", type=float, default=0.5, help="lower-wick fraction")
    parser.add_argument("--range-mult", type=float, default=1.5, help="range>=mult*ATR for Table B")
    parser.add_argument("--rvols", type=float, nargs="*", default=DEFAULT_RVOLS, help="rvol thresholds")
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
    print(f"Volume-threshold sweep (long-only) | symbols={len(symbols)} | "
          f"{start}->{end_s} | {args.timeframe}  std={STD['stop']}/{STD['target']} "
          f"wide={WIDE['stop']}/{WIDE['target']}")

    records = []
    for sym in symbols:
        try:
            bars = fetch_alpaca_bars(sym, start, end_s, args.timeframe)
            if bars is None or len(bars) < 60:
                continue
            records += _simulate(bars, args.short, args.long, args.min_wick)
        except Exception as e:
            print(f"  {sym:<8}  error: {e}")

    if not records:
        print("\n  no signals in window.")
        return

    base_std = np.array([r[R_STD] for r in records])
    print(f"\n  signals={len(records)}   base PF(std)={_pf(base_std):.2f}   "
          f"base EXP(std)={base_std.mean()*100:.3f}%")
    header = (f"  {'RVOL':>5}{'N':>7}{'EXPstd':>11}{'EXPwide':>11}"
             f"{'PFstd':>8}{'PFwide':>9}{'DIFFERENTIAL':>14}")

    print("\n  Table A — exhaustion = decel & wick & volume(>=rvol)")
    print(header)
    _sweep(records, args.rvols, require_range=False, range_mult=args.range_mult)

    print(f"\n  Table B — + range expansion (range >= {args.range_mult} x ATR)")
    print(header)
    _sweep(records, args.rvols, require_range=True, range_mult=args.range_mult)

    print("\n  Sweet spot = N>100 AND PFwide>~1.2 AND DIFFERENTIAL>0 — not the highest")
    print("  PF at tiny N. If Table B holds N while raising PF/differential, range earns")
    print("  its place. Validate the chosen definition OOS (--end) before wiring.")


if __name__ == "__main__":
    main()
