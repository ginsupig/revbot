"""Exit-modification experiment: do exhaustion setups deserve a WIDER bracket?

The size ladder tested "does the feature predict magnitude at fixed exits?" (no).
This tests a different hypothesis: the decel+wick(+volume) setup catches *larger,
slower* reversions, so it should get more room — wider stop, higher target,
longer hold — rather than more size.

Per-setup design (occupancy-free): for EVERY engine LONG_REVERSION signal, the
same entry is simulated under BOTH bracket policies, so wide-vs-standard is a
clean within-entry contrast (no entry-timing confound). Because trades can
overlap, this measures per-setup EXPECTANCY, not a tradeable equity curve — if
the signal is positive, a sequenced equity test is the follow-up.

The control that matters: wider brackets may help *every* trade in a trending
window. So we compare the wide-vs-standard lift on EXHAUSTION setups against the
same lift on NON-exhaustion setups. Only the DIFFERENTIAL (exhaustion lift minus
non-exhaustion lift) is evidence the feature predicts duration — anything else is
just "this window liked wide stops."

    standard bracket : 1.20 ATR stop / 2.00 ATR target / 78-bar hold
    wide bracket     : 1.50 ATR stop / 2.50 ATR target / 120-bar hold
    exhaustion       : decel AND capitulation-wick  (+ volume climax if --require-volume)

Research-only; nothing wired into the engine. Validate OOS (--end).

Usage:
    python exhaustion_exit_ab_backtest.py --days 30
    python exhaustion_exit_ab_backtest.py --require-volume --rvol 2.5 --days 30
    python exhaustion_exit_ab_backtest.py --end 2026-05-24 --days 20
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


def _outcome(high, low, close, i, entry, atr, br) -> float:
    """Round-trip return for an entry at bar i under bracket ``br`` (ATR stop/
    target + time stop). Stop checked before target within a bar (conservative)."""
    stop = entry - atr * br["stop"]
    target = entry + atr * br["target"]
    end = min(i + br["hold"], len(close) - 1)
    for j in range(i + 1, end + 1):
        if low[j] <= stop:
            return (stop / entry - 1.0) - COST_PCT
        if high[j] >= target:
            return (target / entry - 1.0) - COST_PCT
    return (close[end] / entry - 1.0) - COST_PCT


def _simulate(bars, short, long, min_wick, rvol_thr) -> list:
    """Per-signal records: (ret_std, ret_wide, decel, wick, volume_climax)."""
    engine = ReversionEngine(ReversionConfig(min_history=60, enable_shorts=False))
    enriched = engine.calculate_indicators(bars)

    decel = selling_velocity_decelerating(bars, short=short, long=long).to_numpy()
    wick = ((lower_wick_fraction(bars) >= min_wick) & (close_position(bars) >= 0.5)).to_numpy()
    vol = (relative_volume(bars["volume"], 20) >= rvol_thr).fillna(False).to_numpy()

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
            bool(decel[i]), bool(wick[i]), bool(vol[i]),
        ))
    return out


def _group_stats(rows) -> dict:
    """Expectancy / PF / win-rate under each bracket for one group of setups."""
    if not rows:
        return dict(n=0, exp_std=0.0, exp_wide=0.0, d_exp=0.0,
                    pf_std=0.0, pf_wide=0.0, win_std=0.0, win_wide=0.0)
    std = np.array([r[0] for r in rows])
    wide = np.array([r[1] for r in rows])

    def pf(a):
        w, l = a[a > 0].sum(), -a[a < 0].sum()
        return float(w / l) if l > 0 else (10.0 if w > 0 else 0.0)

    return dict(
        n=len(rows),
        exp_std=float(std.mean()), exp_wide=float(wide.mean()),
        d_exp=float(wide.mean() - std.mean()),
        pf_std=pf(std), pf_wide=pf(wide),
        win_std=float((std > 0).mean()), win_wide=float((wide > 0).mean()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Exhaustion exit-modification experiment (research-only)")
    parser.add_argument("symbols", nargs="*", help="symbols (default: core universe)")
    parser.add_argument("--days", type=int, default=20, help="lookback in days (default 20)")
    parser.add_argument("--end", default=None, help="window end YYYY-MM-DD (default today; for OOS)")
    parser.add_argument("--timeframe", default="5Min")
    parser.add_argument("--short", type=int, default=3, help="deceleration short window")
    parser.add_argument("--long", type=int, default=10, help="deceleration long window")
    parser.add_argument("--min-wick", type=float, default=0.5, help="lower-wick fraction")
    parser.add_argument("--rvol", type=float, default=2.5, help="relative-volume climax threshold")
    parser.add_argument("--require-volume", action="store_true",
                        help="fold volume climax into the exhaustion definition (decel & wick & vol)")
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
    exh_def = "decel & wick" + (" & volume" if args.require_volume else "")
    print(f"Exit-modification experiment (long-only) | symbols={len(symbols)} | "
          f"{start}->{end_s} | {args.timeframe}")
    print(f"  std={STD['stop']}/{STD['target']}/{STD['hold']}  "
          f"wide={WIDE['stop']}/{WIDE['target']}/{WIDE['hold']}  exhaustion = {exh_def} "
          f"(rvol>={args.rvol})")

    rows = []
    for sym in symbols:
        try:
            bars = fetch_alpaca_bars(sym, start, end_s, args.timeframe)
            if bars is None or len(bars) < 60:
                continue
            rows += _simulate(bars, args.short, args.long, args.min_wick, args.rvol)
        except Exception as e:
            print(f"  {sym:<8}  error: {e}")

    if not rows:
        print("\n  no signals in window.")
        return

    def is_exh(r):
        return r[2] and r[3] and (r[4] if args.require_volume else True)

    groups = {
        "ALL": rows,
        "EXHAUSTION": [r for r in rows if is_exh(r)],
        "NON-EXH": [r for r in rows if not is_exh(r)],
    }
    print(f"\n  signals={len(rows)}   exhaustion={len(groups['EXHAUSTION'])}   "
          f"(decel={sum(r[2] for r in rows)} wick={sum(r[3] for r in rows)} "
          f"vol={sum(r[4] for r in rows)})")
    print(f"\n  {'GROUP':<12}{'N':>6}{'EXP std':>10}{'EXP wide':>10}{'Δexp':>9}"
          f"{'PF std':>8}{'PF wide':>9}{'win std':>9}{'win wide':>9}")
    stats = {}
    for name in ("ALL", "EXHAUSTION", "NON-EXH"):
        s = _group_stats(groups[name])
        stats[name] = s
        print(f"  {name:<12}{s['n']:>6}{s['exp_std']*100:>9.3f}%{s['exp_wide']*100:>9.3f}%"
              f"{s['d_exp']*100:>8.3f}%{s['pf_std']:>8.2f}{s['pf_wide']:>9.2f}"
              f"{s['win_std']*100:>8.1f}%{s['win_wide']*100:>8.1f}%")

    diff = stats["EXHAUSTION"]["d_exp"] - stats["NON-EXH"]["d_exp"]
    print(f"\n  DIFFERENTIAL (exhaustion Δexp - non-exh Δexp) = {diff*100:+.3f}%")
    print("  > 0 means wider brackets help exhaustion setups MORE than the rest —")
    print("    i.e. the feature predicts duration/room. <= 0 means it was just the")
    print("    window liking wide stops. Validate the sign holds OOS (--end).")


if __name__ == "__main__":
    main()
