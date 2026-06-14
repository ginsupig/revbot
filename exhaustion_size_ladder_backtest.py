"""Size-modifier experiment: does upsizing high-quality reversion entries improve
risk-adjusted return WITHOUT worsening the left tail?

The exhaustion-feature A/B showed decel/wick are poor *gates* (they fail OOS as
include/exclude filters). The better hypothesis — keep every signal, but size it
by quality — is tested here at the SIZING layer (where it would feed the
governor), never in the signal. Every engine LONG_REVERSION entry is taken; only
the position multiplier changes.

Three arms, all on the same entries:
    base   : every trade at 1.0x (current behavior)
    flat   : every trade at the ladder's AVERAGE multiplier (a pure-leverage
             control — same gross exposure as ladder, spread uniformly)
    ladder : graded multiplier by exhaustion quality (below)

``ladder`` vs ``flat`` is the real test: it is exposure-matched, so any edge is
the *selection*, not just leverage. The quality ladder:
    both strong  (deep decel + long wick) -> 1.50x
    both         (decel & capitulation)   -> 1.25x
    either       (decel | capitulation)   -> 1.10x
    neither                                -> 1.00x

Reports expectancy, PF, per-trade Sharpe, max drawdown, ulcer index and worst
losing streak per arm, so the question "more return, not more tail" is answered
directly. Research-only; nothing is wired into the engine. Validate across 3-4
windows (--end) before deciding.

Usage:
    python exhaustion_size_ladder_backtest.py --days 30
    python exhaustion_size_ladder_backtest.py --end 2026-05-24 --days 20   # OOS
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from reversion_bot.config import ReversionConfig
from reversion_bot.engine import ReversionEngine
from reversion_bot.exhaustion import selling_velocity_decelerating, lower_wick_fraction, close_position
from reversion_bot.analytics import ulcer_index, worst_losing_streak
from run_real_backtest import fetch_alpaca_bars

CORE_UNIVERSE = [
    "ASTS", "MU", "AMD", "NVDA", "SMCI", "META", "AAPL", "APP", "ARM", "CRWD",
    "WDC", "MSFT", "LRCX", "JPM", "FCX", "OXY", "LLY", "CLF", "HIMS", "TEM",
]

STOP_ATR_MULTIPLE = 1.20
TARGET_ATR_MULTIPLE = 2.00
ATR_FLOOR_PCT = 0.0035
COST_PCT = 0.0008
MAX_HOLD_BARS = 78

# Quality ladder multipliers.
MULT_BOTH_STRONG = 1.50
MULT_BOTH = 1.25
MULT_EITHER = 1.10
MULT_BASE = 1.00


class _Trade:
    __slots__ = ("entry", "stop", "target", "bars_held", "mult")

    def __init__(self, entry, stop, target, mult):
        self.entry, self.stop, self.target, self.bars_held, self.mult = entry, stop, target, 0, mult


def _quality_mult(decel, wick, decel_strong, wick_strong) -> float:
    if decel_strong and wick_strong:
        return MULT_BOTH_STRONG
    if decel and wick:
        return MULT_BOTH
    if decel or wick:
        return MULT_EITHER
    return MULT_BASE


def _simulate(bars, short, long, min_wick, strong_wick) -> list:
    """Return a list of (base_return, multiplier) for each closed trade.

    base_return is the raw per-trade return (the size-1.0 outcome); multiplier is
    the quality size captured at entry. Arms are built from these downstream.
    """
    engine = ReversionEngine(ReversionConfig(min_history=60, enable_shorts=False))
    enriched = engine.calculate_indicators(bars)

    decel = selling_velocity_decelerating(bars, short=short, long=long).to_numpy()
    wick_frac = lower_wick_fraction(bars).to_numpy()
    close_pos = close_position(bars).to_numpy()
    close = bars["close"].astype(float)
    ret_short = (close / close.shift(short) - 1.0).to_numpy()
    ret_long = (close / close.shift(long) - 1.0).to_numpy()

    wick = (wick_frac >= min_wick) & (close_pos >= 0.5)
    wick_strong = (wick_frac >= strong_wick) & (close_pos >= 0.5)
    # "Strong" deceleration: still down over the long window but already UP over
    # the short window -> the bounce has actually started (deepest exhaustion).
    decel_strong = (ret_long < 0) & (ret_short > 0)

    records = []
    open_trade = None
    for i in range(len(enriched)):
        row = enriched.iloc[i]
        c, high, low = float(row["close"]), float(row["high"]), float(row["low"])

        if open_trade is not None:
            open_trade.bars_held += 1
            exit_px = None
            if low <= open_trade.stop:
                exit_px = open_trade.stop
            elif high >= open_trade.target:
                exit_px = open_trade.target
            elif open_trade.bars_held >= MAX_HOLD_BARS:
                exit_px = c
            if exit_px is not None:
                records.append(((exit_px / open_trade.entry - 1.0) - COST_PCT, open_trade.mult))
                open_trade = None

        if open_trade is None and engine.get_decision(enriched.iloc[max(0, i - 1): i + 1]).signal == "LONG_REVERSION":
            mult = _quality_mult(bool(decel[i]), bool(wick[i]), bool(decel_strong[i]), bool(wick_strong[i]))
            atr = max(float(row["atr"]) if pd.notna(row["atr"]) else 0.0, c * ATR_FLOOR_PCT)
            open_trade = _Trade(c, c - atr * STOP_ATR_MULTIPLE, c + atr * TARGET_ATR_MULTIPLE, mult)
    return records


def _metrics(stream) -> dict:
    """Risk/return metrics for one size-weighted return stream."""
    arr = np.asarray(stream, dtype=float)
    if arr.size == 0:
        return dict(n=0, net=0.0, expectancy=0.0, pf=0.0, sharpe=0.0,
                    max_dd=0.0, ulcer=0.0, worst_streak=0)
    wins, losses = arr[arr > 0].sum(), -arr[arr < 0].sum()
    eq = np.cumprod(1.0 + arr)
    peak = np.maximum.accumulate(eq)
    max_dd = float(((eq - peak) / peak).min())
    std = arr.std(ddof=0)
    return dict(
        n=int(arr.size),
        net=float(arr.sum()),
        expectancy=float(arr.mean()),
        pf=float(wins / losses) if losses > 0 else (10.0 if wins > 0 else 0.0),
        sharpe=float(arr.mean() / std) if std > 0 else 0.0,   # per-trade Sharpe
        max_dd=max_dd,
        ulcer=ulcer_index(eq),
        worst_streak=worst_losing_streak(arr),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Exhaustion size-modifier ladder experiment (research-only)")
    parser.add_argument("symbols", nargs="*", help="symbols (default: core universe)")
    parser.add_argument("--days", type=int, default=20, help="lookback in days (default 20)")
    parser.add_argument("--end", default=None, help="window end YYYY-MM-DD (default today; for OOS)")
    parser.add_argument("--timeframe", default="5Min")
    parser.add_argument("--short", type=int, default=3, help="deceleration short window (bars)")
    parser.add_argument("--long", type=int, default=10, help="deceleration long window (bars)")
    parser.add_argument("--min-wick", type=float, default=0.5, help="lower-wick fraction for capitulation")
    parser.add_argument("--strong-wick", type=float, default=0.66, help="lower-wick fraction for STRONG")
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
    print(f"Size-ladder experiment (long-only) | symbols={len(symbols)} | "
          f"{start}->{end_s} | {args.timeframe} | decel={args.short}/{args.long} "
          f"wick>={args.min_wick}/{args.strong_wick}")

    records = []   # (base_return, multiplier) across the whole universe
    for sym in symbols:
        try:
            bars = fetch_alpaca_bars(sym, start, end_s, args.timeframe)
            if bars is None or len(bars) < 60:
                continue
            records += _simulate(bars, args.short, args.long, args.min_wick, args.strong_wick)
        except Exception as e:
            print(f"  {sym:<8}  error: {e}")

    if not records:
        print("\n  no trades in window.")
        return

    base_r = np.array([r for r, _ in records])
    mult = np.array([m for _, m in records])
    avg_mult = float(mult.mean())

    arms = {
        "base": base_r,                 # 1.0x
        "flat": base_r * avg_mult,      # uniform upsize (leverage control)
        "ladder": base_r * mult,        # quality-weighted upsize
    }

    print(f"\n  trades={len(records)}   avg ladder multiplier={avg_mult:.3f}   "
          f"(mult mix: 1.50x={int((mult==MULT_BOTH_STRONG).sum())} "
          f"1.25x={int((mult==MULT_BOTH).sum())} "
          f"1.10x={int((mult==MULT_EITHER).sum())} "
          f"1.00x={int((mult==MULT_BASE).sum())})")
    print(f"\n  {'ARM':<7}{'NET':>10}{'EXPECT':>10}{'PF':>7}{'SHARPE':>8}"
          f"{'MAXDD':>9}{'ULCER':>8}{'STREAK':>8}")
    for name in ("base", "flat", "ladder"):
        m = _metrics(arms[name])
        print(f"  {name:<7}{m['net']:>10.4f}{m['expectancy']*100:>9.3f}%{m['pf']:>7.2f}"
              f"{m['sharpe']:>8.3f}{m['max_dd']*100:>8.2f}%{m['ulcer']:>8.2f}{m['worst_streak']:>8}")

    print("\n  ladder vs base  = does quality-sizing beat 1.0x?")
    print("  ladder vs flat  = does it beat the SAME exposure spread uniformly (i.e. is it")
    print("                    selection, not just leverage)? This is the one that matters.")
    print("  Want: ladder >= flat on Sharpe/PF AND not worse on MAXDD/ulcer/streak. Validate OOS.")


if __name__ == "__main__":
    main()
