"""A/B backtest: do the long-side exhaustion features actually improve entries?

The reversion engine already measures "price is stretched" (RSI/RI/Bollinger).
These features measure the orthogonal thing — "is the selling dying?" — to
separate a drop that is *finishing* (buyable) from one that is *beginning* (a
falling knife). This harness replays the REAL engine LONG_REVERSION decision per
bar, then masks the entries by each feature, running every arm through one
identical ATR-bracket long sim:

    base   : every engine LONG_REVERSION entry (the current behavior)
    decel  : only when selling velocity is decelerating
    wick   : only on a capitulation candle (long lower wick + close near high)
    both   : decel AND wick

Research-only: imports the live engine + the pure features, changes nothing in
the live path. Entries are a subset of base in every arm (features only ever
*remove* trades), so the question each arm answers is "did the trades it cut
lose more than they made?" — i.e. is the feature net-additive. Validate OOS
(via --end) before wiring anything into the engine.

Usage:
    python exhaustion_feature_ab_backtest.py                 # core universe, 20d
    python exhaustion_feature_ab_backtest.py WDC SMCI AMD --days 30
    python exhaustion_feature_ab_backtest.py --end 2026-05-24 --days 20   # OOS
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import pandas as pd

from reversion_bot.config import ReversionConfig
from reversion_bot.engine import ReversionEngine
from reversion_bot.exhaustion import selling_velocity_decelerating, capitulation_candle
from reversion_bot.analytics import profit_factor, sharpe_ratio
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

ARMS = ("base", "decel", "wick", "both")


class _Trade:
    __slots__ = ("entry", "stop", "target", "bars_held")

    def __init__(self, entry, stop, target):
        self.entry, self.stop, self.target, self.bars_held = entry, stop, target, 0


def _simulate(bars, short, long, min_wick, min_close_pos) -> dict:
    """Return {arm: [pnl...]} for one symbol. Live default engine config so the
    features are measured on top of the current (trend-filter-on) entries."""
    engine = ReversionEngine(ReversionConfig(min_history=60, enable_shorts=False))
    enriched = engine.calculate_indicators(bars)

    decel = selling_velocity_decelerating(bars, short=short, long=long).to_numpy()
    capit = capitulation_candle(bars, min_wick=min_wick, min_close_pos=min_close_pos).to_numpy()

    pnls = {a: [] for a in ARMS}
    open_trade = {a: None for a in ARMS}

    for i in range(len(enriched)):
        row = enriched.iloc[i]
        close, high, low = float(row["close"]), float(row["high"]), float(row["low"])

        for arm, tr in list(open_trade.items()):
            if tr is None:
                continue
            tr.bars_held += 1
            exit_px = None
            if low <= tr.stop:
                exit_px = tr.stop
            elif high >= tr.target:
                exit_px = tr.target
            elif tr.bars_held >= MAX_HOLD_BARS:
                exit_px = close
            if exit_px is not None:
                pnls[arm].append((exit_px / tr.entry - 1.0) - COST_PCT)
                open_trade[arm] = None

        if all(open_trade[a] is not None for a in ARMS):
            continue
        window = enriched.iloc[max(0, i - 1): i + 1]
        if engine.get_decision(window).signal != "LONG_REVERSION":
            continue
        atr = max(float(row["atr"]) if pd.notna(row["atr"]) else 0.0, close * ATR_FLOOR_PCT)
        allow = {
            "base": True,
            "decel": bool(decel[i]),
            "wick": bool(capit[i]),
            "both": bool(decel[i] and capit[i]),
        }
        for arm in ARMS:
            if open_trade[arm] is None and allow[arm]:
                open_trade[arm] = _Trade(
                    close, close - atr * STOP_ATR_MULTIPLE, close + atr * TARGET_ATR_MULTIPLE)
    return pnls


def _metrics(pnls) -> tuple:
    df = pd.DataFrame({"pnl": pnls}) if pnls else pd.DataFrame({"pnl": [0.0, 0.0]})
    return len(pnls), float(df["pnl"].sum()), profit_factor(df), sharpe_ratio(df)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exhaustion-feature entry A/B (research-only)")
    parser.add_argument("symbols", nargs="*", help="symbols (default: core universe)")
    parser.add_argument("--days", type=int, default=20, help="lookback in days (default 20)")
    parser.add_argument("--end", default=None, help="window end YYYY-MM-DD (default today; for OOS)")
    parser.add_argument("--timeframe", default="5Min")
    parser.add_argument("--short", type=int, default=3, help="deceleration short window (bars)")
    parser.add_argument("--long", type=int, default=10, help="deceleration long window (bars)")
    parser.add_argument("--min-wick", type=float, default=0.5, help="min lower-wick fraction")
    parser.add_argument("--min-close-pos", type=float, default=0.5, help="min close position in range")
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
    print(f"Exhaustion-feature A/B (long-only) | symbols={len(symbols)} | "
          f"{start}->{end_s} | {args.timeframe} | decel={args.short}/{args.long} "
          f"wick>={args.min_wick}")

    tot = {a: [] for a in ARMS}
    print(f"\n  {'SYMBOL':<8}" + "".join(f"{a+' net':>11}" for a in ARMS) + f"{'base/both n':>13}")
    for sym in symbols:
        try:
            bars = fetch_alpaca_bars(sym, start, end_s, args.timeframe)
            if bars is None or len(bars) < 60:
                print(f"  {sym:<8}  (insufficient data)")
                continue
            pnls = _simulate(bars, args.short, args.long, args.min_wick, args.min_close_pos)
            for a in ARMS:
                tot[a] += pnls[a]
            m = {a: _metrics(pnls[a]) for a in ARMS}
            counts = f"{m['base'][0]}/{m['both'][0]}"
            print(f"  {sym:<8}" + "".join(f"{m[a][1]:>11.4f}" for a in ARMS)
                  + f"{counts:>13}")
        except Exception as e:
            print(f"  {sym:<8}  error: {e}")

    print("\n=== Universe totals ===")
    base_net = _metrics(tot["base"])[1]
    for a in ARMS:
        n, net, pf, sr = _metrics(tot[a])
        delta = f"{net - base_net:+.4f}" if a != "base" else "  (ref)"
        print(f"  {a:<6} trades={n:<5} net={net:>9.4f}  PF={pf:.2f}  Sharpe={sr:>6.2f}  Δvs base {delta}")
    print("\n  research-only; features NOT wired into the engine. A positive Δ on")
    print("  fewer trades means the feature cut net-losing entries. Validate OOS (--end).")


if __name__ == "__main__":
    main()
