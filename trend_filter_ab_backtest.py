"""A/B backtest: per-symbol trend filter OFF vs ON (the gate enabled by
"Gate single-name downtrends", PR #65).

Live now defaults ``use_trend_filter=True``, which refuses a long-reversion
entry extended more than ``trend_filter_band_pct`` below the trend EMA — the
falling-knife case that produced the WDC bleed. ``walkforward.py`` still pins
the filter OFF so its historical results stay comparable, so this script is the
way to measure the filter's effect: it replays the REAL ``ReversionEngine``
decision per bar under both configs through one identical ATR-bracket long sim
and reports which trades better.

Research-only: imports the live engine and changes nothing in the live path.
Both arms share the same enriched bars; the ONLY difference is the trend gate
(and, on the ON arm, ``trend_filter_band_pct``), so the delta is attributable.

Long-only by design: the filter also gates shorts, but the WDC failure mode —
and the live default flip we're validating — is dip-buying, so we isolate it.

Usage:
    python trend_filter_ab_backtest.py                  # core universe, 20d, 5Min
    python trend_filter_ab_backtest.py WDC SMCI AMD --days 30
    python trend_filter_ab_backtest.py --band 0.02      # tighter veto
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import pandas as pd

from reversion_bot.config import ReversionConfig
from reversion_bot.engine import ReversionEngine
from reversion_bot.analytics import profit_factor, sharpe_ratio
from run_real_backtest import fetch_alpaca_bars

# Mirror blend_ab_backtest / walkforward so the sim is comparable across scripts.
CORE_UNIVERSE = [
    "ASTS", "MU", "AMD", "NVDA", "SMCI", "META", "AAPL", "APP", "ARM", "CRWD",
    "WDC", "MSFT", "LRCX", "JPM", "FCX", "OXY", "LLY", "CLF", "HIMS", "TEM",
]

STOP_ATR_MULTIPLE = 1.20
TARGET_ATR_MULTIPLE = 2.00
ATR_FLOOR_PCT = 0.0035
COST_PCT = 0.0008
MAX_HOLD_BARS = 78          # ~1 session of 5-min bars; constant across both arms


class _Trade:
    __slots__ = ("entry", "stop", "target", "bars_held")

    def __init__(self, entry, stop, target):
        self.entry, self.stop, self.target, self.bars_held = entry, stop, target, 0


def _simulate(bars, band_pct) -> dict:
    """Return {'off': [pnl...], 'on': [pnl...]} for one symbol.

    Indicators are filter-independent, so enrich once and run both engines over
    the same enriched frame; only the trend gate inside get_decision differs.
    A 2-row window is fed to get_decision (it reads only the last two bars), so
    the replay stays O(n) instead of O(n^2).
    """
    base = dict(min_history=60, enable_shorts=False)
    engines = {
        "off": ReversionEngine(ReversionConfig(use_trend_filter=False, **base)),
        "on": ReversionEngine(ReversionConfig(
            use_trend_filter=True, trend_filter_band_pct=band_pct, **base)),
    }
    enriched = engines["off"].calculate_indicators(bars)

    pnls = {"off": [], "on": []}
    open_trade = {"off": None, "on": None}

    for i in range(len(enriched)):
        row = enriched.iloc[i]
        close, high, low = float(row["close"]), float(row["high"]), float(row["low"])

        # --- exits: identical ATR bracket + time stop for both arms ---
        for arm, tr in list(open_trade.items()):
            if tr is None:
                continue
            tr.bars_held += 1
            exit_px = None
            if low <= tr.stop:
                exit_px = tr.stop                       # stop first (conservative)
            elif high >= tr.target:
                exit_px = tr.target
            elif tr.bars_held >= MAX_HOLD_BARS:
                exit_px = close
            if exit_px is not None:
                pnls[arm].append((exit_px / tr.entry - 1.0) - COST_PCT)
                open_trade[arm] = None

        # --- entries: same bar, two gates ---
        if open_trade["off"] is not None and open_trade["on"] is not None:
            continue
        window = enriched.iloc[max(0, i - 1): i + 1]
        atr = max(float(row["atr"]) if pd.notna(row["atr"]) else 0.0, close * ATR_FLOOR_PCT)
        for arm, engine in engines.items():
            if open_trade[arm] is not None:
                continue
            if engine.get_decision(window).signal == "LONG_REVERSION":
                open_trade[arm] = _Trade(
                    close, close - atr * STOP_ATR_MULTIPLE, close + atr * TARGET_ATR_MULTIPLE)
    return pnls


def _metrics(pnls) -> tuple:
    df = pd.DataFrame({"pnl": pnls}) if pnls else pd.DataFrame({"pnl": [0.0, 0.0]})
    return len(pnls), float(df["pnl"].sum()), profit_factor(df), sharpe_ratio(df)


def main() -> None:
    parser = argparse.ArgumentParser(description="Trend-filter OFF vs ON A/B (research-only)")
    parser.add_argument("symbols", nargs="*", help="symbols (default: core universe)")
    parser.add_argument("--days", type=int, default=20, help="lookback in days (default 20)")
    parser.add_argument("--end", default=None,
                        help="window end date YYYY-MM-DD (default: today; use for OOS validation)")
    parser.add_argument("--timeframe", default="5Min")
    parser.add_argument("--band", type=float, default=0.035,
                        help="trend_filter_band_pct for the ON arm (default 0.035 = 3.5%%)")
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
    print(f"Trend-filter A/B (long-only) | symbols={len(symbols)} | "
          f"{start}->{end_s} | {args.timeframe} | band={args.band:.3f}")

    tot = {"off": [], "on": []}
    print(f"\n  {'SYMBOL':<8}{'off PF':>9}{'on PF':>9}{'off net':>11}{'on net':>11}"
          f"{'off/on n':>11}{'Δnet':>11}")
    for sym in symbols:
        try:
            bars = fetch_alpaca_bars(sym, start, end_s, args.timeframe)
            if bars is None or len(bars) < 60:
                print(f"  {sym:<8}  (insufficient data)")
                continue
            pnls = _simulate(bars, args.band)
            tot["off"] += pnls["off"]
            tot["on"] += pnls["on"]
            mo, mn = _metrics(pnls["off"]), _metrics(pnls["on"])
            print(f"  {sym:<8}{mo[2]:>9.2f}{mn[2]:>9.2f}{mo[1]:>11.4f}{mn[1]:>11.4f}"
                  f"{f'{mo[0]}/{mn[0]}':>11}{mn[1]-mo[1]:>11.4f}")
        except Exception as e:
            print(f"  {sym:<8}  error: {e}")

    print("\n=== Universe totals ===")
    for arm in ("off", "on"):
        n, net, pf, sr = _metrics(tot[arm])
        print(f"  filter {arm:<3} trades={n:<5} net={net:>9.4f}  PF={pf:.2f}  Sharpe={sr:.2f}")
    no, non = _metrics(tot["off"]), _metrics(tot["on"])
    print(f"\n  delta (on - off): net={non[1]-no[1]:+.4f}  "
          f"trades={non[0]-no[0]:+d}  (fewer trades + higher net/PF => filter earns its keep)")
    print("  research-only; live default already has the filter ON.")


if __name__ == "__main__":
    main()
