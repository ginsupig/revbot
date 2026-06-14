"""A/B backtest: loss-aware re-entry brake OFF vs ON (the second gate from
"Gate single-name downtrends", PR #65).

The trend filter (measured by trend_filter_ab_backtest.py) refuses dip-buys
extended too far below trend. The *brake* is the complementary lever and the
one actually aimed at the WDC failure mode: after a LOSING long exit it blocks
re-entry in that name for ``loss_reentry_cooldown_minutes`` — so the bot can't
re-buy a name that just stopped it out and is still falling.

This replays the REAL ReversionEngine long decision per bar through one
identical ATR-bracket sim, and runs two arms that differ ONLY by the brake:
  off   : re-enter as soon as the engine signals again
  brake : after a losing exit, suppress entries for the cooldown window
Both arms share the same enriched bars and the same trend-filter config (the
live default), so the delta is the brake's marginal effect on top of it.

Research-only: imports the live engine, changes nothing in the live path.

Usage:
    python reentry_brake_ab_backtest.py                       # core universe, 20d
    python reentry_brake_ab_backtest.py WDC SMCI AMD --days 30
    python reentry_brake_ab_backtest.py --cooldown-min 120    # longer brake
    python reentry_brake_ab_backtest.py --end 2026-05-24 --days 20   # OOS window
"""
from __future__ import annotations

import argparse
import re
from datetime import datetime, timedelta

import pandas as pd

from reversion_bot.config import ReversionConfig
from reversion_bot.engine import ReversionEngine
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
MAX_HOLD_BARS = 78          # ~1 session of 5-min bars; constant across both arms


class _Trade:
    __slots__ = ("entry", "stop", "target", "bars_held")

    def __init__(self, entry, stop, target):
        self.entry, self.stop, self.target, self.bars_held = entry, stop, target, 0


def _bar_minutes(timeframe: str) -> int:
    """Minutes per bar from a timeframe string ('5Min'->5, '1Min'->1, 'Hour'->60)."""
    key = str(timeframe).strip().lower()
    if key in ("hour", "1hour"):
        return 60
    if key in ("day", "1day", "daily"):
        return 390
    m = re.match(r"(\d+)\s*min", key)
    if m:
        return int(m.group(1))
    if key in ("minute", "min"):
        return 1
    return 5


def _simulate(bars, cooldown_bars, band_pct) -> dict:
    """Return {'off': [pnl...], 'brake': [pnl...]} for one symbol.

    Both arms use the live trend-filter config; the brake arm additionally
    suppresses a new entry while within ``cooldown_bars`` of a *losing* exit.
    A 2-row window is fed to get_decision (it reads only the last two bars).
    """
    engine = ReversionEngine(ReversionConfig(
        min_history=60, enable_shorts=False,
        use_trend_filter=True, trend_filter_band_pct=band_pct))
    enriched = engine.calculate_indicators(bars)

    pnls = {"off": [], "brake": []}
    open_trade = {"off": None, "brake": None}
    last_loss_idx = {"off": None, "brake": None}

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
                ret = (exit_px / tr.entry - 1.0) - COST_PCT
                pnls[arm].append(ret)
                if ret < 0:
                    last_loss_idx[arm] = i              # arm the brake on a loss
                open_trade[arm] = None

        # --- entries: same engine decision, brake gates re-entry after a loss ---
        if open_trade["off"] is not None and open_trade["brake"] is not None:
            continue
        window = enriched.iloc[max(0, i - 1): i + 1]
        signal_long = engine.get_decision(window).signal == "LONG_REVERSION"
        if not signal_long:
            continue
        atr = max(float(row["atr"]) if pd.notna(row["atr"]) else 0.0, close * ATR_FLOOR_PCT)
        for arm in ("off", "brake"):
            if open_trade[arm] is not None:
                continue
            if arm == "brake" and last_loss_idx["brake"] is not None \
                    and (i - last_loss_idx["brake"]) < cooldown_bars:
                continue                                # still inside the loss cooldown
            open_trade[arm] = _Trade(
                close, close - atr * STOP_ATR_MULTIPLE, close + atr * TARGET_ATR_MULTIPLE)
    return pnls


def _metrics(pnls) -> tuple:
    df = pd.DataFrame({"pnl": pnls}) if pnls else pd.DataFrame({"pnl": [0.0, 0.0]})
    return len(pnls), float(df["pnl"].sum()), profit_factor(df), sharpe_ratio(df)


def main() -> None:
    parser = argparse.ArgumentParser(description="Loss-aware re-entry brake OFF vs ON A/B (research-only)")
    parser.add_argument("symbols", nargs="*", help="symbols (default: core universe)")
    parser.add_argument("--days", type=int, default=20, help="lookback in days (default 20)")
    parser.add_argument("--end", default=None, help="window end date YYYY-MM-DD (default: today; use for OOS)")
    parser.add_argument("--timeframe", default="5Min")
    parser.add_argument("--cooldown-min", type=int, default=90,
                        help="loss_reentry_cooldown_minutes for the brake arm (default 90)")
    parser.add_argument("--band", type=float, default=0.035,
                        help="trend_filter_band_pct applied to BOTH arms (default 0.035)")
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
    bar_min = _bar_minutes(args.timeframe)
    cooldown_bars = max(1, round(args.cooldown_min / bar_min))
    print(f"Re-entry-brake A/B (long-only) | symbols={len(symbols)} | {start}->{end_s} | "
          f"{args.timeframe} | brake={args.cooldown_min}min ({cooldown_bars} bars) | band={args.band:.3f}")

    tot = {"off": [], "brake": []}
    print(f"\n  {'SYMBOL':<8}{'off PF':>9}{'brk PF':>9}{'off net':>11}{'brk net':>11}"
          f"{'off/brk n':>11}{'Δnet':>11}")
    for sym in symbols:
        try:
            bars = fetch_alpaca_bars(sym, start, end_s, args.timeframe)
            if bars is None or len(bars) < 60:
                print(f"  {sym:<8}  (insufficient data)")
                continue
            pnls = _simulate(bars, cooldown_bars, args.band)
            tot["off"] += pnls["off"]
            tot["brake"] += pnls["brake"]
            mo, mb = _metrics(pnls["off"]), _metrics(pnls["brake"])
            print(f"  {sym:<8}{mo[2]:>9.2f}{mb[2]:>9.2f}{mo[1]:>11.4f}{mb[1]:>11.4f}"
                  f"{f'{mo[0]}/{mb[0]}':>11}{mb[1]-mo[1]:>11.4f}")
        except Exception as e:
            print(f"  {sym:<8}  error: {e}")

    print("\n=== Universe totals ===")
    for arm in ("off", "brake"):
        n, net, pf, sr = _metrics(tot[arm])
        print(f"  brake {arm:<5} trades={n:<5} net={net:>9.4f}  PF={pf:.2f}  Sharpe={sr:.2f}")
    no, nb = _metrics(tot["off"]), _metrics(tot["brake"])
    print(f"\n  delta (brake - off): net={nb[1]-no[1]:+.4f}  trades={nb[0]-no[0]:+d}  "
          f"(the brake only removes re-entries after a loss; win => those re-buys were net losers)")
    print("  research-only; live default already has the brake ON (90 min).")


if __name__ == "__main__":
    main()
