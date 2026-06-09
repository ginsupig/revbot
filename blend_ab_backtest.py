"""A/B backtest: blended entry gate vs routed-component gate (review #4b).

The live service gates entries on the *blended* weighted score. The concern is
that for a routed mean-reversion setup, the anti-correlated trend/ml components
mush the decision. This script replays the REAL service scorers per bar (no
duplicated logic) and, from the same component scores, evaluates BOTH gates —
"blended" (current) and "routed" (gate on the routed style's own component) —
through an identical ATR-bracket sim, then reports which gate trades better.

Research-only: it imports the live scoring + the pure gate helper and changes
nothing in the live path. If "routed" wins out of sample, a follow-up wires a
gate_mode toggle into the service.

It drives the full per-bar service evaluation (incl. rolling ML), so it's
O(n^2)-ish — keep the window small (default 5 days of 5-min bars).

Usage:
    python blend_ab_backtest.py                 # core universe, 5d, 5Min
    python blend_ab_backtest.py NVDA SMCI --days 5
"""
from __future__ import annotations

import argparse
import os
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from reversion_bot.config import ReversionConfig, RiskConfig, PerformanceConfig
from reversion_bot.service import ReversionService
from reversion_bot.gating import gate_decision
from reversion_bot.trendfail import trend_fail_continue_strategy
from reversion_bot.analytics import profit_factor, sharpe_ratio
from run_real_backtest import fetch_alpaca_bars

CORE_UNIVERSE = [
    "ASTS", "MU", "AMD", "NVDA", "SMCI", "META", "AAPL", "APP", "ARM", "CRWD",
    "MSFT", "LRCX", "JPM", "FCX", "OXY", "LLY", "CLF", "HIMS", "TEM",
]

STOP_ATR_MULTIPLE = 1.20
TARGET_ATR_MULTIPLE = 2.00
ATR_FLOOR_PCT = 0.0035
COST_PCT = 0.0008
MAX_HOLD_BARS = 78          # ~1 session of 5-min bars; constant across both modes


class _Trade:
    __slots__ = ("side", "entry", "stop", "target", "bars_held")

    def __init__(self, side, entry, stop, target):
        self.side, self.entry, self.stop, self.target, self.bars_held = side, entry, stop, target, 0


def _trendfail_scores(bars, cfg) -> np.ndarray:
    """Per-bar trendfail score, computed once (vectorized) to avoid O(n^2)."""
    try:
        out = trend_fail_continue_strategy(bars, window=cfg.trendfail_window,
                                           threshold=cfg.trendfail_threshold)
        sig = out["signal"].to_numpy()
    except Exception:
        sig = np.zeros(len(bars))
    return np.where(sig == 1, 0.75, np.where(sig == -1, 0.10, 0.45))


def _simulate(bars, svc) -> dict:
    """Return {'blended': pnls, 'routed': pnls} for one symbol."""
    engine = svc.engine
    enriched = engine.calculate_indicators(bars)
    tf_scores = _trendfail_scores(bars, engine.config)
    threshold = svc.min_trade_score

    pnls = {"blended": [], "routed": []}
    open_trade = {"blended": None, "routed": None}

    for i in range(len(enriched)):
        row = enriched.iloc[i]
        close, high, low = float(row["close"]), float(row["high"]), float(row["low"])

        # --- exits (identical logic for both modes) ---
        for mode, tr in list(open_trade.items()):
            if tr is None:
                continue
            tr.bars_held += 1
            exit_px = None
            if tr.side == "long":
                if low <= tr.stop:
                    exit_px = tr.stop
                elif high >= tr.target:
                    exit_px = tr.target
            else:  # short
                if high >= tr.stop:
                    exit_px = tr.stop
                elif low <= tr.target:
                    exit_px = tr.target
            if exit_px is None and tr.bars_held >= MAX_HOLD_BARS:
                exit_px = close
            if exit_px is not None:
                ret = (exit_px / tr.entry - 1.0) if tr.side == "long" else (tr.entry / exit_px - 1.0)
                pnls[mode].append(ret - COST_PCT)
                open_trade[mode] = None

        # --- decision + component scores at bar i (shared by both modes) ---
        decision = engine.get_decision(enriched.iloc[: i + 1])
        if decision.signal == "WAIT" and decision.reason == "Indicators_Not_Ready":
            continue
        comps = {
            "mean_reversion": svc._score_mean_reversion(decision, enriched.iloc[: i + 1], engine),
            "ml": svc._get_ml_probability(bars.iloc[: i + 1]),
            "trendfail": float(tf_scores[i]),
            "trend_following": svc._get_trend_following_score(enriched.iloc[: i + 1]),
        }
        regime = svc._classify_regime(enriched.iloc[: i + 1], engine)
        entry_style, router = svc._route_style(comps, regime, decision.signal)
        weighted = svc._weighted_score(comps)
        atr = max(float(row["atr"]) if pd.notna(row["atr"]) else 0.0, close * ATR_FLOOR_PCT)

        for mode in ("blended", "routed"):
            if open_trade[mode] is not None:
                continue
            go_long, go_short = gate_decision(
                component_scores=comps, weighted_score=weighted, entry_style=entry_style,
                decision_signal=decision.signal, router_reason=router, threshold=threshold, mode=mode,
            )
            if go_long:
                open_trade[mode] = _Trade("long", close, close - atr * STOP_ATR_MULTIPLE,
                                          close + atr * TARGET_ATR_MULTIPLE)
            elif go_short:
                open_trade[mode] = _Trade("short", close, close + atr * STOP_ATR_MULTIPLE,
                                          close - atr * TARGET_ATR_MULTIPLE)
    return pnls


def _metrics(pnls) -> tuple:
    df = pd.DataFrame({"pnl": pnls}) if pnls else pd.DataFrame({"pnl": [0.0, 0.0]})
    return len(pnls), float(df["pnl"].sum()), profit_factor(df), sharpe_ratio(df)


def main() -> None:
    parser = argparse.ArgumentParser(description="Blended vs routed entry-gate A/B (research-only)")
    parser.add_argument("symbols", nargs="*", help="symbols (default: core universe)")
    parser.add_argument("--days", type=int, default=5, help="lookback (default 5; keep small — O(n^2))")
    parser.add_argument("--timeframe", default="5Min")
    args = parser.parse_args()
    load_dotenv()
    symbols = [s.strip().upper() for s in args.symbols] if args.symbols else CORE_UNIVERSE

    end = datetime.now()
    start = (end - timedelta(days=args.days)).strftime("%Y-%m-%d")
    end_s = end.strftime("%Y-%m-%d")
    print(f"Blend-gate A/B | symbols={len(symbols)} | {start}->{end_s} | {args.timeframe}")

    # One shared log path OUTSIDE any per-symbol tempdir: the ReversionService
    # logger keeps a FileHandler open, which would block TemporaryDirectory
    # cleanup on Windows (WinError 32). Perf state stays isolated per symbol.
    log_fd, log_path = tempfile.mkstemp(prefix="blend_ab_", suffix=".log")
    os.close(log_fd)

    tot = {"blended": [], "routed": []}
    print(f"\n  {'SYMBOL':<8}{'blended PF':>12}{'routed PF':>11}{'b net':>10}{'r net':>10}{'b/r trades':>12}")
    for sym in symbols:
        try:
            bars = fetch_alpaca_bars(sym, start, end_s, args.timeframe)
            if bars is None or len(bars) < 60:
                print(f"  {sym:<8}  (insufficient data)")
                continue
            with tempfile.TemporaryDirectory() as tmp:
                svc = ReversionService(ReversionConfig(min_history=60, enable_shorts=True),
                                       RiskConfig(), PerformanceConfig(state_dir=tmp),
                                       log_file=log_path)
                pnls = _simulate(bars, svc)
            tot["blended"] += pnls["blended"]
            tot["routed"] += pnls["routed"]
            bn, br = _metrics(pnls["blended"]), _metrics(pnls["routed"])
            print(f"  {sym:<8}{bn[2]:>12.2f}{br[2]:>11.2f}{bn[1]:>10.4f}{br[1]:>10.4f}"
                  f"{f'{bn[0]}/{br[0]}':>12}")
        except Exception as e:
            print(f"  {sym:<8}  error: {e}")

    try:
        os.unlink(log_path)
    except OSError:
        pass

    print("\n=== Universe totals ===")
    for mode in ("blended", "routed"):
        n, net, pf, sr = _metrics(tot[mode])
        print(f"  {mode:<8} trades={n:<5} net={net:>9.4f}  PF={pf:.2f}  Sharpe={sr:.2f}")
    print("\n  (research-only; live gate unchanged. If 'routed' wins OOS, wire the toggle.)")


if __name__ == "__main__":
    main()
