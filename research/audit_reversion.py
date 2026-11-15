"""Offline, reproducible single-symbol signal diagnostic; never places orders.

Uses the production engine and RiskManager, prior completed-bar decisions,
next-open limit-constrained fills, median targets, causal trailing stops,
per-symbol occupancy and loss cooldown. This is NOT full live-service parity:
no historical quotes, broker queues, adaptive routing, portfolio governor, or
SPY regime gate. Session-end exits assume execution at the last recorded close.
Results cannot authorize deployment; they can falsify weak signal hypotheses.

python -m research.audit_reversion state/audit_20260928/AAPL_5Min.csv
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from reversion_bot.config import ReversionConfig, RiskConfig
from reversion_bot.engine import ReversionEngine
from reversion_bot.risk import RiskManager

VARIANTS = {
    "baseline": {},
    "bullish_close": {"require_bullish_close": True},
    "band_reclaim": {"require_reclaim_lb1": True},
    "rsi_30": {"rsi_max": 30.0},
    "trend_1pct": {"trend_filter_band_pct": 0.01},
}
WINDOWS = {"research": ("2025-01-01", "2026-01-01"),
           "validation": ("2026-01-01", "2026-07-01"),
           "holdout": ("2026-07-01", "2026-09-26")}


class _PlanView:
    def __init__(self, plan):
        self.entry_price = float(plan["entry_price"])
        self.stop_price = float(plan["stop_price"])
        self.target_price = float(plan["target_price"])
        self.qty = int(plan["qty"])


def prepare(bars, overrides):
    bars = bars.copy()
    bars["date"] = pd.to_datetime(bars["date"], utc=True)
    if bars["date"].duplicated().any() or not bars["date"].is_monotonic_increasing:
        raise ValueError("bars must have unique increasing timestamps")
    values = bars[["open", "high", "low", "close", "volume"]].to_numpy(float)
    if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4] < 0).any():
        raise ValueError("invalid OHLCV data")
    if ((bars.high < bars[["open", "close", "low"]].max(axis=1)) |
        (bars.low > bars[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("inconsistent OHLC geometry")
    local = bars.date.dt.tz_convert("America/New_York")
    minute = local.dt.hour * 60 + local.dt.minute
    bars = bars.loc[(minute >= 570) & (minute < 960)].reset_index(drop=True)
    engine = ReversionEngine(replace(ReversionConfig(), **overrides))
    enriched = engine.calculate_indicators(bars)
    decisions = {}
    for i in range(engine.config.min_history - 1, len(enriched) - 1):
        d = engine.get_decision(enriched.iloc[i-1:i+1])
        if d.signal == "LONG_REVERSION":
            decisions[i] = d
    return enriched, decisions


def simulate(bars, decisions, start, end, cost_bps=8, *, initial_equity=100000., details=False,
             plans=None, trail_atr_multiple=1.5, limit_entry_offset_bps=8.0, risk_config=None):
    """Return diagnostic account metrics with fixed 0.5% risk / 15% value caps.

    ``plans`` ({signal bar: PositionPlan-like dict}) replaces the internally
    built plan with the one the live service produced for that bar (used by
    research/regime_gate_ab.py); its qty is taken as is. ``risk_config``,
    ``trail_atr_multiple`` and ``limit_entry_offset_bps`` default to the
    original diagnostic's fixed values.
    """
    if not np.isfinite(initial_equity) or initial_equity <= 0:
        raise ValueError("initial_equity must be finite and positive")
    rm = RiskManager(risk_config or RiskConfig(max_position_value_pct=.15, min_rr=1.5))
    times = pd.DatetimeIndex(bars.date)
    local = times.tz_convert("America/New_York")
    day = local.date
    minute = local.hour * 60 + local.minute
    active = np.flatnonzero((times >= pd.Timestamp(start, tz="UTC")) &
                            (times < pd.Timestamp(end, tz="UTC")))
    if len(active) == 0:
        return {"trades": 0, "return_pct": 0., "max_drawdown_pct": 0., "pf": 0.}
    op, hi, lo, cl, sma = (bars[c].to_numpy(float) for c in ["open", "high", "low", "close", "sma"])
    equity = float(initial_equity)
    curve = [equity]
    pnl, trade_rets, trades = [], [], []
    pos = None
    next_entry = times[active[0]]
    skipped_limit = 0
    for i in active:
        if pos is None and i > 0 and i-1 in decisions and times[i] >= next_entry:
            # Marketable-limit order based on the completed signal bar. Gaps
            # above the limit are skipped; no optimistic later-in-bar fill.
            if day[i] == day[i-1] and times[i] - times[i-1] == pd.Timedelta(minutes=5) and 600 <= minute[i] < 940:
                d = decisions[i-1]
                if plans is not None:
                    plan = _PlanView(plans[i-1]) if i-1 in plans else None
                else:
                    try:
                        plan = rm.build_long_plan(equity, d)
                    except ValueError:
                        plan = None
                if plan is not None:
                    limit = round(plan.entry_price * (1.0 + limit_entry_offset_bps / 10000), 2)
                    entry = op[i]
                    if entry > limit:
                        skipped_limit += 1
                    elif plan.stop_price < entry < plan.target_price:
                        if plans is not None:
                            qty = int(plan.qty)
                        else:
                            qty = min(plan.qty, int(equity * .005 / (entry - plan.stop_price)),
                                      int(equity * .15 / entry))
                        if qty > 0:
                            pos = dict(entry=entry, qty=qty, stop=plan.stop_price,
                                       target=plan.target_price, peak=entry,
                                       trail=max(float(d.atr), float(d.close)*.0035)*trail_atr_multiple,
                                       median_exit=False, signal_bar=i-1, entry_bar=i)
        if pos is not None:
            exit_price = None
            if op[i] <= pos["stop"]:
                exit_price = op[i]
            elif pos["median_exit"] or minute[i] >= 950:
                exit_price = op[i]
            elif op[i] >= pos["target"]:
                exit_price = pos["target"]
            elif lo[i] <= pos["stop"]:
                exit_price = pos["stop"]
            elif hi[i] >= pos["target"]:
                exit_price = pos["target"]
            elif i == active[-1] or i == len(bars)-1 or day[i+1] != day[i]:
                exit_price = cl[i]
            if exit_price is not None:
                net = (exit_price - pos["entry"]) * pos["qty"] - pos["entry"] * pos["qty"] * cost_bps / 10000
                equity += net
                pnl.append(net)
                trade_rets.append(net / (pos["entry"] * pos["qty"]))
                trades.append(dict(signal_bar=pos["signal_bar"], entry_bar=pos["entry_bar"], exit_bar=int(i),
                                   entry=pos["entry"], exit=float(exit_price), qty=pos["qty"], pnl=net,
                                   ret_bps=net / (pos["entry"] * pos["qty"]) * 10000))
                next_entry = times[i] + pd.Timedelta(minutes=90 if net < 0 else 30)
                pos = None
            else:
                # The close is observable at this point; trail only affects
                # the next bar. It cannot use this bar's high before its low.
                pos["peak"] = max(pos["peak"], cl[i])
                pos["stop"] = max(pos["stop"], pos["peak"] - pos["trail"])
                pos["median_exit"] = bool(cl[i] >= sma[i])
        mark = equity if pos is None else equity + pos["qty"] * (cl[i] - pos["entry"]) - pos["entry"] * pos["qty"] * cost_bps / 10000
        curve.append(mark)
    eq = np.asarray(curve)
    arr = np.asarray(pnl)
    wins, losses = arr[arr>0].sum(), -arr[arr<0].sum()
    result = dict(trades=len(pnl), return_pct=(equity/initial_equity-1)*100,
                max_drawdown_pct=float((eq/np.maximum.accumulate(eq)-1).min()*100),
                pf=float(wins/losses) if losses else None,
                mean_trade_bps=float(np.mean(trade_rets)*10000) if pnl else 0.,
                skipped_limit=skipped_limit)
    if details:
        result.update(final_equity=equity, equity_curve=curve, trade_pnl=pnl, trades=trades)
    return result


def run(path, output):
    bars = pd.read_csv(path)
    result = dict(data_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                  rows=len(bars), windows=WINDOWS, base_cost_bps=8, candidates={},
                  limitations=__doc__.split("Results cannot")[0], promoted=False)
    prepared = {}
    for name, override in VARIANTS.items():
        enriched, decisions = prepare(bars, override)
        prepared[name] = (enriched, decisions)
        result["candidates"][name] = {
            w: {str(c): simulate(enriched, decisions, *WINDOWS[w], c) for c in (8, 16, 24)}
            for w in ("research", "validation")}
        print(name, json.dumps(result["candidates"][name]), flush=True)
    # Freeze selection BEFORE computing holdout. Baseline retained as a control.
    eligible = [n for n, r in result["candidates"].items()
                if all(r[w]["16"]["trades"] >= 30 and r[w]["16"]["return_pct"] > 0
                       for w in ("research", "validation"))]
    chosen = max(eligible, key=lambda n: result["candidates"][n]["validation"]["16"]["return_pct"]) if eligible else None
    result["selected"] = chosen
    for name in dict.fromkeys(["baseline"] + ([chosen] if chosen else [])):
        result["candidates"][name]["holdout"] = {
            str(c): simulate(*prepared[name], *WINDOWS["holdout"], c) for c in (8, 16, 24)}
    result["verdict"] = "RESEARCH ONLY: no full execution parity or independent deployment proof"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2, allow_nan=False))
    print("selected:", chosen, "report:", output, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv")
    p.add_argument("--output", default="state/audit_20260928/signal_diagnostic.json")
    a = p.parse_args()
    run(a.csv, a.output)
