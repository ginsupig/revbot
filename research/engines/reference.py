"""Reference flush-bounce sweep (pure numpy) — the canonical, testable engine.

Hypothesis (the MU/ASTS/WDC question): does a stock bounce after an N-bar flush
of >= flush_pct while oversold? Sweeps flush_pct x rsi_max, exits with the
validated ATR bracket + trailing stop, and returns per-config trade returns for
the orchestrator to gate. The VectorBT template mirrors this far faster.
"""
from __future__ import annotations

import itertools

import numpy as np

from reversion_bot.indicators import calculate_rsi, calculate_atr

ATR_FLOOR_PCT = 0.0035
COST_PCT = 0.0008
EXIT = dict(stop=1.20, target=3.50, trail=1.50, hold=78)
COOLDOWN, LOSS_COOLDOWN = 6, 18


def flush_bounce_signals(bars, flush_pct: float, drop_window: int, rsi_max: float):
    """Bool array: a >= ``flush_pct`` drop over ``drop_window`` bars while RSI<=rsi_max."""
    close = bars["close"].astype(float)
    rsi = calculate_rsi(bars)
    drop = close / close.shift(drop_window) - 1.0
    return ((drop <= -abs(flush_pct)) & (rsi <= rsi_max)).fillna(False).to_numpy()


def _trade_returns(high, low, close, atr_arr, signal_idx, exit):
    sig = set(signal_idx)
    n = len(close)
    out = []
    in_pos = False
    entry = init_stop = target = highest = 0.0
    trail_dist = None
    held = 0
    last_exit = -10 ** 9
    last_loss = False
    for i in range(n):
        if in_pos:
            held += 1
            highest = max(highest, high[i])
            cur_stop = init_stop if trail_dist is None else max(init_stop, highest - trail_dist)
            r = None
            if low[i] <= cur_stop:
                r = cur_stop / entry - 1.0
            elif high[i] >= target:
                r = target / entry - 1.0
            elif held >= exit["hold"]:
                r = close[i] / entry - 1.0
            if r is not None:
                out.append(r - COST_PCT)
                in_pos = False
                last_exit = i
                last_loss = (r - COST_PCT) < 0
        if not in_pos and i in sig:
            cd = LOSS_COOLDOWN if last_loss else COOLDOWN
            if i - last_exit > cd:
                entry = close[i]
                a = max(float(atr_arr[i]) if not np.isnan(atr_arr[i]) else 0.0, entry * ATR_FLOOR_PCT)
                init_stop = entry - exit["stop"] * a
                target = entry + exit["target"] * a
                trail_dist = exit["trail"] * a if exit.get("trail") else None
                highest = high[i]
                in_pos = True
                held = 0
    return out


def sweep(bars, flush_grid, rsi_grid, drop_window: int = 3, exit=EXIT) -> dict:
    """``{config_key: per_trade_returns}`` over the flush_pct x rsi_max grid."""
    high = bars["high"].astype(float).to_numpy()
    low = bars["low"].astype(float).to_numpy()
    close = bars["close"].astype(float).to_numpy()
    atr_arr = calculate_atr(bars).to_numpy()
    out = {}
    for fp, rm in itertools.product(flush_grid, rsi_grid):
        idx = np.flatnonzero(flush_bounce_signals(bars, fp, drop_window, rm))
        out[f"flush{fp:g}_rsi{rm:g}"] = np.asarray(
            _trade_returns(high, low, close, atr_arr, idx, exit), dtype=float)
    return out
