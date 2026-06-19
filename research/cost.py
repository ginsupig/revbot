"""Slippage/fees sweep: is the edge gross-only, eaten by realistic friction?

A thin gross edge is worthless if a few bps of round-trip cost erases it, and
daily-turnover strategies are especially cost-fragile. Sweep the cost and find
the breakeven.
"""
from __future__ import annotations

import numpy as np


def _pf(arr: np.ndarray) -> float:
    w, l = arr[arr > 0].sum(), -arr[arr < 0].sum()
    return float(w / l) if l > 0 else (10.0 if w > 0 else 0.0)


def apply_cost(returns, bps: float) -> np.ndarray:
    """Per-trade returns net of a ``bps`` round-trip cost."""
    return np.asarray(returns, dtype=float) - bps / 10000.0


def cost_sweep(returns, bps_list=(0, 2, 4, 6, 8, 10)) -> dict:
    """``{bps: {mean, pf, win}}`` net of each round-trip cost."""
    arr = np.asarray(returns, dtype=float)
    out = {}
    for b in bps_list:
        net = arr - b / 10000.0
        out[float(b)] = dict(mean=float(net.mean()) if net.size else 0.0,
                             pf=_pf(net) if net.size else 0.0,
                             win=float((net > 0).mean()) if net.size else 0.0)
    return out


def breakeven_cost_bps(returns) -> float:
    """Round-trip cost (bps) at which net expectancy hits zero = gross mean in bps.
    Positive = the edge can absorb that many bps before turning negative."""
    arr = np.asarray(returns, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(arr.mean() * 10000.0)
