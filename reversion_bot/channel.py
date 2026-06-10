"""Linear-regression channel — the structural exit the backtest validated.

The bot's fixed ATR target clips winners: an A/B over the universe (entry held
constant, only the exit changed) showed that exiting when price reaches the
UPPER regression channel ("sell the breakout") lifted the signal-layer median
profit factor 0.85 -> 1.00 and rescued exactly the high-beta names the fixed
target was capping (AMD 0.93->1.36, ARM 1.06->1.71, MU 0.96->1.33, CLF
0.84->1.38). A naive ATR trailing stop did the opposite (0.52) — whipsawed out
on noise — so the exit must be STRUCTURAL (the channel), not a tight trail.

These are pure helpers: main.py fetches the bars and calls close_position; the
broker-side stop + (wide) bracket target remain the safety net underneath.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def regression_channel_position(
    closes: Sequence[float], lookback: int = 80, k: float = 2.0
) -> Optional[float]:
    """Where the latest close sits within its linear-regression channel.

    Fit a least-squares line over the last ``lookback`` closes; band it by ``k``
    residual standard deviations. Returns 0.0 at the lower band, 1.0 at the
    upper band (clamped), so >= ~0.8 means "near the breakout / top of channel".

    Fail-safe: returns None on insufficient (< lookback) or degenerate (flat,
    non-finite) data, so the caller simply doesn't act rather than mis-exit.
    """
    if lookback < 2 or k <= 0:
        return None
    arr = np.asarray(list(closes), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < lookback:
        return None
    w = arr[-lookback:]
    x = np.arange(lookback, dtype=float)
    slope, intercept = np.polyfit(x, w, 1)
    line_at_last = intercept + slope * (lookback - 1)
    resid_sd = (w - (intercept + slope * x)).std()
    # A flat/near-flat series has no meaningful channel; its residual spread is
    # just float dust, which would otherwise divide into a spurious extreme.
    # Guard scale-relative to price so it holds across $10 and $1000 names.
    if not np.isfinite(resid_sd) or resid_sd <= abs(line_at_last) * 1e-9:
        return None
    pos = 0.5 + (w[-1] - line_at_last) / (2.0 * k * resid_sd)
    return float(min(max(pos, 0.0), 1.0))


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def select_breakout_exits(
    positions,
    closes_by_symbol: Dict[str, Sequence[float]],
    threshold: float = 0.80,
    lookback: int = 80,
    k: float = 2.0,
) -> List[str]:
    """Symbols of LONG positions whose latest close has reached the upper channel.

    ``closes_by_symbol`` maps symbol -> recent closes. Longs only: the breakout
    exit is an upside take-profit, so shorts (qty < 0) are left to their bracket.
    """
    out: List[str] = []
    for p in positions:
        if _safe_float(getattr(p, "qty", 0)) <= 0:
            continue
        symbol = str(getattr(p, "symbol", "")).upper()
        closes = closes_by_symbol.get(symbol)
        if closes is None:
            continue
        pos = regression_channel_position(closes, lookback, k)
        if pos is not None and pos >= threshold:
            out.append(symbol)
    return out
