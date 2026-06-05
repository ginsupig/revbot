from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd


def is_risk_off(bars: pd.DataFrame | None, ema_length: int = 50) -> bool:
    """True when the benchmark's latest close sits below its trend EMA.

    This is the bot's market-regime gate: when a broad benchmark (e.g. SPY) is
    trending down, dip-buying every oversold name is the losing trade, so the
    caller suppresses new longs.

    Fail-open by design: returns False (risk-on) on missing/short/NaN data so a
    benchmark fetch hiccup never halts all trading.
    """
    if bars is None or len(bars) < ema_length or "close" not in bars.columns:
        return False
    close = bars["close"].astype(float)
    ema = close.ewm(span=ema_length, adjust=False).mean()
    last_close = float(close.iloc[-1])
    last_ema = float(ema.iloc[-1])
    # NaN guard (NaN != NaN).
    if last_close != last_close or last_ema != last_ema:
        return False
    return last_close < last_ema


def suppress_longs_if_risk_off(
    candidates: List[Dict], risk_off: bool
) -> Tuple[List[Dict], List[Dict]]:
    """When risk-off, drop long candidates; shorts pass through untouched.

    Returns (kept, dropped_longs). A candidate is a long if it carries go_long;
    go_short candidates (and anything else) are always kept.
    """
    if not risk_off:
        return candidates, []
    kept = [c for c in candidates if not c.get("go_long")]
    dropped = [c for c in candidates if c.get("go_long")]
    return kept, dropped
