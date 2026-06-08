"""Bearish exhaustion signal: higher highs on fading relative volume.

A market making fresh highs on *declining* participation is a classic
exhaustion / distribution tell — the push is running out of fuel. This is the
mirror concern to the bot's oversold dip-buy: it argues against chasing longs
and for fading the failing push (short).

Everything here is a pure pandas transform (no engine/network) so it unit-tests
in isolation and drops straight into the walkforward backtest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def relative_volume(volume: pd.Series, lookback: int = 20) -> pd.Series:
    """RVOL = volume / its own rolling average. 1.0 == average participation."""
    volume = volume.astype(float)
    avg = volume.rolling(lookback, min_periods=lookback).mean()
    return volume / avg.replace(0.0, np.nan)


def chaikin_money_flow(df: pd.DataFrame, period: int = 21) -> pd.Series:
    """Chaikin Money Flow: volume weighted by where each bar closes in its range.

    A price/volume *proxy* for buying vs selling pressure (not true order flow —
    it never sees the tape). The Money Flow Multiplier is +1 when the bar closes
    at its high (accumulation) and -1 at its low (distribution); CMF is the
    volume-weighted average of that over ``period``, normalized to [-1, +1].
    Negative CMF on a new high = price up but money flowing out = distribution.

    Known limitation: the multiplier is close-within-bar only, so it ignores
    gaps between bars and can mislead on gappy intraday data.
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    rng = (high - low).replace(0.0, np.nan)
    mfm = ((close - low) - (high - close)) / rng
    mfv = mfm * volume
    return (
        mfv.rolling(period, min_periods=period).sum()
        / volume.rolling(period, min_periods=period).sum().replace(0.0, np.nan)
    )


def exhaustion_short_signal(
    df: pd.DataFrame,
    hh_lookback: int = 20,
    rvol_lookback: int = 20,
    rvol_max: float = 1.0,
    require_divergence: bool = True,
    cmf_max: float | None = None,
    cmf_period: int = 21,
) -> pd.Series:
    """Boolean Series: True where a bearish-exhaustion short setup prints.

    Conditions:
      - higher high: the bar makes a new ``hh_lookback``-bar high (price still
        pushing up), and
      - low RVOL: relative volume is below ``rvol_max`` (below-average
        participation on the new high), and
      - (optional) divergence: RVOL is also below the RVOL recorded at the
        *previous* new high — i.e. price higher, volume lower, the textbook
        price/volume divergence, and
      - (optional) distribution: Chaikin Money Flow at or below ``cmf_max``
        (default None = off; 0.0 means money is net flowing OUT on the high).

    Returns a bool Series aligned to ``df.index`` (False where indicators aren't
    warm yet), so it can be used as an entry mask or merged as a feature.
    """
    high = df["high"].astype(float)
    volume = df["volume"].astype(float)
    rvol = relative_volume(volume, rvol_lookback)

    # New local high: current high is the max over the inclusive window.
    roll_max = high.rolling(hh_lookback, min_periods=hh_lookback).max()
    is_new_high = high >= roll_max

    low_rvol = rvol < rvol_max
    signal = is_new_high & low_rvol

    if require_divergence:
        # RVOL at the most recent prior new-high bar.
        prev_high_rvol = rvol.where(is_new_high).ffill().shift(1)
        signal = signal & (rvol < prev_high_rvol)

    if cmf_max is not None:
        # Distribution confirmation: money flowing out while price makes highs.
        cmf = chaikin_money_flow(df, cmf_period)
        signal = signal & (cmf <= cmf_max)

    return signal.fillna(False).astype(bool)
