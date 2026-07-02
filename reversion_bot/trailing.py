"""Trailing-stop math for long positions — pure, broker-free, unit-testable.

Mirrors the backtest that validated the edge (execution_tuning_backtest.py):
the stop ratchets UP to ``high_water - trail_dist`` and never down, floored by the
initial stop. The trail distance is derived from the entry's stop distance so no
separate ATR has to be stored:

    entry - init_stop = stop_atr_mult * ATR   (the bracket's stop leg)
    =>  ATR        = (entry - init_stop) / stop_atr_mult
    =>  trail_dist = trail_atr_mult * ATR
                   = (trail_atr_mult / stop_atr_mult) * (entry - init_stop)
"""
from __future__ import annotations


def trailing_stop_price(
    entry_price: float,
    stop_distance: float,
    high_water: float,
    stop_atr_mult: float,
    trail_atr_mult: float,
) -> float:
    """Desired trailing-stop price for a long.

    ``stop_distance`` is the entry's initial risk-per-share (entry - init_stop,
    i.e. stop_atr_mult * ATR). Returns ``max(init_stop, high_water - trail_dist)``
    — the ratcheted stop, never below the initial stop.
    """
    if stop_distance <= 0 or stop_atr_mult <= 0:
        return entry_price - max(stop_distance, 0.0)
    init_stop = entry_price - stop_distance
    trail_dist = (trail_atr_mult / stop_atr_mult) * stop_distance
    return max(init_stop, high_water - trail_dist)


def trailing_stop_price_short(
    entry_price: float,
    stop_distance: float,
    low_water: float,
    stop_atr_mult: float,
    trail_atr_mult: float,
) -> float:
    """Desired trailing-stop price for a short (mirror of long trail).

    ``stop_distance`` is the initial risk-per-share (stop - entry, i.e.
    stop_atr_mult * ATR). Returns ``min(init_stop, low_water + trail_dist)``
    — the ratcheted stop, never above the initial stop.
    """
    if stop_distance <= 0 or stop_atr_mult <= 0:
        return entry_price + max(stop_distance, 0.0)
    init_stop = entry_price + stop_distance
    trail_dist = (trail_atr_mult / stop_atr_mult) * stop_distance
    return min(init_stop, low_water + trail_dist)


def should_lower_stop(
    current_stop: float,
    desired_stop: float,
    last_price: float,
    min_increment: float = 0.01,
) -> bool:
    """True only when it's safe and worthwhile to move a SHORT's stop DOWN.

    Mirror of should_raise_stop for longs:
      - only lower (desired must be below current by >= min_increment);
      - never place the stop at/below the last price (a short buy-stop below
        market would trigger immediately).
    """
    if desired_stop <= last_price:
        return False
    return current_stop - desired_stop >= min_increment


def should_raise_stop(
    current_stop: float,
    desired_stop: float,
    last_price: float,
    min_increment: float = 0.01,
) -> bool:
    """True only when it's safe and worthwhile to move the stop UP.

    Guards:
      - only raise (desired must exceed current by >= ``min_increment``) — a
        trailing stop never moves down, and tiny moves aren't worth an API call;
      - never place the stop at/above the last price (a long sell-stop above
        market would trigger immediately).
    """
    if desired_stop >= last_price:
        return False
    return desired_stop - current_stop >= min_increment
