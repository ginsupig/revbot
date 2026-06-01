"""Pure, network-free helpers for cost- and churn-aware trading.

Two ideas, both aimed at the one lever that actually moves *net* PnL on
wide-spread leveraged ETFs — trading less, and only when the expected move
clears costs:

1. Re-entry cooldown — after an exit, sit out N bars before re-entering the
   same symbol. The live bot already does this (``symbol_cooldown_minutes``);
   the backtester did not, so it over-counted trades and under-counted cost.

2. Reward-to-cost gate — skip an entry whose bracket target doesn't clear the
   round-trip cost by a minimum multiple. Removes negative-expectancy churn.

Kept free of pandas / engine imports so they unit-test in isolation.
"""
from __future__ import annotations


def reward_pct(entry_price: float, target_price: float) -> float:
    """Fractional reward from entry to target, e.g. 0.012 == +1.2%."""
    if entry_price <= 0:
        return 0.0
    return target_price / entry_price - 1.0


def passes_cost_gate(
    entry_price: float,
    target_price: float,
    cost_pct: float,
    min_reward_cost_ratio: float,
) -> bool:
    """True if the trade's reward clears cost by the required multiple.

    ``min_reward_cost_ratio <= 0`` disables the gate (always True). Otherwise
    the bracket reward must be at least ``cost_pct * min_reward_cost_ratio``;
    e.g. ratio=3 with an 8 bps cost requires a target at least +0.24% away.
    """
    if min_reward_cost_ratio <= 0:
        return True
    return reward_pct(entry_price, target_price) >= cost_pct * min_reward_cost_ratio


def in_cooldown(bar_idx: int, last_exit_idx: int | None, cooldown_bars: int) -> bool:
    """True if ``bar_idx`` is still inside the post-exit cooldown window.

    ``cooldown_bars <= 0`` disables the cooldown (always False). With a window
    of N, entries are blocked on the N bars immediately after the exit bar, so
    the earliest re-entry is ``last_exit_idx + cooldown_bars + 1``.
    """
    if cooldown_bars <= 0 or last_exit_idx is None:
        return False
    return (bar_idx - last_exit_idx) <= cooldown_bars


def minutes_to_bars(minutes: int, timeframe_minutes: int) -> int:
    """Convert a cooldown expressed in minutes to bars for a given timeframe.

    Mirrors the live ``symbol_cooldown_minutes`` so a backtest can match live:
    30 minutes on 5-minute bars -> 6 bars. Rounds down; never negative.
    """
    if timeframe_minutes <= 0:
        return 0
    return max(0, minutes // timeframe_minutes)
