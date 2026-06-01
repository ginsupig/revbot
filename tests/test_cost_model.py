from reversion_bot.cost_model import (
    reward_pct,
    passes_cost_gate,
    in_cooldown,
    minutes_to_bars,
)


# --- reward_pct -------------------------------------------------------------

def test_reward_pct_basic():
    assert reward_pct(100.0, 102.0) == round(0.02, 9) or abs(reward_pct(100.0, 102.0) - 0.02) < 1e-9


def test_reward_pct_guards_zero_entry():
    assert reward_pct(0.0, 5.0) == 0.0


# --- passes_cost_gate -------------------------------------------------------

def test_gate_disabled_when_ratio_zero():
    # ratio <= 0 -> always allowed, even for a tiny reward
    assert passes_cost_gate(100.0, 100.01, cost_pct=0.0008, min_reward_cost_ratio=0.0)


def test_gate_blocks_reward_below_cost_multiple():
    # cost 8 bps, ratio 3 -> need >= 0.24% reward; 0.20% fails
    assert not passes_cost_gate(100.0, 100.20, cost_pct=0.0008, min_reward_cost_ratio=3.0)


def test_gate_allows_reward_above_cost_multiple():
    # 0.30% reward clears the 0.24% bar
    assert passes_cost_gate(100.0, 100.30, cost_pct=0.0008, min_reward_cost_ratio=3.0)


# --- in_cooldown ------------------------------------------------------------

def test_cooldown_disabled_when_zero():
    assert not in_cooldown(bar_idx=5, last_exit_idx=4, cooldown_bars=0)


def test_cooldown_none_last_exit():
    assert not in_cooldown(bar_idx=5, last_exit_idx=None, cooldown_bars=6)


def test_cooldown_blocks_within_window():
    # exit at 10, cooldown 6 -> bars 11..16 blocked, 17 allowed
    assert in_cooldown(11, 10, 6)
    assert in_cooldown(16, 10, 6)
    assert not in_cooldown(17, 10, 6)


# --- minutes_to_bars --------------------------------------------------------

def test_minutes_to_bars_live_parity():
    assert minutes_to_bars(30, 5) == 6     # 30-min live cooldown on 5-min bars


def test_minutes_to_bars_rounds_down_and_guards():
    assert minutes_to_bars(29, 5) == 5
    assert minutes_to_bars(30, 0) == 0
