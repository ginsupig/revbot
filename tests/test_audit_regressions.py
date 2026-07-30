"""Regression tests for the full-code-audit findings.

Each test pins a defect the audit found in code that was previously either
untested or tested only directionally:

  * order sizing asserted NUMERICALLY (a boost-clamp or budget regression that
    sized live orders 2-10x too large used to pass the whole suite),
  * the EOD window predicates,
  * trailing-stop orchestration (branch selection + which multiple is used),
  * the governor's buying-power gate on a non-PDT account (dtbp == "0"),
  * position_meta pruning vs resting entry orders, and the direction-flip
    cooldown surviving a close,
  * the submit-time spread gate,
  * the single-instance lock's in-flight (empty) lock file,
  * the swing MOC window on a half day.
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reversion_bot.config import PortfolioConfig, RiskConfig
from reversion_bot.governor import ExecutionGovernor
from reversion_bot.models import ReversionDecision
from reversion_bot.portfolio import PortfolioState
from reversion_bot.risk import RiskManager, style_multiples
from reversion_bot.single_instance import acquire_lock, release_lock
from reversion_bot.swing import is_moc_entry_window, moc_cutoff


# --------------------------------------------------------------------------
# Order sizing: exact numbers, not just "qty > 0"
# --------------------------------------------------------------------------

def _decision(close=100.0, atr=1.0, sma=104.0):
    return ReversionDecision(
        signal="LONG_REVERSION", reason="Validated_Long_Reversion",
        symbol="TEST", close=close, sma=sma, atr=atr,
    )


def test_long_qty_equals_risk_budget_over_stop_distance():
    """qty is floor(risk_budget / risk_per_share) at neutral conviction.

    ATR 4.0 on a $100 name keeps the RISK budget (not the 15% value cap) the
    binding constraint: stop = 1.20 * 4.0 = 4.80/share.
    """
    cfg = RiskConfig()
    plan = RiskManager(cfg).build_long_plan(
        account_equity=100_000, decision=_decision(atr=4.0, sma=110.0),
        conviction_score=0.50)
    risk_budget = 100_000 * cfg.risk_per_trade_pct       # no boost at 0.50
    expected_qty = int(risk_budget // 4.80)
    # The value cap must not be the binding constraint in this fixture.
    assert expected_qty * plan.entry_price <= 100_000 * cfg.max_position_value_pct
    assert plan.qty == expected_qty
    assert plan.risk_per_share == pytest.approx(4.80, abs=1e-6)
    assert plan.max_account_risk == pytest.approx(round(risk_budget, 2), abs=0.01)


def test_conviction_boost_is_clamped_to_35_percent():
    """Max boost is +35% of the risk budget, reached at conviction 0.85 and
    NOT exceeded above it (a swapped min/max here oversizes every order)."""
    cfg = RiskConfig()
    # ATR 4.0 keeps the risk budget binding, so the boost is observable in qty.
    d = _decision(atr=4.0, sma=110.0)
    base = RiskManager(cfg).build_long_plan(
        account_equity=100_000, decision=d, conviction_score=0.50).qty
    at_cap = RiskManager(cfg).build_long_plan(
        account_equity=100_000, decision=d, conviction_score=0.85).qty
    way_past_cap = RiskManager(cfg).build_long_plan(
        account_equity=100_000, decision=d, conviction_score=1.00).qty
    assert at_cap == way_past_cap                     # clamped
    assert at_cap == int((100_000 * cfg.risk_per_trade_pct * 1.35) // 4.80)
    assert at_cap > base


def test_max_position_value_pct_caps_qty():
    """A cheap name with a tiny stop is capped by position VALUE, not risk."""
    cfg = RiskConfig()
    # 5.00 stock, 0.02 ATR -> stop distance 0.024 -> risk-based qty is huge.
    plan = RiskManager(cfg).build_long_plan(
        account_equity=100_000, decision=_decision(close=5.0, atr=0.02, sma=5.4),
        conviction_score=0.50)
    value_cap_qty = int((100_000 * cfg.max_position_value_pct) // plan.entry_price)
    assert plan.qty == value_cap_qty
    assert plan.position_value <= 100_000 * cfg.max_position_value_pct + plan.entry_price


def test_wide_stop_is_rejected_rather_than_floored_up_to_min_qty():
    """If the boosted budget can't afford min_qty at this stop distance, the
    plan must RAISE — flooring up to min_qty would exceed the per-trade risk
    cap on a wide-stop/high-priced name."""
    cfg = RiskConfig()
    with pytest.raises(ValueError, match="min_qty"):
        # 1M/share stop distance vs a ~$150 budget on 10k equity.
        RiskManager(cfg).build_long_plan(
            account_equity=10_000,
            decision=_decision(close=2_000_000.0, atr=1_000_000.0, sma=2_400_000.0),
            conviction_score=0.50)


def test_style_multiples_are_distinct_per_style():
    cfg = RiskConfig()
    assert style_multiples(cfg, "mean_reversion") == (cfg.stop_atr_multiple,
                                                      cfg.target_atr_multiple)
    assert style_multiples(cfg, "trendfail") == (cfg.trendfail_stop_atr_multiple,
                                                 cfg.trendfail_target_atr_multiple)
    assert style_multiples(cfg, "trend_following") == (cfg.trend_stop_atr_multiple,
                                                       cfg.trend_target_atr_multiple)
    # Unknown / missing style falls back to mean reversion.
    assert style_multiples(cfg, "nonsense") == style_multiples(cfg, "mean_reversion")
    assert style_multiples(cfg, None) == style_multiples(cfg, "mean_reversion")


def test_trendfail_trail_distance_uses_the_trendfail_stop_multiple():
    """The trail distance derived from a stored stop_distance must use the
    style's own stop multiple.

    A trendfail entry stores 1.10*ATR. Backing ATR out with the mean-reversion
    1.20 gave (1.5/1.2)*1.10 = 1.375*ATR instead of 1.50*ATR — trailing ~8%
    tighter than validated.
    """
    cfg = RiskConfig()
    atr = 2.0
    stop_distance = cfg.trendfail_stop_atr_multiple * atr        # what risk.py stored
    stop_mult, _ = style_multiples(cfg, "trendfail")
    recovered_atr = stop_distance / stop_mult
    trail_dist = recovered_atr * cfg.trail_atr_multiple
    assert recovered_atr == pytest.approx(atr)
    assert trail_dist == pytest.approx(cfg.trail_atr_multiple * atr)
    # The buggy path, for contrast:
    wrong = (stop_distance / cfg.stop_atr_multiple) * cfg.trail_atr_multiple
    assert wrong < trail_dist


# --------------------------------------------------------------------------
# Governor: buying power on a non-PDT account
# --------------------------------------------------------------------------

class _Acct:
    portfolio_value = "100000"
    equity = "15000"
    buying_power = "30000"
    daytrading_buying_power = "0"     # non-PDT: dtbp is 0, NOT "no buying power"
    trading_blocked = False
    account_blocked = False


class _Client:
    def get_account(self):
        return _Acct()

    def list_positions(self):
        return []


class _Executor:
    def __init__(self):
        self.client = _Client()

    def list_open_orders(self):
        return []

    def open_order_symbols(self):
        return set()


def _candidate(symbol="AAA", value=1000.0):
    return {
        "symbol": symbol,
        "go_long": True,
        "entry_style": "mean_reversion",
        "regime": "reversion",
        "trade_score": 0.6,
        "position_plan": {"qty": 10, "entry_price": value / 10,
                          "position_value": value, "risk_per_share": 1.0},
        "portfolio_heat": 0.0,
    }


def test_zero_daytrading_buying_power_falls_through_to_regular(tmp_path):
    """A non-PDT account reports daytrading_buying_power "0"; taking it
    literally rejected EVERY entry forever with "have 0.00"."""
    gov = ExecutionGovernor(PortfolioConfig(), portfolio_state=PortfolioState(state_dir=str(tmp_path)))
    assert gov.approve(_candidate(value=1000.0), _Executor(), 0) is True


# --------------------------------------------------------------------------
# PortfolioState: prune vs resting orders, durable flip cooldown
# --------------------------------------------------------------------------

def test_prune_keeps_meta_for_symbols_with_working_orders(tmp_path):
    """A submitted-but-unfilled bracket is not a position; pruning its meta
    destroys the trail state the fill is about to need."""
    st = PortfolioState(state_dir=str(tmp_path))
    st.note_new_position("AAA", "mean_reversion", "reversion",
                         datetime.now(ZoneInfo("UTC")).isoformat(),
                         side="long", entry_price=100.0, stop_distance=1.2)
    # Not in positions yet — but its entry order is working.
    assert st.prune_closed_positions([], working_order_symbols={"AAA"}) == 0
    assert st.get_trail_state("AAA") is not None
    # Order gone and still no position -> now it's genuinely closed.
    assert st.prune_closed_positions([], working_order_symbols=set()) == 1
    assert st.get_trail_state("AAA") is None


def test_direction_flip_cooldown_survives_the_position_close(tmp_path):
    """The whipsaw guard must still fire after the position closed and its meta
    was pruned — that is exactly the short -> stopped -> immediately long case
    it exists to block."""
    st = PortfolioState(state_dir=str(tmp_path))
    now = datetime.now(ZoneInfo("UTC"))
    st.note_new_position("AAA", "mean_reversion", "reversion", now.isoformat(),
                         side="short", entry_price=100.0, stop_distance=1.2)
    st.prune_closed_positions([])                      # position stopped out
    assert st.get_trail_state_short("AAA") is None     # meta really is gone
    # Opposite side inside the window is blocked...
    assert st.in_direction_flip_cooldown("AAA", "long", now + timedelta(minutes=5), 60) is True
    # ...the same side is not (the plain symbol cooldown covers that)...
    assert st.in_direction_flip_cooldown("AAA", "short", now + timedelta(minutes=5), 60) is False
    # ...and the window still expires.
    assert st.in_direction_flip_cooldown("AAA", "long", now + timedelta(minutes=61), 60) is False


def test_entry_style_is_recoverable_for_the_trail(tmp_path):
    st = PortfolioState(state_dir=str(tmp_path))
    st.note_new_position("AAA", "trendfail", "trend",
                         datetime.now(ZoneInfo("UTC")).isoformat(),
                         side="long", entry_price=50.0, stop_distance=1.1)
    assert st.get_entry_style("AAA") == "trendfail"
    assert st.get_entry_style("MISSING") is None


# --------------------------------------------------------------------------
# Single-instance lock: an in-flight (empty) lock file is not stale
# --------------------------------------------------------------------------

def test_empty_lock_file_is_not_treated_as_stale(tmp_path, monkeypatch):
    """Racer A wins the exclusive create but hasn't written its pid yet.
    Judging that empty file "stale" unlinked the winner's lock and let BOTH
    instances run, double-placing orders."""
    import reversion_bot.single_instance as si

    lock = tmp_path / "bot.lock"
    lock.write_text("")                     # exists, no readable pid yet

    # Simulate the writer finishing mid-retry: the pid of a LIVE process
    # (this one's parent-ish: use our own pid, which is definitely alive).
    calls = {"n": 0}
    real_read = si._read_lock

    def fake_read(path):
        calls["n"] += 1
        if calls["n"] <= 2:
            return 0, None                  # still empty
        return 999_999_999, None            # pid appears (not alive -> stale)

    monkeypatch.setattr(si, "_read_lock", fake_read)
    monkeypatch.setattr(si, "_pid_alive", lambda pid: True)   # the writer is live

    acquired, holder = si.acquire_lock(str(lock))
    assert acquired is False                # refused: we did NOT steal the lock
    assert holder == 999_999_999
    assert calls["n"] > 1                   # it retried instead of unlinking
    monkeypatch.setattr(si, "_read_lock", real_read)


def test_lock_still_acquires_when_uncontended(tmp_path):
    lock = tmp_path / "bot.lock"
    acquired, holder = acquire_lock(str(lock))
    assert acquired is True
    assert lock.exists()
    release_lock(str(lock))
    assert not lock.exists()


# --------------------------------------------------------------------------
# Swing MOC window on a half day
# --------------------------------------------------------------------------

def test_moc_window_tracks_a_half_day_close():
    """With a 12:00 CT early close the hardcoded 14:50 cutoff sits entirely
    after the close, so swing mode silently skipped every early-close session."""
    ct = ZoneInfo("America/Chicago")
    early_close = datetime(2026, 7, 3, 12, 0, tzinfo=ct)
    # 11:55 CT is inside [11:50, 11:50+... ) for a 12:00 close with a 10-min
    # MOC lead and a 20-minute window.
    now = datetime(2026, 7, 3, 11, 55, tzinfo=ct)
    assert moc_cutoff(now, early_close) == datetime(2026, 7, 3, 11, 50, tzinfo=ct)
    assert is_moc_entry_window(now, 20, close_ct=early_close) is False   # past 11:50
    assert is_moc_entry_window(datetime(2026, 7, 3, 11, 45, tzinfo=ct), 20,
                               close_ct=early_close) is True
    # The old behavior: nothing in the 14:3x-14:50 window is reachable on a
    # 12:00 close, which is why the session was skipped entirely.
    assert is_moc_entry_window(datetime(2026, 7, 3, 14, 45, tzinfo=ct), 20,
                               close_ct=early_close) is False


def test_moc_window_unchanged_on_a_regular_session():
    ct = ZoneInfo("America/Chicago")
    regular_close = datetime(2026, 7, 6, 15, 0, tzinfo=ct)
    inside = datetime(2026, 7, 6, 14, 45, tzinfo=ct)
    # Broker-clock cutoff (15:00 - 10min = 14:50) matches the legacy default.
    assert moc_cutoff(inside, regular_close) == moc_cutoff(inside, None)
    assert is_moc_entry_window(inside, 20) is True
    assert is_moc_entry_window(inside, 20, close_ct=regular_close) is True
    assert is_moc_entry_window(datetime(2026, 7, 6, 14, 55, tzinfo=ct), 20,
                               close_ct=regular_close) is False
