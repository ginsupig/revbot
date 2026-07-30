"""Regression tests for main.py loop logic the audit found untested.

Covers the EOD window predicates, trailing-stop orchestration (branch selection
and WHICH stop multiple is used per style), and the submit-time spread gate.
"""
import asyncio
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if "alpaca_trade_api" not in sys.modules:
    _stub = types.ModuleType("alpaca_trade_api")
    _rest = types.ModuleType("alpaca_trade_api.rest")
    _rest.REST = object
    _stub.rest = _rest
    sys.modules["alpaca_trade_api"] = _stub
    sys.modules["alpaca_trade_api.rest"] = _rest

import main
from reversion_bot.config import RiskConfig

CT = ZoneInfo("America/Chicago")


# --------------------------------------------------------------------------
# EOD window predicates
# --------------------------------------------------------------------------

class _ClockExecutor:
    """Executor whose broker clock reports a fixed close time."""

    def __init__(self, close_utc):
        outer = self

        class _Clock:
            next_close = close_utc

        class _Client:
            def get_clock(self):
                return _Clock()

        self.client = _Client()


def _executor_closing_at(now_ct, hour, minute):
    close_ct = now_ct.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return _ClockExecutor(close_ct.astimezone(timezone.utc))


def _freeze_ct(monkeypatch, when_ct):
    """Freeze datetime.now(tz) inside main to a fixed CT instant."""
    real_datetime = main.datetime

    class _Frozen(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return when_ct.astimezone(tz) if tz else when_ct.replace(tzinfo=None)

    monkeypatch.setattr(main, "datetime", _Frozen)


def test_eod_liquidation_window_fires_only_inside_the_lead(monkeypatch):
    monkeypatch.setenv("EOD_LIQUIDATION_MINUTES", "10")
    main._CLOSE_CT_CACHE.clear()

    inside = datetime(2026, 7, 6, 14, 55, tzinfo=CT)     # 5 min before a 15:00 close
    _freeze_ct(monkeypatch, inside)
    assert main.is_eod_liquidation_window(_executor_closing_at(inside, 15, 0)) is True

    main._CLOSE_CT_CACHE.clear()
    before = datetime(2026, 7, 6, 14, 30, tzinfo=CT)     # 30 min out: too early
    _freeze_ct(monkeypatch, before)
    assert main.is_eod_liquidation_window(_executor_closing_at(before, 15, 0)) is False

    main._CLOSE_CT_CACHE.clear()
    after = datetime(2026, 7, 6, 15, 5, tzinfo=CT)       # past the close
    _freeze_ct(monkeypatch, after)
    assert main.is_eod_liquidation_window(_executor_closing_at(after, 15, 0)) is False


def test_eod_liquidation_window_tracks_a_half_day_close(monkeypatch):
    """On a 12:00 CT early close the window must move with it — a regular-close
    assumption would leave positions carried overnight."""
    monkeypatch.setenv("EOD_LIQUIDATION_MINUTES", "10")
    main._CLOSE_CT_CACHE.clear()
    inside = datetime(2026, 7, 3, 11, 55, tzinfo=CT)
    _freeze_ct(monkeypatch, inside)
    assert main.is_eod_liquidation_window(_executor_closing_at(inside, 12, 0)) is True

    main._CLOSE_CT_CACHE.clear()
    regular_time = datetime(2026, 7, 3, 14, 55, tzinfo=CT)   # after the early close
    _freeze_ct(monkeypatch, regular_time)
    assert main.is_eod_liquidation_window(_executor_closing_at(regular_time, 12, 0)) is False


def test_eod_entry_cutoff_blocks_late_entries_and_can_be_disabled(monkeypatch):
    monkeypatch.setenv("EOD_ENTRY_CUTOFF_MINUTES", "20")
    main._CLOSE_CT_CACHE.clear()
    late = datetime(2026, 7, 6, 14, 45, tzinfo=CT)      # 15 min before close
    _freeze_ct(monkeypatch, late)
    assert main.is_eod_entry_cutoff(_executor_closing_at(late, 15, 0)) is True

    main._CLOSE_CT_CACHE.clear()
    early = datetime(2026, 7, 6, 10, 0, tzinfo=CT)
    _freeze_ct(monkeypatch, early)
    assert main.is_eod_entry_cutoff(_executor_closing_at(early, 15, 0)) is False

    monkeypatch.setenv("EOD_ENTRY_CUTOFF_MINUTES", "0")   # disabled
    main._CLOSE_CT_CACHE.clear()
    _freeze_ct(monkeypatch, late)
    assert main.is_eod_entry_cutoff(_executor_closing_at(late, 15, 0)) is False


# --------------------------------------------------------------------------
# Trailing-stop orchestration
# --------------------------------------------------------------------------

class _TrailPosition:
    def __init__(self, symbol, qty, current_price):
        self.symbol = symbol
        self.qty = str(qty)
        self.current_price = str(current_price)


class _TrailExecutor:
    def __init__(self, positions):
        self._positions = positions
        self.long_calls = []
        self.short_calls = []

    def get_positions(self):
        return self._positions

    def update_trailing_stop(self, symbol, entry, stop_distance, hw, last, stop_mult, trail_mult):
        self.long_calls.append(dict(symbol=symbol, entry=entry, stop_distance=stop_distance,
                                    water=hw, stop_mult=stop_mult, trail_mult=trail_mult))
        return 101.0

    def update_trailing_stop_short(self, symbol, entry, stop_distance, lw, last, stop_mult, trail_mult):
        self.short_calls.append(dict(symbol=symbol, entry=entry, stop_distance=stop_distance,
                                     water=lw, stop_mult=stop_mult, trail_mult=trail_mult))
        return 99.0


class _TrailState:
    """Minimal PortfolioState stand-in with per-symbol style + trail state."""

    def __init__(self, states, styles):
        self._states = states       # symbol -> (entry, stop_distance, water)
        self._styles = styles       # symbol -> entry_style

    def get_trail_state(self, symbol):
        return self._states.get(symbol)

    def get_trail_state_short(self, symbol):
        return self._states.get(symbol)

    def get_entry_style(self, symbol):
        return self._styles.get(symbol)

    def update_high_water(self, symbol, price):
        return max(self._states[symbol][2], price)

    def update_low_water(self, symbol, price):
        return min(self._states[symbol][2], price)


def test_trailing_uses_long_branch_for_positive_qty_and_short_for_negative():
    cfg = RiskConfig()
    ex = _TrailExecutor([_TrailPosition("LONGY", 10, 105.0),
                         _TrailPosition("SHORTY", -10, 95.0)])
    st = _TrailState(
        states={"LONGY": (100.0, 1.2, 104.0), "SHORTY": (100.0, 1.2, 96.0)},
        styles={"LONGY": "mean_reversion", "SHORTY": "mean_reversion"},
    )
    asyncio.run(main.manage_trailing_stops(ex, st, cfg))

    assert [c["symbol"] for c in ex.long_calls] == ["LONGY"]
    assert [c["symbol"] for c in ex.short_calls] == ["SHORTY"]
    # High-water ratchets UP toward the price, low-water ratchets DOWN.
    assert ex.long_calls[0]["water"] == 105.0
    assert ex.short_calls[0]["water"] == 95.0


def test_trailing_passes_the_style_specific_stop_multiple():
    """A trendfail position's stop_distance was built with
    trendfail_stop_atr_multiple; passing the mean-reversion multiple made the
    derived trail ~8% too tight at defaults."""
    cfg = RiskConfig()
    atr = 2.0
    ex = _TrailExecutor([_TrailPosition("TF", 10, 110.0),
                         _TrailPosition("MR", 10, 110.0)])
    st = _TrailState(
        states={"TF": (100.0, cfg.trendfail_stop_atr_multiple * atr, 108.0),
                "MR": (100.0, cfg.stop_atr_multiple * atr, 108.0)},
        styles={"TF": "trendfail", "MR": "mean_reversion"},
    )
    asyncio.run(main.manage_trailing_stops(ex, st, cfg))

    by_symbol = {c["symbol"]: c for c in ex.long_calls}
    assert by_symbol["TF"]["stop_mult"] == cfg.trendfail_stop_atr_multiple
    assert by_symbol["MR"]["stop_mult"] == cfg.stop_atr_multiple
    # Both must recover the SAME underlying ATR from their own stop distance.
    for c in by_symbol.values():
        assert c["stop_distance"] / c["stop_mult"] == atr


def test_trailing_skips_symbols_without_trail_state():
    cfg = RiskConfig()
    ex = _TrailExecutor([_TrailPosition("NOSTATE", 10, 105.0)])
    st = _TrailState(states={}, styles={})
    asyncio.run(main.manage_trailing_stops(ex, st, cfg))
    assert ex.long_calls == [] and ex.short_calls == []


# --------------------------------------------------------------------------
# Submit-time spread gate
# --------------------------------------------------------------------------

class _SpreadExecutor:
    def __init__(self, spread):
        self._spread = spread

    def latest_spread_bps(self, symbol):
        return self._spread


def test_spread_gate_blocks_wide_spreads_and_allows_tight_ones():
    ex_wide = _SpreadExecutor(60.0)
    ex_tight = _SpreadExecutor(12.0)
    assert asyncio.run(main._spread_blocks_entry(ex_wide, "AAA", 40.0)) is True
    assert asyncio.run(main._spread_blocks_entry(ex_tight, "AAA", 40.0)) is False


def test_spread_gate_fails_open_without_a_quote_and_when_disabled():
    ex_noquote = _SpreadExecutor(None)
    assert asyncio.run(main._spread_blocks_entry(ex_noquote, "AAA", 40.0)) is False
    # Limit of 0 disables the gate entirely (SPREAD_GATE_AT_SUBMIT=false).
    assert asyncio.run(main._spread_blocks_entry(_SpreadExecutor(999.0), "AAA", 0.0)) is False


def test_wide_spread_candidate_is_not_submitted():
    """End-to-end through execute_candidates: an approved candidate whose quote
    is too wide must not reach submit_order."""

    class _Gov:
        def rank_candidates(self, c):
            return list(c)

        def approve(self, candidate, executor, n):
            return True

    class _Ex(_SpreadExecutor):
        def __init__(self, spread):
            super().__init__(spread)
            self.submitted = []

        def submit_order(self, candidate):
            self.submitted.append(candidate["symbol"])

    class _St:
        def __init__(self):
            self.updated = []

        def update(self, candidate):
            self.updated.append(candidate["symbol"])

    ex = _Ex(80.0)
    st = _St()
    asyncio.run(main.execute_candidates(
        _Gov(), ex, st, [{"symbol": "WIDE", "go_long": True, "trade_score": 0.9}],
        max_spread_bps=40.0))
    assert ex.submitted == []
    assert st.updated == []          # no state written for a skipped entry

    ex_ok = _Ex(10.0)
    st_ok = _St()
    asyncio.run(main.execute_candidates(
        _Gov(), ex_ok, st_ok, [{"symbol": "TIGHT", "go_long": True, "trade_score": 0.9}],
        max_spread_bps=40.0))
    assert ex_ok.submitted == ["TIGHT"]
