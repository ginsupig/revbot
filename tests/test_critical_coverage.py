"""Critical coverage gaps identified by the audit.

Tests for:
1. engine.validate_input() — missing columns, insufficient rows
2. engine.is_market_safe() — all 6 safety gates
3. engine.get_decision() ATR <= 0 guard
4. trailing_stop_price_short() and should_lower_stop()
5. execution._validate_plan() for longs
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from reversion_bot.config import ReversionConfig
from reversion_bot.engine import ReversionEngine
from reversion_bot.trailing import (
    trailing_stop_price_short,
    should_lower_stop,
    trailing_stop_price,
)
from reversion_bot.models import PositionPlan
from reversion_bot.execution import AlpacaExecutor


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_df(n=160, close_val=100.0, volume=50_000):
    idx = pd.date_range("2026-01-01", periods=n, freq="min")
    close = np.full(n, close_val)
    return pd.DataFrame({
        "open": close,
        "high": close + 0.10,
        "low": close - 0.10,
        "close": close,
        "volume": np.full(n, volume),
    }, index=idx)


def _engine(min_history=60, **kw):
    return ReversionEngine(ReversionConfig(min_history=min_history, **kw))


def _indicators_df(engine, n=160, close_val=100.0, volume=50_000):
    """Return a df that has already passed through calculate_indicators."""
    return engine.calculate_indicators(_make_df(n, close_val, volume))


# ── 1. validate_input ───────────────────────────────────────────────────

class TestValidateInput:
    def test_missing_columns_raises(self):
        engine = _engine()
        df = pd.DataFrame({"open": [1], "high": [2], "low": [0.5]})
        with pytest.raises(ValueError, match="Missing required columns"):
            engine.validate_input(df)

    def test_insufficient_rows_raises(self):
        engine = _engine(min_history=100)
        df = _make_df(n=50)
        with pytest.raises(ValueError, match="Not enough rows"):
            engine.validate_input(df)

    def test_exact_min_history_passes(self):
        engine = _engine(min_history=60)
        df = _make_df(n=60)
        engine.validate_input(df)  # should not raise

    def test_valid_input_passes(self):
        engine = _engine(min_history=60)
        df = _make_df(n=120)
        engine.validate_input(df)  # should not raise


# ── 2. is_market_safe — all 6 gates ─────────────────────────────────────

class TestIsMarketSafe:
    def _safe_df(self, engine, **overrides):
        """Build an indicators df then override the last row's values."""
        df = _indicators_df(engine, n=160)
        for col, val in overrides.items():
            df.iloc[-1, df.columns.get_loc(col)] = val
        return df

    def test_indicators_not_ready(self):
        engine = _engine()
        df = _indicators_df(engine)
        df.iloc[-1, df.columns.get_loc("adx")] = np.nan
        result = engine.is_market_safe(df)
        assert not result.is_safe
        assert result.reason == "Indicators_Not_Ready"

    def test_price_too_low(self):
        engine = _engine(min_price=5.0)
        df = self._safe_df(engine, close=2.0)
        result = engine.is_market_safe(df)
        assert not result.is_safe
        assert result.reason == "Price_Too_Low"

    def test_dollar_volume_too_low(self):
        engine = _engine(min_dollar_volume=1_000_000)
        df = self._safe_df(engine, avg_dollar_volume=500_000.0)
        result = engine.is_market_safe(df)
        assert not result.is_safe
        assert result.reason == "Dollar_Volume_Too_Low"

    def test_spread_too_wide(self):
        engine = _engine(max_spread_bps=10.0)
        df = _indicators_df(engine)
        df["spread_bps"] = 5.0
        df.iloc[-1, df.columns.get_loc("spread_bps")] = 50.0
        result = engine.is_market_safe(df)
        assert not result.is_safe
        assert result.reason == "Spread_Too_Wide"

    def test_momentum_too_extended(self):
        engine = _engine(adx_hard_max=50.0, rsi_hard_max=70.0)
        df = self._safe_df(engine, adx=60.0, rsi=80.0)
        result = engine.is_market_safe(df)
        assert not result.is_safe
        assert result.reason == "Momentum_Too_Extended"

    def test_downtrend_too_extended(self):
        engine = _engine(adx_hard_max=50.0, rsi_hard_min=30.0)
        df = self._safe_df(engine, adx=60.0, rsi=20.0)
        result = engine.is_market_safe(df)
        assert not result.is_safe
        assert result.reason == "Downtrend_Too_Extended"

    def test_safe_when_all_ok(self):
        engine = _engine()
        df = _indicators_df(engine)
        result = engine.is_market_safe(df)
        assert result.is_safe
        assert result.reason == "Safe"


# ── 3. ATR <= 0 guard ───────────────────────────────────────────────────

class TestAtrGuard:
    def _make_safe_df_with_atr(self, engine, atr_val):
        """Build a df with valid indicators, then force ATR to a specific value."""
        n = 160
        idx = pd.date_range("2026-01-01", periods=n, freq="min")
        base = np.linspace(100, 101, n)
        noise = np.random.default_rng(7).normal(0, 0.15, n)
        close = base + noise
        open_ = close + np.random.default_rng(8).normal(0, 0.05, n)
        high = np.maximum(open_, close) + 0.10
        low = np.minimum(open_, close) - 0.10
        raw = pd.DataFrame({"open": open_, "high": high, "low": low,
                            "close": close, "volume": np.full(n, 50000)}, index=idx)
        df = engine.calculate_indicators(raw)
        # Ensure safety checks pass (indicators are valid)
        df.iloc[-1, df.columns.get_loc("atr")] = atr_val
        return df

    def test_zero_atr_returns_wait(self):
        engine = _engine()
        df = self._make_safe_df_with_atr(engine, 0.0)
        decision = engine.get_decision(df, symbol="TEST")
        assert decision.signal == "WAIT"
        assert decision.reason == "ATR_Invalid"

    def test_negative_atr_returns_wait(self):
        engine = _engine()
        df = self._make_safe_df_with_atr(engine, -1.0)
        decision = engine.get_decision(df, symbol="TEST")
        assert decision.signal == "WAIT"
        assert decision.reason == "ATR_Invalid"


# ── 3b. ADX trend filter + AND oversold gate ────────────────────────────

class TestAdxAndOversoldGate:
    def _make_df_with_values(self, engine, adx, rsi, ri, close_val=95.0, lb1_val=96.0):
        """Build indicators df then override key values on the last row."""
        n = 160
        idx = pd.date_range("2026-01-01", periods=n, freq="min")
        base = np.linspace(100, 101, n)
        noise = np.random.default_rng(7).normal(0, 0.15, n)
        close = base + noise
        open_ = close + np.random.default_rng(8).normal(0, 0.05, n)
        high = np.maximum(open_, close) + 0.10
        low = np.minimum(open_, close) - 0.10
        raw = pd.DataFrame({"open": open_, "high": high, "low": low,
                            "close": close, "volume": np.full(n, 50000)}, index=idx)
        df = engine.calculate_indicators(raw)
        # Force last row values to test specific conditions
        df.iloc[-1, df.columns.get_loc("adx")] = adx
        df.iloc[-1, df.columns.get_loc("rsi")] = rsi
        df.iloc[-1, df.columns.get_loc("ri")] = ri
        df.iloc[-1, df.columns.get_loc("close")] = close_val
        df.iloc[-1, df.columns.get_loc("lb1")] = lb1_val
        return df

    def test_adx_too_strong_blocks_entry(self):
        engine = _engine(adx_max=40.0, oversold_gate="or", rsi_max=48.0, use_trend_filter=False)
        df = self._make_df_with_values(engine, adx=45.0, rsi=30.0, ri=-0.8)
        decision = engine.get_decision(df, symbol="TEST")
        assert decision.signal == "WAIT"
        assert decision.reason == "ADX_Trend_Too_Strong"

    def test_adx_below_max_allows_entry(self):
        engine = _engine(adx_max=40.0, oversold_gate="or", rsi_max=48.0, use_trend_filter=False)
        df = self._make_df_with_values(engine, adx=35.0, rsi=30.0, ri=-0.8)
        decision = engine.get_decision(df, symbol="TEST")
        # Should not be blocked by ADX (may still be WAIT for other reasons)
        assert decision.reason != "ADX_Trend_Too_Strong"

    def test_and_gate_requires_both(self):
        engine = _engine(oversold_gate="and", rsi_max=40.0, ri_threshold=-0.5,
                         adx_max=50.0, use_trend_filter=False)
        # RI oversold but RSI NOT oversold -> should fail AND gate
        df = self._make_df_with_values(engine, adx=25.0, rsi=45.0, ri=-0.8)
        decision = engine.get_decision(df, symbol="TEST")
        assert decision.reason == "RI_Not_Oversold"  # AND gate failed

    def test_and_gate_passes_when_both_oversold(self):
        engine = _engine(oversold_gate="and", rsi_max=40.0, ri_threshold=-0.5,
                         adx_max=50.0, use_trend_filter=False)
        df = self._make_df_with_values(engine, adx=25.0, rsi=30.0, ri=-0.8)
        decision = engine.get_decision(df, symbol="TEST")
        assert decision.reason != "RI_Not_Oversold"

    def test_or_gate_passes_with_rsi_only(self):
        engine = _engine(oversold_gate="or", rsi_max=48.0, ri_threshold=-0.5,
                         adx_max=50.0, use_trend_filter=False)
        # RI NOT oversold but RSI IS -> OR gate passes
        df = self._make_df_with_values(engine, adx=25.0, rsi=40.0, ri=-0.2)
        decision = engine.get_decision(df, symbol="TEST")
        assert decision.reason != "RI_Not_Oversold"


# ── 4. Short trailing stop ──────────────────────────────────────────────

class TestTrailingStopShort:
    def test_basic_short_trail(self):
        # Entry at 100, stop_distance = 2 (stop at 102), low_water drops to 95
        stop = trailing_stop_price_short(
            entry_price=100.0,
            stop_distance=2.0,
            low_water=95.0,
            stop_atr_mult=1.0,
            trail_atr_mult=1.5,
        )
        # trail_dist = (1.5/1.0) * 2.0 = 3.0
        # trail candidate = 95 + 3.0 = 98.0
        # init_stop = 100 + 2.0 = 102.0
        # min(102, 98) = 98.0
        assert abs(stop - 98.0) < 1e-6

    def test_short_trail_never_above_init_stop(self):
        # low_water hasn't moved much, trail stays at init_stop
        stop = trailing_stop_price_short(
            entry_price=100.0,
            stop_distance=2.0,
            low_water=100.0,
            stop_atr_mult=1.0,
            trail_atr_mult=1.5,
        )
        # trail candidate = 100 + 3.0 = 103.0
        # init_stop = 102.0
        # min(102, 103) = 102.0
        assert abs(stop - 102.0) < 1e-6

    def test_short_trail_zero_stop_distance(self):
        stop = trailing_stop_price_short(
            entry_price=100.0,
            stop_distance=0.0,
            low_water=95.0,
            stop_atr_mult=1.0,
            trail_atr_mult=1.5,
        )
        assert abs(stop - 100.0) < 1e-6

    def test_should_lower_stop_basic(self):
        # Current stop at 102, desired at 99, last price at 95 (well below)
        assert should_lower_stop(102.0, 99.0, 95.0) is True

    def test_should_lower_stop_rejects_at_market(self):
        # Desired stop at or below last price would trigger immediately
        assert should_lower_stop(102.0, 94.0, 95.0) is False

    def test_should_lower_stop_rejects_higher_stop(self):
        # Desired is above current — wrong direction for lowering
        assert should_lower_stop(99.0, 100.0, 95.0) is False

    def test_should_lower_stop_rejects_tiny_move(self):
        # Move is less than min_increment
        assert should_lower_stop(100.0, 99.999, 95.0, min_increment=0.01) is False


# ── 5. _validate_plan for longs ─────────────────────────────────────────

class TestValidatePlanLong:
    def _plan(self, **overrides):
        defaults = dict(
            qty=10, entry_price=100.0, stop_price=95.0, target_price=110.0,
            risk_per_share=5.0, reward_per_share=10.0, rr_ratio=2.0,
            position_value=1000.0, max_account_risk=50.0,
        )
        defaults.update(overrides)
        return PositionPlan(**defaults)

    def test_valid_long_plan(self):
        AlpacaExecutor._validate_plan(self._plan())  # should not raise

    def test_zero_qty_raises(self):
        with pytest.raises(ValueError, match="Quantity"):
            AlpacaExecutor._validate_plan(self._plan(qty=0))

    def test_negative_entry_raises(self):
        with pytest.raises(ValueError, match="Entry price"):
            AlpacaExecutor._validate_plan(self._plan(entry_price=-1.0))

    def test_stop_above_entry_raises(self):
        with pytest.raises(ValueError, match="Stop price must be below"):
            AlpacaExecutor._validate_plan(self._plan(stop_price=105.0))

    def test_stop_equal_entry_raises(self):
        with pytest.raises(ValueError, match="Stop price must be below"):
            AlpacaExecutor._validate_plan(self._plan(stop_price=100.0))

    def test_target_below_entry_raises(self):
        with pytest.raises(ValueError, match="Target price must be above"):
            AlpacaExecutor._validate_plan(self._plan(target_price=90.0))

    def test_target_equal_entry_raises(self):
        with pytest.raises(ValueError, match="Target price must be above"):
            AlpacaExecutor._validate_plan(self._plan(target_price=100.0))
