"""Service-level guard: an engine WAIT must never become a trade.

Regression tests for the router-bypass bug where component-score bonuses
(band depth + RI stretch + RSI softness = up to 0.45 on top of the 0.15 base)
could push an engine-REJECTED setup past the entry threshold, and
`go_long = passes_score and not is_short_signal` never re-checked the engine
signal — so the bot would buy names the safety guards explicitly vetoed
(the ADX 97 / RSI 0.3 falling knife reproduced in the 2026-07 audit).
"""
import pandas as pd
import pytest

from reversion_bot.config import PerformanceConfig
from reversion_bot.models import ReversionDecision
from reversion_bot.service import ReversionService


def _svc(tmp_path):
    return ReversionService(
        performance_config=PerformanceConfig(state_dir=str(tmp_path)),
        log_file=str(tmp_path / "svc.log"),
    )


def _enriched(close=80.0, ri=-1.5, rsi=5.0, adx=20.0):
    """Minimal indicator frame with the columns evaluate_symbol reads."""
    return pd.DataFrame({
        "open": [close + 1.0],
        "high": [close + 2.0],
        "low": [close - 2.0],
        "close": [close],
        "volume": [1_000_000.0],
        "ri": [ri],
        "rsi": [rsi],
        "adx": [adx],
        "trend_following_signal": [0],
        "macd_line": [0.0],
        "macd_signal": [0.0],
        "trend_ema": [close],
    })


def _decision(signal, reason, close=80.0, ri=-1.5, rsi=5.0, adx=96.0):
    """A maximally 'attractive' setup: deep below the band (full depth bonus),
    extreme RI stretch, soft RSI — mr component scores 0.60 as WAIT, 0.80 when
    validated. Exactly the shape that used to leak through the router."""
    return ReversionDecision(
        signal=signal, reason=reason, symbol="T",
        close=close, lb1=100.0, lb2=90.0, ub1=120.0, ub2=130.0,
        sma=110.0, ri=ri, rsi=rsi, adx=adx, atr=2.0, vwap=100.0,
    )


def _evaluate(tmp_path, decision, adx=96.0, rsi=5.0):
    svc = _svc(tmp_path)
    enriched = _enriched(adx=adx, rsi=rsi)
    svc.engine.calculate_indicators = lambda df: enriched
    svc.engine.get_decision = lambda df, symbol=None, short_bias=False: decision
    return svc.evaluate_symbol("T", enriched, account_equity=100_000.0)


SAFETY_VETOES = [
    "Dollar_Volume_Too_Low",
    "Spread_Too_Wide",
    "Price_Too_Low",
    "Momentum_Too_Extended",
    "Downtrend_Too_Extended",
    "ATR_Invalid",
]


@pytest.mark.parametrize("reason", SAFETY_VETOES)
def test_engine_safety_veto_is_a_hard_gate(tmp_path, reason):
    result = _evaluate(tmp_path, _decision("WAIT", reason))
    assert result["go_long"] is False
    assert result["go_short"] is False
    assert result["router_reason"] == f"hard_gate:{reason}"
    assert "position_plan" not in result


def test_indicators_not_ready_is_a_hard_gate(tmp_path):
    # Early-return decision: every numeric field is None.
    decision = ReversionDecision(signal="WAIT", reason="Indicators_Not_Ready", symbol="T")
    result = _evaluate(tmp_path, decision)
    assert result["go_long"] is False
    assert result["go_short"] is False
    assert result["router_reason"] == "hard_gate:Indicators_Not_Ready"


def test_falling_knife_reproduction_never_goes_long(tmp_path):
    # The audit's end-to-end repro: accelerating downtrend, ADX 96.9, RSI 0.3,
    # engine says Downtrend_Too_Extended — the old router said go_long=True.
    decision = _decision("WAIT", "Downtrend_Too_Extended", ri=-2.0, rsi=0.3, adx=96.9)
    result = _evaluate(tmp_path, decision, adx=96.9, rsi=0.3)
    assert result["go_long"] is False
    assert "position_plan" not in result


@pytest.mark.parametrize("reason", [
    "RI_Not_Oversold",
    "Not_In_Reversion_Zone",
    "No_Reclaim_Trigger",
    "ADX_Trend_Too_Strong",
])
def test_unvalidated_setup_cannot_trade_as_mean_reversion(tmp_path, reason):
    # Non-safety WAIT reasons aren't hard gates (other styles may trade), but a
    # mean-reversion LONG must carry the engine's LONG_REVERSION signal even
    # when the mr component score clears the threshold (0.60 here).
    result = _evaluate(tmp_path, _decision("WAIT", reason, adx=20.0), adx=20.0)
    assert result["go_long"] is False
    assert "position_plan" not in result
    if result["entry_style"] == "mean_reversion":
        assert result["router_reason"] == "mr_requires_engine_signal"


def test_validated_long_reversion_still_trades(tmp_path):
    # Positive control: the identical setup WITH the engine signal goes long.
    decision = _decision("LONG_REVERSION", "Validated_Long_Reversion", adx=20.0)
    result = _evaluate(tmp_path, decision, adx=20.0)
    assert result["go_long"] is True
    assert result["entry_style"] == "mean_reversion"
    assert "position_plan" in result
