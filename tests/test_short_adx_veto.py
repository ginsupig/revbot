"""Short-side ADX trend-strength veto (audit H5).

The long side rejects reversion entries when ADX > adx_max (a strong trend:
pullbacks are continuations). The short side had NO equivalent — it would
short an ADX-45 uptrend rip the long side would call ADX_Trend_Too_Strong,
i.e. fade a breakout. The veto is now mirrored.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reversion_bot.config import ReversionConfig
from reversion_bot.engine import ReversionEngine
from reversion_bot.models import SafetyDecision


def _short_eval(adx: float, rsi: float = 75.0):
    engine = ReversionEngine(ReversionConfig(enable_shorts=True, use_trend_filter=False))
    row = pd.Series({
        "adx": adx, "open": 111.0, "close": 110.0, "volume": 1_000_000.0,
        "vwap": 100.0, "avg_volume": 900_000.0, "trend_ema": 100.0,
    })
    return engine._evaluate_short(
        row=row,
        prev=None,
        safety=SafetyDecision(True, "Safe"),
        symbol="T",
        close=110.0, lb1=90.0, lb2=85.0, ub1=105.0, ub2=112.0,
        sma=97.0, ri=0.9, current_rsi=rsi, atr=2.0, vwap=100.0,
        trend_ema=100.0,
    )


def test_valid_short_fires_below_adx_max():
    dec = _short_eval(adx=30.0)
    assert dec is not None
    assert dec.signal == "SHORT_REVERSION"


def test_short_vetoed_when_trend_too_strong():
    # ADX 45 > adx_max 40: the long side would say ADX_Trend_Too_Strong;
    # the short side used to fire here (RSI 75 doesn't trip the hard guard,
    # which needs RSI <= rsi_hard_min).
    assert _short_eval(adx=45.0) is None


def test_short_vetoed_at_adx_hard_range_too():
    # Even past adx_hard_max with a HIGH RSI the reflected hard guard never
    # applied (it requires RSI <= 30); the new adx_max veto covers it.
    assert _short_eval(adx=55.0, rsi=75.0) is None
