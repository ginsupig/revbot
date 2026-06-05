import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from reversion_bot.market_regime import is_risk_off, suppress_longs_if_risk_off


def _bars(values):
    return pd.DataFrame({"close": values})


def test_risk_off_true_when_below_trend():
    # Steady decline -> last close sits below its EMA -> risk-off.
    bars = _bars(np.linspace(120, 100, 80))
    assert is_risk_off(bars, ema_length=50) is True


def test_risk_on_when_above_trend():
    # Steady climb -> last close above its EMA -> risk-on.
    bars = _bars(np.linspace(100, 120, 80))
    assert is_risk_off(bars, ema_length=50) is False


def test_fail_open_on_thin_or_missing_data():
    assert is_risk_off(None, ema_length=50) is False
    assert is_risk_off(_bars([100, 101, 102]), ema_length=50) is False          # too few rows
    assert is_risk_off(pd.DataFrame({"open": [1, 2, 3]}), ema_length=2) is False  # no close col


def test_suppress_drops_longs_keeps_shorts():
    cands = [
        {"symbol": "AAA", "go_long": True},
        {"symbol": "BBB", "go_short": True},
        {"symbol": "CCC", "go_long": True},
    ]
    kept, dropped = suppress_longs_if_risk_off(cands, risk_off=True)
    assert [c["symbol"] for c in kept] == ["BBB"]
    assert [c["symbol"] for c in dropped] == ["AAA", "CCC"]


def test_suppress_noop_when_risk_on():
    cands = [{"symbol": "AAA", "go_long": True}]
    kept, dropped = suppress_longs_if_risk_off(cands, risk_off=False)
    assert kept == cands and dropped == []
