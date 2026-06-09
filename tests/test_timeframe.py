"""Regression for the live regime-filter trap: MARKET_REGIME_TIMEFRAME defaulted
to '1Day', but fetch_alpaca_bars' exact-match chain only accepted 'Day' — so the
SPY (and later sector) regime fetch raised every cycle, failed open to risk-on,
and never suppressed a long. _parse_timeframe must accept both spellings.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from run_real_backtest import _parse_timeframe


def test_alpaca_style_strings_resolve():
    # The exact value that broke live.
    assert _parse_timeframe("1Day").value == "1Day"
    assert _parse_timeframe("1Min").value == "1Min"
    assert _parse_timeframe("1Hour").value == "1Hour"


def test_bare_and_alias_forms_resolve():
    assert _parse_timeframe("Day").value == "1Day"
    assert _parse_timeframe("daily").value == "1Day"
    assert _parse_timeframe("Minute").value == "1Min"
    assert _parse_timeframe("Hour").value == "1Hour"
    assert _parse_timeframe("5Min").value == "5Min"
    assert _parse_timeframe("15Min").value == "15Min"


def test_case_and_whitespace_insensitive():
    assert _parse_timeframe("  1day  ").value == "1Day"
    assert _parse_timeframe("5MIN").value == "5Min"


def test_unsupported_raises():
    with pytest.raises(ValueError):
        _parse_timeframe("fortnight")
