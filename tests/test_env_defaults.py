"""Env parsing + .env.example drift guards (audit H2/H3).

H2: parse_bool("") returned False — blanking a line like `USE_TRAILING_STOP=`
silently disabled default-ON safety features. Empty/unrecognized values now
fall back to the default.

H3: .env.example claimed "defaults shown" while contradicting the validated
code defaults on 8+ settings (ENABLE_SHORTS=true, USE_TREND_FILTER=False,
MIN_TRADE_SCORE=0.36, TARGET_ATR_MULTIPLE=2.00, ...). The drift test pins the
example file to the live defaults for every safety-relevant var so it can't
silently diverge again.
"""
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

if "alpaca_trade_api" not in sys.modules:
    _stub = types.ModuleType("alpaca_trade_api")
    _rest = types.ModuleType("alpaca_trade_api.rest")
    _rest.REST = object
    _stub.rest = _rest
    sys.modules["alpaca_trade_api"] = _stub
    sys.modules["alpaca_trade_api.rest"] = _rest

from main import parse_bool


# --- parse_bool --------------------------------------------------------------

@pytest.mark.parametrize("value,default,expected", [
    (None, True, True),
    (None, False, False),
    ("", True, True),            # blanked line keeps the default (was: False)
    ("", False, False),
    ("   ", True, True),
    ("true", False, True),
    ("YES", False, True),
    ("1", False, True),
    ("on", False, True),
    ("false", True, False),
    ("No", True, False),
    ("0", True, False),
    ("off", True, False),
    ("Ture", True, True),        # typo keeps the default (was: silently False)
    ("enabled", False, False),
])
def test_parse_bool(value, default, expected):
    assert parse_bool(value, default=default) is expected


# --- .env.example must match the code's effective defaults -------------------

def _env_example() -> dict:
    values = {}
    for line in (REPO / ".env.example").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        values[key.strip()] = val.split("#")[0].strip()
    return values


# Safety-relevant vars: expected values are the effective live defaults in
# main.py's env block. If a default is deliberately changed, update BOTH
# main.py and .env.example — this test exists to force that.
EXPECTED_DEFAULTS = {
    "ENABLE_SHORTS": "false",
    "USE_TREND_FILTER": "True",
    "USE_TRAILING_STOP": "true",
    "USE_MARKET_REGIME_FILTER": "true",
    "FLATTEN_CARRYOVER_ON_START": "true",
    "MIN_TRADE_SCORE": "0.45",
    "TARGET_ATR_MULTIPLE": "3.50",
    "TRAIL_ATR_MULTIPLE": "1.50",
    "RSI_MAX": "40.0",
    "RSI_MIN": "65.0",
    "OVERSOLD_GATE": "and",
    "MIN_PRICE": "2.0",
    "MIN_DOLLAR_VOLUME": "750000.0",
    "RISK_OFF_RSI_MIN": "55.0",
    "RISK_OFF_RI_SHORT_THRESHOLD": "0.30",
    "TRADE_LOOKBACK": "160",
    "ALPACA_DATA_FEED": "iex",
}


@pytest.mark.parametrize("key,expected", sorted(EXPECTED_DEFAULTS.items()))
def test_env_example_matches_code_default(key, expected):
    values = _env_example()
    assert key in values, f".env.example is missing {key}"
    got = values[key]
    if expected.lower() in ("true", "false"):
        assert got.lower() == expected.lower(), f"{key}: example={got!r} code default={expected!r}"
    else:
        try:
            assert float(got) == float(expected), f"{key}: example={got!r} code default={expected!r}"
        except ValueError:
            assert got.lower() == expected.lower(), f"{key}: example={got!r} code default={expected!r}"
