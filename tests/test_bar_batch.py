import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from reversion_bot.bar_batch import split_bars_by_symbol


def _multi_symbol_df():
    # Mimic alpaca-py's .df: MultiIndex (symbol, timestamp) with OHLCV columns.
    idx = pd.MultiIndex.from_tuples(
        [("AAPL", "2026-06-29"), ("AAPL", "2026-06-30"),
         ("MSFT", "2026-06-29"), ("MSFT", "2026-06-30")],
        names=["symbol", "timestamp"],
    )
    return pd.DataFrame(
        {"open": [1, 2, 3, 4], "high": [1, 2, 3, 4], "low": [1, 2, 3, 4],
         "close": [10, 11, 20, 21], "volume": [100, 100, 200, 200]},
        index=idx,
    )


def test_splits_into_one_frame_per_symbol():
    out = split_bars_by_symbol(_multi_symbol_df())
    assert set(out) == {"AAPL", "MSFT"}
    assert len(out["AAPL"]) == 2 and len(out["MSFT"]) == 2


def test_each_frame_has_date_column_not_timestamp():
    out = split_bars_by_symbol(_multi_symbol_df())
    for df in out.values():
        assert "date" in df.columns and "timestamp" not in df.columns


def test_per_symbol_values_are_isolated():
    out = split_bars_by_symbol(_multi_symbol_df())
    assert list(out["AAPL"]["close"]) == [10, 11]
    assert list(out["MSFT"]["close"]) == [20, 21]


def test_index_is_reset_per_group():
    out = split_bars_by_symbol(_multi_symbol_df())
    # MSFT's rows were positions 2,3 in the source; after split they must be 0,1.
    assert list(out["MSFT"].index) == [0, 1]


def test_empty_or_none_returns_empty_dict():
    assert split_bars_by_symbol(None) == {}
    assert split_bars_by_symbol(pd.DataFrame()) == {}


def test_single_symbol_frame_without_symbol_column():
    # A one-symbol df may come back with just a timestamp index and no symbol level.
    df = pd.DataFrame(
        {"close": [10, 11]},
        index=pd.Index(["2026-06-29", "2026-06-30"], name="timestamp"),
    )
    out = split_bars_by_symbol(df, single_symbol="NVDA")
    assert set(out) == {"NVDA"}
    assert "date" in out["NVDA"].columns
    assert list(out["NVDA"]["close"]) == [10, 11]


def test_no_symbol_column_and_no_fallback_yields_empty():
    df = pd.DataFrame({"close": [1]}, index=pd.Index(["2026-06-29"], name="timestamp"))
    assert split_bars_by_symbol(df) == {}
