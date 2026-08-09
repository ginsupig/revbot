import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reversion_bot.indicators import calculate_rsi


def _df_from_close(close_values):
    close = pd.Series(close_values, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1_000.0,
        }
    )


def test_rsi_is_100_on_all_gains_window():
    df = _df_from_close(range(1, 50))
    rsi = calculate_rsi(df, length=14)
    assert float(rsi.iloc[-1]) == 100.0


def test_rsi_is_0_on_all_losses_window():
    df = _df_from_close(range(50, 1, -1))
    rsi = calculate_rsi(df, length=14)
    assert float(rsi.iloc[-1]) == 0.0


def test_rsi_is_50_on_flat_window():
    df = _df_from_close([100.0] * 49)
    rsi = calculate_rsi(df, length=14)
    assert float(rsi.iloc[-1]) == 50.0
