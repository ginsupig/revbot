import pandas as pd
import numpy as np

from reversion_bot.config import ReversionConfig
from reversion_bot.engine import ReversionEngine


def make_df(n=120):
    idx = pd.date_range('2026-01-01', periods=n, freq='min')
    base = np.linspace(100, 101, n)
    noise = np.random.default_rng(7).normal(0, 0.15, n)
    close = base + noise
    open_ = close + np.random.default_rng(8).normal(0, 0.05, n)
    high = np.maximum(open_, close) + 0.10
    low = np.minimum(open_, close) - 0.10
    volume = np.full(n, 50000)
    return pd.DataFrame({
        'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume,
    }, index=idx)


def test_calculate_indicators_adds_columns():
    engine = ReversionEngine(ReversionConfig(min_history=60))
    df = engine.calculate_indicators(make_df())
    for col in ['sma', 'lb1', 'lb2', 'ri', 'rsi', 'adx', 'atr', 'vwap', 'trend_ema']:
        assert col in df.columns


def test_wait_when_no_valid_setup():
    engine = ReversionEngine(ReversionConfig(min_history=60))
    df = engine.calculate_indicators(make_df())
    decision = engine.get_decision(df, symbol='SPY')
    assert decision.signal == 'WAIT'
