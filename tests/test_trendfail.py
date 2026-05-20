import numpy as np
import pandas as pd
from reversion_bot.trendfail import trend_fail_continue_strategy

def test_trendfail_strategy_runs():
    n = 100
    np.random.seed(42)
    close = np.cumsum(np.random.randn(n)) + 100
    high = close + np.random.uniform(0, 1, n)
    low = close - np.random.uniform(0, 1, n)
    df = pd.DataFrame({'close': close, 'high': high, 'low': low})
    out = trend_fail_continue_strategy(df)
    assert 'signal' in out.columns
    assert 'pnl' in out.columns
    assert out['signal'].abs().max() <= 1
