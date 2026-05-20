import numpy as np
import pandas as pd
from reversion_bot.autotune import AutoTuner

def dummy_strategy(df, fast=5, slow=10):
    df = df.copy()
    df['fast'] = df['close'].rolling(fast).mean()
    df['slow'] = df['close'].rolling(slow).mean()
    df['signal'] = (df['fast'] > df['slow']).astype(int)
    df['pnl'] = df['signal'] * df['close'].pct_change().fillna(0)
    return df

def profit_factor(trades):
    gross_profit = trades[trades['pnl'] > 0]['pnl'].sum()
    gross_loss = -trades[trades['pnl'] < 0]['pnl'].sum()
    if gross_loss == 0:
        return np.inf
    return gross_profit / gross_loss

def test_autotuner_tune():
    n = 100
    np.random.seed(1)
    close = np.cumsum(np.random.randn(n)) + 100
    df = pd.DataFrame({'close': close})
    param_grid = {'fast': [3, 5], 'slow': [8, 10]}
    tuner = AutoTuner(dummy_strategy, df, param_grid)
    best_params, best_score = tuner.tune(profit_factor, n_splits=3)
    assert isinstance(best_params, dict)
    assert best_score > 0
