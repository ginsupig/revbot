import numpy as np
import pandas as pd
from reversion_bot.analytics import profit_factor, sharpe_ratio, max_drawdown

def make_trades():
    np.random.seed(0)
    pnl = np.random.normal(0.001, 0.01, 100)
    df = pd.DataFrame({'pnl': pnl})
    return df

def test_profit_factor():
    trades = make_trades()
    pf = profit_factor(trades)
    assert pf > 0

def test_sharpe_ratio():
    trades = make_trades()
    sr = sharpe_ratio(trades)
    assert isinstance(sr, float)

def test_max_drawdown():
    trades = make_trades()
    dd = max_drawdown(trades)
    assert dd <= 0
