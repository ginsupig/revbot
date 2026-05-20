import numpy as np
import pandas as pd
from reversion_bot.walkforward import run_walkforward_backtest
from reversion_bot.analytics import profit_factor, sharpe_ratio, max_drawdown

def show_walkforward_results():
    n = 120
    np.random.seed(42)
    close = np.cumsum(np.random.randn(n)) + 100
    volume = np.random.randint(1000, 2000, n)
    open_ = close + np.random.normal(0, 0.1, n)
    high = np.maximum(open_, close) + 0.1
    low = np.minimum(open_, close) - 0.1
    df = pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume})
    results, _, _ = run_walkforward_backtest(df, n_splits=3)
    summary = []
    for i, res in enumerate(results):
        pf = profit_factor(res)
        sr = sharpe_ratio(res)
        dd = max_drawdown(res)
        # Print more details for debugging
        print(f"Fold {i+1} summary:")
        print(res[['close','signal','position','pnl']].tail(10))
        print(f"PF={pf}, Sharpe={sr}, MaxDD={dd}")
        summary.append({'fold': i+1, 'profit_factor': pf, 'sharpe': sr, 'max_drawdown': dd})
    print('Walkforward Summary:')
    for row in summary:
        print(row)
    return summary

if __name__ == "__main__":
    show_walkforward_results()
