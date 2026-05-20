import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from reversion_bot.walkforward import run_walkforward_backtest

def test_run_walkforward_backtest():
    n = 120
    np.random.seed(42)
    close = np.cumsum(np.random.randn(n)) + 100
    volume = np.random.randint(1000, 2000, n)
    open_ = close + np.random.normal(0, 0.1, n)
    high = np.maximum(open_, close) + 0.1
    low = np.minimum(open_, close) - 0.1
    df = pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume})
    results, oos_metrics, _ = run_walkforward_backtest(df, n_splits=3)
    assert len(results) == 3
    for res in results:
        assert 'pnl' in res.columns
