import pandas as pd
import numpy as np
from reversion_bot.service import ReversionService

def test_logging_runs():
    n = 80
    np.random.seed(42)
    close = np.cumsum(np.random.randn(n)) + 100
    volume = np.random.randint(1000, 2000, n)
    open_ = close + np.random.normal(0, 0.1, n)
    high = np.maximum(open_, close) + 0.1
    low = np.minimum(open_, close) - 0.1
    df = pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume})
    svc = ReversionService(log_file='test_reversion_service.log')
    result = svc.evaluate_symbol('AAPL', df, account_equity=100000)
    assert 'decision' in result
