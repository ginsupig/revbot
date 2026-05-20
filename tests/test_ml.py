import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from reversion_bot.ml import MLSignalLearner

def test_ml_signal_learner_fit_predict():
    # Create dummy data
    n = 100
    np.random.seed(42)
    close = np.cumsum(np.random.randn(n)) + 100
    volume = np.random.randint(1000, 2000, n)
    open_ = close + np.random.normal(0, 0.1, n)
    high = np.maximum(open_, close) + 0.1
    low = np.minimum(open_, close) - 0.1
    df = pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume})
    learner = MLSignalLearner()
    learner.fit(df)
    # Prepare features to get the number of valid rows after dropna
    X, _ = learner.prepare_features(df)
    preds = learner.predict(df)
    assert len(preds) == len(X)
    # Check that predictions are 0 or 1
    assert set(preds).issubset({0, 1})
