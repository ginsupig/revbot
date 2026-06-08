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


def _ramp(n=120):
    idx = pd.date_range("2026-01-01", periods=n, freq="5min")
    close = 100 + np.cumsum(np.random.default_rng(0).normal(0, 0.2, n))
    return pd.DataFrame(
        {"open": close, "high": close + 0.2, "low": close - 0.2, "close": close,
         "volume": np.random.default_rng(1).integers(100000, 200000, n)},
        index=idx,
    )


def test_predict_features_keep_the_current_bar():
    # The bug: prepare_features() drops the last bar (its forward label is NaN),
    # so live scoring used the PRIOR bar. The predict path must keep the current
    # bar so the ML signal is for the decision bar, not a stale one.
    df = _ramp()
    learner = MLSignalLearner()
    X_pred = learner.prepare_features_for_predict(df)
    X_train, _ = learner.prepare_features(df)
    assert X_pred.index[-1] == df.index[-1]      # current bar scored
    assert X_train.index[-1] == df.index[-2]      # training still drops the unlabeled bar
    assert len(X_pred) == len(X_train) + 1
