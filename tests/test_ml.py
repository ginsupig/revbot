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
    # predict() scores through the PREDICT featurizer, which keeps the current
    # bar (the one being decided on). The training featurizer drops it because
    # its forward label is NaN, so scoring through that path silently returned
    # the previous bar's prediction for anything reading preds[-1].
    X_predict = learner.prepare_features_for_predict(df)
    X_train, _ = learner.prepare_features(df)
    preds = learner.predict(df)
    assert len(preds) == len(X_predict)
    assert len(X_predict) == len(X_train) + 1   # the current bar is included
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


# --- Probability calibration -------------------------------------------------

def test_default_model_is_calibrated_and_optional():
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    assert isinstance(MLSignalLearner().model, CalibratedClassifierCV)
    assert isinstance(MLSignalLearner(calibration=None).model, RandomForestClassifier)


def test_calibrated_probabilities_are_valid():
    # Fit on a 2-class set and confirm predict_proba returns real probabilities
    # in [0, 1] (the value that feeds the 25%-weight blend).
    df = _ramp(200)
    learner = MLSignalLearner()           # calibrated (sigmoid, cv=3)
    learner.fit(df)
    proba = learner.predict_proba(df)
    assert proba.shape[1] == 2
    assert float(proba.min()) >= 0.0 and float(proba.max()) <= 1.0
    # Rows sum to 1 (proper probability distribution).
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
