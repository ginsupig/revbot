import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from reversion_bot.service import ReversionService
from reversion_bot.config import PerformanceConfig


def _svc(tmp_path, **kwargs):
    # Isolate performance state per-test so log writes don't collide.
    return ReversionService(
        performance_config=PerformanceConfig(state_dir=str(tmp_path)),
        **kwargs,
    )


def test_min_trade_score_defaults_to_036(tmp_path):
    assert _svc(tmp_path).min_trade_score == 0.36


def test_min_trade_score_override_is_respected(tmp_path):
    # The value main.py reads from MIN_TRADE_SCORE must flow through and become
    # both the routing floor and the (pre-adaptive) trade threshold.
    assert _svc(tmp_path, min_trade_score=0.40).min_trade_score == 0.40


def test_weighted_score_neutralizes_non_finite(tmp_path):
    # An inf/nan component (e.g. a blown-up ML probability) must not produce an
    # inf/nan trade score that would slip past the entry threshold.
    svc = _svc(tmp_path)
    for bad in (float("inf"), float("nan"), float("-inf")):
        s = svc._weighted_score(
            {"mean_reversion": 0.5, "ml": bad, "trendfail": 0.45, "trend_following": 0.3}
        )
        assert math.isfinite(s)
        # Bad component counts as 0, so the score equals the sum of the rest.
        expected = 0.30 * 0.5 + 0.25 * 0.0 + 0.15 * 0.45 + 0.30 * 0.3
        assert abs(s - expected) < 1e-9


def test_ml_probability_clamps_non_finite(tmp_path):
    # If the model returns a non-finite probability, fall back to neutral 0.5.
    svc = _svc(tmp_path)
    svc.min_rows_for_ml = 0
    svc.ml_is_fitted = True   # reach the predict path (skip the unfit short-circuit)
    svc.ml_learner.prepare_features = lambda df: (pd.DataFrame({"a": [1.0]}), None)
    svc.ml_learner.prepare_features_for_predict = lambda df: pd.DataFrame({"a": [1.0]})

    class _InfModel:
        def predict_proba(self, X):
            return np.array([[float("nan"), float("inf")]])

    svc.ml_learner.model = _InfModel()
    assert svc._get_ml_probability(pd.DataFrame({"close": [1.0, 2.0, 3.0]})) == 0.5


# --- cold-start ML: never-fitted model must not raise/spam (regression) ------

def _two_class(n=60):
    X = pd.DataFrame({"a": list(range(n))})
    y = pd.Series([0, 1] * (n // 2))
    return X, y


def test_unfit_model_returns_neutral_without_predicting(tmp_path):
    # One-class labels skip the fit, so the model stays unfit. _get_ml_probability
    # must return neutral 0.5 WITHOUT calling predict_proba (which would raise
    # "not fitted" and spam the log for every symbol).
    svc = _svc(tmp_path)
    svc.min_rows_for_ml = 0
    svc.ml_learner.prepare_features = lambda df: (pd.DataFrame({"a": [1.0] * 60}), pd.Series([1] * 60))

    class _Boom:
        def fit(self, X, y):
            raise AssertionError("one-class data must not reach fit")  # nunique<2 guards it
        def predict_proba(self, X):
            raise AssertionError("predict_proba must not run on an unfit model")

    svc.ml_learner.model = _Boom()
    assert svc._get_ml_probability(pd.DataFrame({"close": [1.0, 2.0, 3.0]})) == 0.5
    assert svc.ml_is_fitted is False


def test_unfit_model_retries_fit_off_the_25_tick(tmp_path):
    # The bug: training was only attempted on eval #1 / every 25th. A cold start
    # in a one-directional tape could go many evals unfit. While unfit, a fit must
    # be attempted on EVERY eval — here on eval #8 (neither 1 nor a multiple of 25).
    svc = _svc(tmp_path)
    svc.min_rows_for_ml = 0
    svc.ml_eval_count = 7
    X, y = _two_class()
    svc.ml_learner.prepare_features = lambda df: (X, y)
    svc.ml_learner.prepare_features_for_predict = lambda df: X

    class _Model:
        def fit(self, X, y):
            pass
        def predict_proba(self, X):
            return np.array([[0.3, 0.7]])

    svc.ml_learner.model = _Model()
    p = svc._get_ml_probability(pd.DataFrame({"close": [1.0, 2.0, 3.0]}))
    assert svc.ml_is_fitted is True   # fit happened off-tick because the model was unfit
    assert p == 0.7


def test_short_bias_threads_through_service(tmp_path):
    # With unreachable normal short thresholds, a short only fires when the
    # service passes short_bias=True down to the engine.
    from reversion_bot.config import ReversionConfig, RiskConfig
    from test_engine import make_overbought_df

    cfg = ReversionConfig(
        min_history=60, enable_shorts=True, use_trend_filter=False,  # isolate short_bias threading
        rsi_min=99.0, ri_short_threshold=9.0,
        risk_off_rsi_min=30.0, risk_off_ri_short_threshold=0.0,
    )
    svc = ReversionService(cfg, RiskConfig(), PerformanceConfig(state_dir=str(tmp_path)))
    df = make_overbought_df()
    assert svc.evaluate_symbol("T", df, 100000, short_bias=False).get("go_short") is not True
    assert svc.evaluate_symbol("T", df, 100000, short_bias=True).get("go_short") is True


# --- ML model persistence across restarts -----------------------------------

import os
import time
from reversion_bot.config import PerformanceConfig as _PC


def _fit_twoclass(svc):
    """Drive one successful fit on a tiny picklable model."""
    from sklearn.dummy import DummyClassifier
    X = pd.DataFrame({"a": list(range(60))})
    y = pd.Series([0, 1] * 30)
    svc.min_rows_for_ml = 0
    svc.ml_learner.prepare_features = lambda df: (X, y)
    svc.ml_learner.prepare_features_for_predict = lambda df: X
    svc.ml_learner.model = DummyClassifier(strategy="prior")
    return svc._get_ml_probability(pd.DataFrame({"close": [1.0, 2.0, 3.0]}))


def test_fitted_model_is_saved_then_reloaded(tmp_path):
    svc = _svc(tmp_path)
    _fit_twoclass(svc)
    assert svc.ml_is_fitted is True
    assert os.path.exists(svc._ml_model_path)

    # A fresh service (same state dir) loads it -> fitted from startup.
    svc2 = _svc(tmp_path)
    assert svc2.ml_is_fitted is True
    assert svc2._ml_from_disk is True


def test_missing_model_file_cold_starts(tmp_path):
    svc = _svc(tmp_path)
    assert svc.ml_is_fitted is False
    assert svc._ml_from_disk is False


def test_stale_model_is_ignored(tmp_path):
    import joblib
    from sklearn.dummy import DummyClassifier
    path = tmp_path / "ml_model.pkl"
    joblib.dump(DummyClassifier(strategy="prior").fit([[0], [1]], [0, 1]), path)
    old = time.time() - 10 * 3600
    os.utime(path, (old, old))
    cfg = _PC(state_dir=str(tmp_path), ml_model_max_age_hours=1.0)
    svc = ReversionService(performance_config=cfg)
    assert svc.ml_is_fitted is False   # too old -> cold-start


def test_persist_disabled_neither_loads_nor_saves(tmp_path):
    cfg = _PC(state_dir=str(tmp_path), persist_ml_model=False)
    svc = ReversionService(performance_config=cfg)
    _fit_twoclass(svc)
    assert svc.ml_is_fitted is True
    assert not os.path.exists(svc._ml_model_path)   # save skipped


class _BrokenModel:
    """Picklable model that errors on predict — simulates a stale schema."""
    def predict_proba(self, X):
        raise RuntimeError("feature schema mismatch")


def test_loaded_model_invalidated_when_predict_fails(tmp_path):
    import joblib
    path = tmp_path / "ml_model.pkl"
    joblib.dump(_BrokenModel(), path)
    svc = _svc(tmp_path)
    assert svc.ml_is_fitted is True and svc._ml_from_disk is True  # loaded
    # A predict that raises must invalidate the loaded model so retrain kicks in.
    svc.min_rows_for_ml = 0
    svc.ml_learner.prepare_features = lambda df: (pd.DataFrame({"a": [1.0]}), pd.Series([0]))
    svc.ml_learner.prepare_features_for_predict = lambda df: pd.DataFrame({"a": [1.0]})
    assert svc._get_ml_probability(pd.DataFrame({"close": [1.0, 2.0, 3.0]})) == 0.5
    assert svc.ml_is_fitted is False
    assert svc._ml_from_disk is False
