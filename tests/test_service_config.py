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
    svc.ml_learner.prepare_features = lambda df: (pd.DataFrame({"a": [1.0]}), None)

    class _InfModel:
        def predict_proba(self, X):
            return np.array([[float("nan"), float("inf")]])

    svc.ml_learner.model = _InfModel()
    assert svc._get_ml_probability(pd.DataFrame({"close": [1.0, 2.0, 3.0]})) == 0.5
