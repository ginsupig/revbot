import numpy as np
import pandas as pd

from reversion_bot.walkforward import evaluate_fixed_params, mean_reversion_strategy


def _synth(n=900, seed=3):
    rng = np.random.default_rng(seed)
    close = np.cumsum(rng.standard_normal(n)) * 0.5 + 100
    o = close + rng.normal(0, 0.05, n)
    h = np.maximum(o, close) + 0.1
    l = np.minimum(o, close) - 0.1
    v = rng.integers(1000, 5000, n)
    idx = pd.date_range("2026-04-01 14:30", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": close, "volume": v}, index=idx)


PARAMS = dict(band_length=20, ri_threshold=-1.0, rsi_max=40.0,
              use_vwap_filter=True, max_vwap_extension_pct=0.012, min_history=160)


def test_returns_one_metric_per_fold():
    m = evaluate_fixed_params(_synth(), PARAMS, n_splits=3)
    assert len(m) == 3
    assert set(m[0]) == {"fold", "profit_factor", "sharpe", "max_drawdown"}


def test_metrics_are_finite_floats():
    m = evaluate_fixed_params(_synth(), PARAMS, n_splits=3)
    for x in m:
        assert isinstance(x["profit_factor"], float)
        assert np.isfinite(x["profit_factor"])
        assert np.isfinite(x["sharpe"])


def test_matches_direct_single_fold_run():
    # evaluate_fixed_params on the last fold == calling the strategy directly
    df = _synth()
    from sklearn.model_selection import TimeSeriesSplit
    from reversion_bot.analytics import profit_factor
    splits = list(TimeSeriesSplit(n_splits=3).split(df))
    _, test_idx = splits[-1]
    direct = profit_factor(mean_reversion_strategy(df.iloc[test_idx], **PARAMS))
    via = evaluate_fixed_params(df, PARAMS, n_splits=3)[-1]["profit_factor"]
    assert via == direct
