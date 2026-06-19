import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from research.leakage import has_lookahead, assert_causal, scan_features


def _df(n=120, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame({"close": close})


# causal features
def f_rolling_mean(df):                       # uses [t-9, t] only
    return df["close"].rolling(10, min_periods=1).mean()

def f_shift1(df):                              # yesterday's close
    return df["close"].shift(1)

def f_expanding_z(df):                          # expanding stats use [0, t]
    c = df["close"]
    return (c - c.expanding().mean()) / c.expanding().std().replace(0, np.nan)


# leaky features
def f_centered(df):                            # centered window peeks ahead
    return df["close"].rolling(11, center=True, min_periods=1).mean()

def f_shift_neg1(df):                           # tomorrow's close = the label
    return df["close"].shift(-1)

def f_fullseries_norm(df):                      # normalized by the WHOLE series' mean
    c = df["close"]
    return c - c.mean()


def test_causal_features_pass():
    df = _df()
    assert has_lookahead(f_rolling_mean, df) is False
    assert has_lookahead(f_shift1, df) is False
    assert has_lookahead(f_expanding_z, df) is False


def test_leaky_features_caught():
    df = _df()
    assert has_lookahead(f_centered, df) is True
    assert has_lookahead(f_shift_neg1, df) is True
    assert has_lookahead(f_fullseries_norm, df) is True


def test_assert_causal_raises_only_on_leak():
    df = _df()
    assert_causal(f_rolling_mean, df, name="rolling_mean")   # no raise
    with pytest.raises(AssertionError):
        assert_causal(f_shift_neg1, df, name="shift(-1)")


def test_scan_features_maps_each():
    df = _df()
    out = scan_features({"ok": f_rolling_mean, "leak": f_centered}, df)
    assert out == {"ok": False, "leak": True}
