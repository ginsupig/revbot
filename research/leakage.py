"""Feature-leakage guard: prove a feature is causal (no look-ahead).

The deadliest backtest bug: a feature at time t that secretly uses data from
t+1.. -- a centered rolling window, a .shift(-1), full-series normalization, a
resample that peeks ahead. It inflates in-sample results and vanishes live. The
only sure test is operational: recompute the feature on truncated history and
check the value at t is INVARIANT to what comes after t. Pure pandas/numpy.
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import pandas as pd


def has_lookahead(feature_fn: Callable[[pd.DataFrame], "pd.Series"], df: pd.DataFrame,
                  probes: int = 25, atol: float = 1e-8, seed: int = 0) -> bool:
    """True if ``feature_fn`` leaks the future.

    For random interior indices t, compares the feature at t computed on the FULL
    frame vs on the prefix ``df.iloc[:t+1]``. If any differ, the value at t
    depends on bars after t -> look-ahead. ``feature_fn(df)`` must return a Series
    aligned to ``df`` (same length/order).
    """
    full = np.asarray(feature_fn(df), dtype=float)
    n = len(df)
    if n < 4:
        return False
    rng = np.random.default_rng(seed)
    lo = max(2, n // 3)                                   # skip warm-up region
    candidates = np.arange(lo, n - 1)
    if candidates.size == 0:
        return False
    probe = rng.choice(candidates, size=int(min(probes, candidates.size)), replace=False)
    for t in probe:
        trunc = np.asarray(feature_fn(df.iloc[: t + 1]), dtype=float)
        if trunc.size == 0:
            continue
        v_full, v_trunc = full[t], trunc[-1]
        if np.isnan(v_full) and np.isnan(v_trunc):
            continue
        if np.isnan(v_full) != np.isnan(v_trunc):
            return True
        if not np.isclose(v_full, v_trunc, atol=atol, rtol=1e-6):
            return True
    return False


def assert_causal(feature_fn, df, name: str = "feature", **kw) -> None:
    """Raise AssertionError if ``feature_fn`` has look-ahead."""
    if has_lookahead(feature_fn, df, **kw):
        raise AssertionError(
            f"{name}: look-ahead detected — its value at t changes when future "
            f"bars are added (centered window / shift(-1) / full-series stat?)."
        )


def scan_features(feature_fns: Dict[str, Callable[[pd.DataFrame], "pd.Series"]],
                  df: pd.DataFrame, **kw) -> Dict[str, bool]:
    """Map ``{name: feature_fn}`` -> ``{name: leaks?}``. Run before any sweep so a
    leaky feature can't inflate every config in the grid."""
    return {name: has_lookahead(fn, df, **kw) for name, fn in feature_fns.items()}
