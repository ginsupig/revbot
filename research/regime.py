"""Regime tagging: does an edge survive regime CHANGES, or live in just one?

Recurring finding (CLAUDE.md): regime dominates signal. A strategy can post a
great blended number while actually being long one regime and bleeding in the
other. This labels each bar bull/bear/range from a benchmark's trend+ADX, then
breaks per-trade returns down by regime so a one-regime mirage can't hide.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from reversion_bot.market_regime import classify_regime_series


def label_regime(bench_bars: pd.DataFrame, adx_min: float = 20.0, **kw) -> pd.Series:
    """Per-bar regime: 'downtrend' / 'uptrend' / 'range' from a benchmark's
    trend (close vs EMA) and ADX. Trending = ADX high; direction from the trend.
    Warm-up bars label 'range' (the safe default)."""
    reg = classify_regime_series(bench_bars, adx_min=adx_min, **kw)
    if len(reg) == 0:
        return pd.Series([], dtype=object)
    out = pd.Series("range", index=reg.index, dtype=object)
    trending = reg["adx_high"].to_numpy()
    down = reg["trend_down"].to_numpy()
    out[trending & down] = "downtrend"
    out[trending & ~down] = "uptrend"
    return out


def _pf(arr: np.ndarray) -> float:
    w, l = arr[arr > 0].sum(), -arr[arr < 0].sum()
    return float(w / l) if l > 0 else (10.0 if w > 0 else 0.0)


def per_regime_stats(returns, labels) -> dict:
    """``{regime: {n, mean, pf, win}}`` for trade returns tagged by regime.

    The edge "survives regime change" only if it's positive in more than one
    regime — a config that's all from 'uptrend' is a beta bet, not an edge."""
    r = np.asarray(returns, dtype=float)
    lab = np.asarray(labels, dtype=object)
    out = {}
    for g in np.unique(lab):
        a = r[lab == g]
        if a.size == 0:
            continue
        out[str(g)] = dict(n=int(a.size), mean=float(a.mean()), pf=_pf(a),
                           win=float((a > 0).mean()))
    return out
