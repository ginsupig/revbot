"""Multiple-testing deflation: the bar a best-of-N sweep must clear.

VectorBT can test thousands of configs in seconds. The catch: the *maximum*
Sharpe over N zero-edge strategies is large purely by chance, so the winner of a
big sweep is almost always partly luck. This deflates for that:

  expected_max_sharpe(N, n_obs)  -- the best Sharpe you'd EXPECT from N random
                                    strategies over n_obs observations (Bailey &
                                    Lopez de Prado's E[max]).
  deflated_sharpe(best, N, n_obs)-- how far the observed best clears that bar.
  passes_overfit_gate(...)       -- bool: did it clear it?

A config is only "real" if it beats the deflated bar AND persists across
walk-forward folds AND holds on the untouched holdout. Pure stdlib.
"""
from __future__ import annotations

import math
from typing import Sequence

_EULER = 0.5772156649015329


def norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation, ~1e-9)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def sharpe_standard_error(n_obs: int) -> float:
    """Approx SE of an estimated (per-obs) Sharpe under iid normality ~ 1/sqrt(n)."""
    if n_obs <= 1:
        return float("inf")
    return 1.0 / math.sqrt(n_obs)


def expected_max_sharpe(n_trials: int, n_obs: int) -> float:
    """Expected MAX (per-obs) Sharpe from ``n_trials`` zero-edge strategies.

    Bailey & Lopez de Prado: E[max of N std normals] ≈
        (1-γ)·Φ⁻¹(1 - 1/N) + γ·Φ⁻¹(1 - 1/(N·e)),  scaled by the Sharpe SE.
    This is the multiple-testing bar the best of a sweep must beat.
    """
    if n_trials <= 1:
        return 0.0
    z = (1 - _EULER) * norm_ppf(1 - 1.0 / n_trials) + \
        _EULER * norm_ppf(1 - 1.0 / (n_trials * math.e))
    return z * sharpe_standard_error(n_obs)


def deflated_sharpe(observed_sharpe: float, n_trials: int, n_obs: int) -> float:
    """Observed best Sharpe minus the expected-max bar (in per-obs Sharpe units).
    > 0 means the winner clears what pure luck over n_trials would produce."""
    return observed_sharpe - expected_max_sharpe(n_trials, n_obs)


def passes_overfit_gate(observed_sharpe: float, n_trials: int, n_obs: int) -> bool:
    """True iff the sweep winner beats the multiple-testing bar."""
    return deflated_sharpe(observed_sharpe, n_trials, n_obs) > 0.0


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _sharpe_and_moments(returns: Sequence[float]):
    """Per-obs Sharpe + sample skewness + kurtosis (n divisor) of a return series."""
    xs = [float(x) for x in returns]
    n = len(xs)
    if n < 2:
        return 0.0, 0.0, 3.0, n
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / n
    sd = var ** 0.5
    if sd <= 0:
        return 0.0, 0.0, 3.0, n
    sr = m / sd
    skew = (sum((x - m) ** 3 for x in xs) / n) / sd ** 3
    kurt = (sum((x - m) ** 4 for x in xs) / n) / sd ** 4
    return sr, skew, kurt, n


def deflated_sharpe_ratio(returns: Sequence[float], n_trials: int) -> float:
    """Full Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014) — the *probability*
    that the observed per-obs Sharpe genuinely exceeds the expected-max bar of
    ``n_trials`` zero-edge strategies, with the Sharpe estimator's standard error
    ADJUSTED for the return series' skewness (γ3) and kurtosis (γ4):

        SE(SR) = sqrt( (1 - γ3·SR + ((γ4-1)/4)·SR²) / (n-1) )
        DSR    = Φ( (SR - SR*) / SE )

    This is statistically superior to the plain normal-approximation deflation for
    real trading returns, which are fat-tailed (high γ4 widens SE -> a HIGHER bar)
    and skewed (a trailing-stop book is positively skewed -> a small offset). Returns
    a probability in [0, 1]; the conventional pass is > 0.95. Research-gating only —
    does NOT touch the live signal path, so it can't invalidate a backtest."""
    sr, skew, kurt, n = _sharpe_and_moments(returns)
    if n < 2:
        return 0.0
    sr_star = expected_max_sharpe(n_trials, n)
    var = max(1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr, 1e-12) / (n - 1)
    se = math.sqrt(var)
    return _norm_cdf((sr - sr_star) / se) if se > 0 else 0.0


def passes_dsr_gate(returns: Sequence[float], n_trials: int, confidence: float = 0.95) -> bool:
    """True iff the moment-adjusted Deflated Sharpe Ratio clears ``confidence``.
    The more rigorous sibling of ``passes_overfit_gate`` for fat-tailed returns."""
    return deflated_sharpe_ratio(returns, n_trials) > confidence


def robust_across_folds(fold_metrics: Sequence[float], min_positive_frac: float = 0.7) -> bool:
    """True iff the metric is positive in at least ``min_positive_frac`` of folds.

    Parameter-decay / regime guard: an edge that only shows in a couple of
    walk-forward folds isn't robust. Empty -> False.
    """
    vals = list(fold_metrics)
    if not vals:
        return False
    pos = sum(1 for v in vals if v > 0)
    return (pos / len(vals)) >= min_positive_frac
