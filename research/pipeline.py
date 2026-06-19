"""The orchestrator: run a parameter sweep through every discipline gate.

A sweep engine (VectorBT, or the numpy reference) produces per-trade returns for
each config on the research window, and on an untouched holdout. This decides
whether any config is a real, deployable edge or sweep luck:

  1. walk-forward robustness  -- positive in most folds (parameter-decay guard)
  2. deflated-Sharpe gate      -- best-of-N beats the multiple-testing bar
  3. MC reshuffle + cost       -- survives streak risk and realistic friction
  4. holdout                   -- holds on data the sweep never saw

Returns a verdict and the evidence. Engine-agnostic: it only sees return arrays.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .selection import deflated_sharpe, robust_across_folds
from .mc_reshuffle import mc_reshuffle_stats
from .cost import cost_sweep, breakeven_cost_bps


def sharpe(returns) -> float:
    """Standardized per-trade Sharpe (mean/std) — matches the deflation's SE~1/sqrt(n)."""
    arr = np.asarray(returns, dtype=float)
    if arr.size < 2:
        return 0.0
    sd = arr.std(ddof=0)
    return float(arr.mean() / sd) if sd > 0 else 0.0


def _fold_sharpes(returns, n_folds: int) -> list:
    arr = np.asarray(returns, dtype=float)
    if arr.size < n_folds:
        return [sharpe(arr)]
    return [sharpe(c) for c in np.array_split(arr, n_folds)]


@dataclass
class PipelineResult:
    chosen: Optional[str]
    n_trials: int
    research_sharpe: float
    deflated_sharpe: float
    passed_gate: bool
    robust: bool
    fold_sharpes: list = field(default_factory=list)
    holdout: dict = field(default_factory=dict)
    mc: dict = field(default_factory=dict)
    cost: dict = field(default_factory=dict)
    breakeven_bps: float = 0.0
    verdict: str = ""


def _holdout_stats(returns) -> dict:
    arr = np.asarray(returns, dtype=float)
    if arr.size == 0:
        return dict(n=0, mean=0.0, sharpe=0.0)
    w, l = arr[arr > 0].sum(), -arr[arr < 0].sum()
    pf = float(w / l) if l > 0 else (10.0 if w > 0 else 0.0)
    return dict(n=int(arr.size), mean=float(arr.mean()), sharpe=sharpe(arr), pf=pf)


def evaluate_sweep(
    config_returns: Dict[str, "np.ndarray"],
    holdout_returns: Dict[str, "np.ndarray"],
    n_folds: int = 5,
    fold_min_positive: float = 0.6,
    costs=(0, 2, 4, 6, 8),
) -> PipelineResult:
    """Gate a sweep. ``config_returns``/``holdout_returns`` map config -> per-trade
    returns on the research / holdout windows. Picks the best *robust* config,
    deflates for the number of trials, and confirms it on the holdout."""
    n_trials = len(config_returns)
    if n_trials == 0:
        return PipelineResult(None, 0, 0.0, 0.0, False, False, verdict="empty sweep")

    # 1) walk-forward robustness, then rank robust configs by research Sharpe.
    scored = []
    for cfg, rets in config_returns.items():
        folds = _fold_sharpes(rets, n_folds)
        robust = robust_across_folds(folds, fold_min_positive)
        scored.append((cfg, sharpe(rets), robust, folds, np.asarray(rets, float).size))
    robust_cfgs = [s for s in scored if s[2]]
    pool = robust_cfgs or scored                       # fall back to report the best even if none robust
    best = max(pool, key=lambda s: s[1])
    cfg, sh, robust, folds, n_obs = best

    # 2) deflated-Sharpe gate (best-of-n_trials must clear the luck bar).
    dsr = deflated_sharpe(sh, n_trials, n_obs)
    passed = dsr > 0.0

    # 3) MC reshuffle + cost on the chosen config.
    chosen_rets = config_returns[cfg]
    mc = mc_reshuffle_stats(chosen_rets, n_paths=5000)
    cs = cost_sweep(chosen_rets, costs)
    be = breakeven_cost_bps(chosen_rets)

    # 4) holdout — the untouched final check.
    ho = _holdout_stats(holdout_returns.get(cfg, []))

    deploy = bool(passed and robust and ho.get("mean", 0.0) > 0 and ho.get("n", 0) > 0)
    if deploy:
        verdict = "DEPLOY-CANDIDATE: clears deflation, robust across folds, holds on holdout"
    elif not robust:
        verdict = "REJECT: best config not robust across walk-forward folds"
    elif not passed:
        verdict = "REJECT: best Sharpe within multiple-testing noise (deflated <= 0)"
    elif ho.get("mean", 0.0) <= 0:
        verdict = "REJECT: failed on the untouched holdout"
    else:
        verdict = "REJECT"

    return PipelineResult(
        chosen=cfg, n_trials=n_trials, research_sharpe=sh, deflated_sharpe=dsr,
        passed_gate=passed, robust=robust, fold_sharpes=folds, holdout=ho, mc=mc,
        cost=cs, breakeven_bps=be, verdict=verdict,
    )
