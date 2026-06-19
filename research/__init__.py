"""Research pipeline: the discipline layer that wraps any fast-sweep engine.

Stages (build order):
  1. walk_forward  -- rolling/anchored splits + a held-out window never touched
                      until the end. (this module)
  2. selection     -- multiple-testing deflation: the bar the best-of-N sweep
                      must clear so a thousand VectorBT combos can't manufacture
                      a mirage. (this module)
  3. mc_reshuffle  -- bootstrap/block-reshuffle trade returns for tail/CI (uses
                      reversion_bot.analytics.run_bootstrap_monte_carlo).
  4. regime        -- tag windows by regime (reversion_bot.market_regime).
  5. leakage       -- no-lookahead / target-leakage assertions on features.
  6. engines       -- adapters: VectorBT (fast stateless signal sweeps) and
                      Backtrader (realistic stateful fills). Run on a machine
                      with those deps + market data.

Framework-agnostic by design: VectorBT/Backtrader are swappable backends; the
gates here are what keep the search honest (see CLAUDE.md).
"""
from .walk_forward import holdout_split, walk_forward_windows
from .leakage import has_lookahead, assert_causal, scan_features
from .regime import label_regime, per_regime_stats
from .cost import apply_cost, cost_sweep, breakeven_cost_bps
from .pipeline import evaluate_sweep, sharpe, PipelineResult
from .mc_reshuffle import block_bootstrap_paths, mc_reshuffle_stats
from .selection import (
    norm_ppf,
    expected_max_sharpe,
    deflated_sharpe,
    passes_overfit_gate,
    robust_across_folds,
)

__all__ = [
    "holdout_split", "walk_forward_windows",
    "norm_ppf", "expected_max_sharpe", "deflated_sharpe",
    "passes_overfit_gate", "robust_across_folds",
    "block_bootstrap_paths", "mc_reshuffle_stats",
    "has_lookahead", "assert_causal", "scan_features",
    "label_regime", "per_regime_stats",
    "apply_cost", "cost_sweep", "breakeven_cost_bps",
    "evaluate_sweep", "sharpe", "PipelineResult",
]
