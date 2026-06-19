"""Monte-Carlo trade reshuffling: how robust is the equity curve to luck/order?

The iid bootstrap (reversion_bot.analytics.run_bootstrap_monte_carlo) resamples
single trades independently -- but real trades CLUSTER (win/loss streaks), and a
strategy can look fine on average while a bad streak blows the account. Block
reshuffling resamples *runs* of consecutive trades, preserving that streak
structure, and reports the drawdown distribution and the chance the path ends
underwater. Pure numpy.
"""
from __future__ import annotations

import numpy as np


def block_bootstrap_paths(returns, block: int = 5, n_paths: int = 10000, seed: int = 42) -> np.ndarray:
    """Resample ``len(returns)`` trades by drawing overlapping BLOCKS of size
    ``block`` (preserving local order/streaks), for ``n_paths`` paths.

    Returns an ``(n_paths, n)`` matrix of resampled per-trade returns.
    """
    arr = np.asarray(returns, dtype=float)
    n = arr.size
    if n == 0:
        return np.empty((0, 0))
    block = max(1, min(int(block), n))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_paths, n_blocks))  # block start idx
    offs = np.arange(block)
    # (n_paths, n_blocks, block) -> indices into arr, then trim to n columns
    idx = (starts[:, :, None] + offs[None, None, :]).reshape(n_paths, n_blocks * block)[:, :n]
    return arr[idx]


def _max_drawdown_frac(equity_2d: np.ndarray) -> np.ndarray:
    """Max fractional drawdown per row of a compounded-equity matrix."""
    peak = np.maximum.accumulate(equity_2d, axis=1)
    dd = (equity_2d - peak) / peak
    return dd.min(axis=1)


def mc_reshuffle_stats(returns, block: int = 5, n_paths: int = 10000, seed: int = 42) -> dict:
    """Block-reshuffle the trades and summarize the path distribution.

    Reports terminal-return percentiles, the max-drawdown distribution (median /
    95th pct), and the probability the path ends below the starting stake -- the
    'could a bad streak sink it?' question the average can't answer.
    """
    paths = block_bootstrap_paths(returns, block=block, n_paths=n_paths, seed=seed)
    if paths.size == 0:
        return dict(n_trades=0, block=block, n_paths=0)
    eq = np.cumprod(1.0 + paths, axis=1)
    terminal = eq[:, -1] - 1.0                       # total compounded return per path
    mdd = _max_drawdown_frac(eq)                      # most-negative fractional DD per path
    return dict(
        n_trades=int(paths.shape[1]),
        block=int(block),
        n_paths=int(paths.shape[0]),
        terminal_median=float(np.median(terminal)),
        terminal_p5=float(np.percentile(terminal, 5)),
        terminal_p95=float(np.percentile(terminal, 95)),
        prob_negative=float((terminal < 0).mean()),
        maxdd_median=float(np.median(mdd)),
        maxdd_p95=float(np.percentile(mdd, 5)),       # 5th pct = the deep (worst) tail
    )
