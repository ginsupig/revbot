import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from research.mc_reshuffle import block_bootstrap_paths, mc_reshuffle_stats


def test_block_paths_shape_and_membership():
    rets = np.array([0.01, -0.02, 0.03, -0.01, 0.02, -0.015, 0.005, 0.04])
    paths = block_bootstrap_paths(rets, block=3, n_paths=500, seed=1)
    assert paths.shape == (500, rets.size)            # trimmed to n columns
    # every resampled value comes from the original set
    assert set(np.unique(paths)).issubset(set(rets.tolist()))


def test_block_preserves_adjacency():
    # With a strictly increasing series, block=4 draws must contain consecutive runs.
    rets = np.arange(20, dtype=float) / 100.0
    paths = block_bootstrap_paths(rets, block=4, n_paths=200, seed=2)
    diffs = np.diff(paths, axis=1)
    # at least some adjacent pairs are consecutive in the original (+0.01 step)
    assert np.isclose(diffs, 0.01).mean() > 0.4


def test_mc_stats_winning_vs_losing():
    win = mc_reshuffle_stats([0.02, 0.01, 0.03, 0.015, 0.005] * 10, n_paths=2000, seed=3)
    assert win["prob_negative"] < 0.05            # all-positive trades rarely end down
    assert win["terminal_median"] > 0
    lose = mc_reshuffle_stats([-0.02, -0.01, -0.03, -0.015] * 10, n_paths=2000, seed=3)
    assert lose["prob_negative"] > 0.95
    assert lose["maxdd_p95"] < 0                  # deep-tail drawdown is negative


def test_mc_stats_empty():
    assert mc_reshuffle_stats([])["n_paths"] == 0
