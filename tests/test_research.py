import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from research import (
    holdout_split, walk_forward_windows,
    norm_ppf, expected_max_sharpe, deflated_sharpe, passes_overfit_gate,
    robust_across_folds,
)


# --- walk_forward ---------------------------------------------------------

def test_holdout_split_reserves_tail():
    research, hold = holdout_split(100, holdout_frac=0.2)
    assert list(research) == list(range(0, 80))
    assert list(hold) == list(range(80, 100))
    # disjoint + complete
    assert set(research).isdisjoint(set(hold))
    assert len(research) + len(hold) == 100


def test_holdout_split_validates():
    with pytest.raises(ValueError):
        holdout_split(100, holdout_frac=0.0)
    with pytest.raises(ValueError):
        holdout_split(100, holdout_frac=1.0)


def test_walk_forward_no_lookahead_and_rolls():
    folds = walk_forward_windows(n=100, train=40, test=20)  # step defaults to 20
    assert len(folds) == 3   # starts 0,20,40 -> 40+40+20=100 ok at start=40
    for tr, te in folds:
        assert tr.stop == te.start            # test immediately follows train
        assert te.start >= tr.stop            # no look-ahead
    # rolling: train start advances
    assert folds[0][0].start == 0 and folds[1][0].start == 20


def test_walk_forward_anchored_expands_train():
    folds = walk_forward_windows(n=100, train=40, test=20, anchored=True)
    assert all(tr.start == 0 for tr, _ in folds)        # train always from 0
    assert folds[1][0].stop > folds[0][0].stop          # train grows


def test_walk_forward_validates():
    with pytest.raises(ValueError):
        walk_forward_windows(100, 0, 20)
    with pytest.raises(ValueError):
        walk_forward_windows(100, 40, 0)


# --- selection / deflation ------------------------------------------------

def test_norm_ppf_known_values():
    assert abs(norm_ppf(0.5) - 0.0) < 1e-6
    assert abs(norm_ppf(0.975) - 1.959964) < 1e-3     # ~1.96
    assert abs(norm_ppf(0.025) + 1.959964) < 1e-3
    assert norm_ppf(0.999) > norm_ppf(0.99) > 0       # monotone in the tail


def test_expected_max_grows_with_trials_and_shrinks_with_obs():
    # More trials -> a higher "lucky best" bar.
    assert expected_max_sharpe(1000, 250) > expected_max_sharpe(10, 250) > 0
    # More observations -> tighter estimate -> lower bar.
    assert expected_max_sharpe(100, 1000) < expected_max_sharpe(100, 100)
    # A single trial has no multiple-testing inflation.
    assert expected_max_sharpe(1, 250) == 0.0


def test_overfit_gate_rejects_lucky_winner_passes_real_edge():
    n_obs = 250
    bar = expected_max_sharpe(1000, n_obs)
    # A winner barely above the per-obs noise but below the 1000-trial bar -> reject.
    assert passes_overfit_gate(bar * 0.5, 1000, n_obs) is False
    # A winner well above the bar -> accept.
    assert passes_overfit_gate(bar * 2.0, 1000, n_obs) is True
    assert deflated_sharpe(bar * 2.0, 1000, n_obs) > 0


def test_robust_across_folds():
    assert robust_across_folds([0.1, 0.05, -0.02, 0.03, 0.01], min_positive_frac=0.7) is True   # 4/5
    assert robust_across_folds([0.1, -0.1, 0.1, -0.1], min_positive_frac=0.7) is False           # 2/4
    assert robust_across_folds([]) is False
