"""Walk-forward splits + a held-out window. Pure index math, no data deps.

Walk-forward is how we reject parameter decay: optimize on each train slice,
score on the *next* (unseen) test slice, and require the edge to persist across
folds — not just in one window. The held-out tail is sacred: the sweep never
sees it, so it's a clean final check on the chosen config (the antidote to
"tuned until it worked").
"""
from __future__ import annotations

from typing import List, Tuple


def holdout_split(n: int, holdout_frac: float = 0.2) -> Tuple[range, range]:
    """Reserve the final ``holdout_frac`` of a length-``n`` series as a holdout.

    Returns ``(research_idx, holdout_idx)``. The sweep/optimizer may only touch
    research_idx; holdout_idx is scored exactly once, at the end, on the single
    config chosen — so it can't leak into the search.
    """
    if not 0.0 < holdout_frac < 1.0:
        raise ValueError("holdout_frac must be in (0, 1)")
    if n <= 0:
        return range(0, 0), range(0, 0)
    cut = int(round(n * (1.0 - holdout_frac)))
    cut = min(max(cut, 1), n - 1) if n >= 2 else n
    return range(0, cut), range(cut, n)


def walk_forward_windows(
    n: int, train: int, test: int, step: int | None = None, anchored: bool = False
) -> List[Tuple[range, range]]:
    """Generate ``(train_range, test_range)`` folds over ``[0, n)``.

    Each test slice is strictly *after* its train slice (no look-ahead). ``step``
    defaults to ``test`` (non-overlapping test slices = a true out-of-sample
    stitch). ``anchored=True`` grows the train window from 0 (expanding); False
    rolls it (fixed length) so parameter *decay* shows up.
    """
    if train <= 0 or test <= 0:
        raise ValueError("train and test must be > 0")
    step = test if step is None else step
    if step <= 0:
        raise ValueError("step must be > 0")
    folds: List[Tuple[range, range]] = []
    start = 0
    while start + train + test <= n:
        tr_start = 0 if anchored else start
        tr_end = start + train
        te_end = tr_end + test
        folds.append((range(tr_start, tr_end), range(tr_end, te_end)))
        start += step
    return folds
