import numpy as np
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit


class AutoTuner:
    """Grid-search parameter SELECTION over the data it is given.

    IMPORTANT: the score `tune` returns is a SELECTION score computed on
    `self.data` — it is NOT an out-of-sample estimate, because the argmax is
    taken over these very windows. For honest OOS numbers, construct the tuner
    on a TRAIN window only and evaluate the returned params on data this tuner
    never saw (see run_walkforward_backtest, which does exactly that per fold).
    The previous protocol scored every param set on the walkforward's own test
    folds and then reported those folds as "OOS" — lookahead by construction:
    a 200-combination grid will routinely print PF > 1 "OOS" with zero real
    edge that way.
    """

    def __init__(self, strategy_func, data, param_grid):
        self.strategy_func = strategy_func
        self.data = data
        self.param_grid = param_grid
        self.best_params = None
        self.best_score = -np.inf

    def tune(self, score_func, n_splits=5, verbose=True):
        # Average each param set's score across time-ordered sub-windows of
        # the PROVIDED data: rewards stability across sub-periods rather than
        # one lucky stretch. Still in-sample for selection (see class doc).
        # Clamp so each window keeps >= 2 rows (a 1-row window scores every
        # param set 0 and degenerates selection to grid order).
        n_splits = max(2, min(int(n_splits), len(self.data) // 2 - 1, len(self.data) - 1))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        param_list = list(ParameterGrid(self.param_grid))
        total = len(param_list)
        for idx, params in enumerate(param_list):
            if verbose:
                print(f"Tuning {idx+1}/{total}: {params}")
            scores = []
            for _train_idx, test_idx in tscv.split(self.data):
                window = self.data.iloc[test_idx]
                results = self.strategy_func(window, **params)
                scores.append(score_func(results))
            mean_score = np.mean(scores)
            if mean_score > self.best_score:
                self.best_score = mean_score
                self.best_params = params
        return self.best_params, self.best_score
