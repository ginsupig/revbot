import numpy as np
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit

class AutoTuner:
    def __init__(self, strategy_func, data, param_grid):
        self.strategy_func = strategy_func
        self.data = data
        self.param_grid = param_grid
        self.best_params = None
        self.best_score = -np.inf

    def tune(self, score_func, n_splits=5):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        param_list = list(ParameterGrid(self.param_grid))
        total = len(param_list)
        for idx, params in enumerate(param_list):
            print(f"Tuning {idx+1}/{total}: {params}")
            scores = []
            for train_idx, test_idx in tscv.split(self.data):
                # Evaluate strategy on OUT-OF-SAMPLE test data, not training data.
                # This ensures we pick parameters based on generalization, not overfitting.
                test = self.data.iloc[test_idx]
                results = self.strategy_func(test, **params)
                score = score_func(results)
                scores.append(score)
            mean_score = np.mean(scores)
            if mean_score > self.best_score:
                self.best_score = mean_score
                self.best_params = params
        return self.best_params, self.best_score
