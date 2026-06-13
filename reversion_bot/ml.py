import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier


FEATURE_COLS = [
    "fast_sma_ratio",
    "slow_sma_ratio",
    "sma_diff_ratio",
    "volatility_pct",
    "rsi",
    "macd_line_ratio",
    "macd_signal_ratio",
    "macd_hist_ratio",
    "momentum_pct",
    "trend_following",
    "trend_ema_ratio",
    "volume_ratio",
    "high_low_pct",
    "close_open_pct",
    "price_position",
]


class MLSignalLearner:
    def __init__(self, calibration: str | None = "sigmoid", calibration_cv: int = 3):
        # A bare RandomForest's predict_proba is notoriously miscalibrated —
        # tree-vote fractions, not true probabilities — yet that number is 25%
        # of every entry score. Wrap it in CalibratedClassifierCV so the value
        # fed to the blend is an actual calibrated probability. "sigmoid" (Platt)
        # is robust on the few-hundred-row windows the bot retrains on; pass
        # calibration=None for the raw forest (e.g. an A/B comparison).
        base = RandomForestClassifier(n_estimators=100, random_state=42)
        if calibration:
            # Calibrate on CHRONOLOGICAL folds, not sklearn's default stratified
            # K-fold. With shuffled folds, temporally adjacent bars land in
            # opposite folds and rolling-window features let the forest score
            # its own neighborhood. Measured on driftless noise (true AUC 0.50):
            # negligible under today's 1-bar label, but a 5-bar overlapping
            # label inflates shuffled-fold AUC to ~0.65 while TimeSeriesSplit
            # stays ~0.51 — so this guards the calibration against any future
            # label-horizon change at zero cost.
            from sklearn.model_selection import TimeSeriesSplit
            self.model = CalibratedClassifierCV(
                base, method=calibration, cv=TimeSeriesSplit(n_splits=calibration_cv)
            )
        else:
            self.model = base

    def _build_features(self, bars):
        """Compute the feature columns (no label). Shared by train and predict."""
        bars = bars.copy()
        close = bars["close"].astype(float)

        fast_sma = close.rolling(window=9).mean()
        slow_sma = close.rolling(window=21).mean()
        # Normalize SMAs relative to close so features are price-scale invariant
        bars["fast_sma_ratio"] = fast_sma / close.replace(0.0, np.nan) - 1.0
        bars["slow_sma_ratio"] = slow_sma / close.replace(0.0, np.nan) - 1.0
        bars["sma_diff_ratio"] = (fast_sma - slow_sma) / close.replace(0.0, np.nan)

        bars["volatility_pct"] = close.rolling(window=14).std() / close.replace(0.0, np.nan)

        bars["rsi"] = self.compute_rsi(close, window=14)

        ema_fast = close.ewm(span=12, adjust=False).mean()
        ema_slow = close.ewm(span=26, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        # Normalize MACD by close price
        bars["macd_line_ratio"] = macd_line / close.replace(0.0, np.nan)
        bars["macd_signal_ratio"] = macd_signal / close.replace(0.0, np.nan)
        bars["macd_hist_ratio"] = (macd_line - macd_signal) / close.replace(0.0, np.nan)

        bars["momentum_pct"] = (close - close.shift(10)) / close.shift(10).replace(0.0, np.nan)

        trend_ema = close.ewm(span=50, adjust=False).mean()
        bars["trend_following"] = (
            (close > trend_ema) & (macd_line > macd_signal)
        ).astype(int)
        bars["trend_ema_ratio"] = trend_ema / close.replace(0.0, np.nan) - 1.0

        if "volume" not in bars.columns:
            bars["volume"] = 0
        avg_vol = bars["volume"].rolling(window=20).mean().replace(0.0, np.nan)
        bars["volume_ratio"] = bars["volume"].astype(float) / avg_vol

        bars["high_low_pct"] = (bars["high"].astype(float) - bars["low"].astype(float)) / close.replace(0.0, np.nan)
        bars["close_open_pct"] = (close - bars["open"].astype(float)) / bars["open"].astype(float).replace(0.0, np.nan)

        rolling_max = close.rolling(window=20).max()
        rolling_min = close.rolling(window=20).min()
        bars["price_position"] = (close - rolling_min) / (rolling_max - rolling_min + 1e-9)
        return bars

    def prepare_features(self, bars):
        """Training features + label. The label is next-bar direction
        (pct_change().shift(-1)), so the final row — whose label is unknowable —
        is dropped. Use prepare_features_for_predict() for live scoring so the
        current bar isn't thrown away."""
        bars = self._build_features(bars)
        close = bars["close"].astype(float)
        bars["return"] = close.pct_change().shift(-1)
        bars["target"] = np.where(bars["return"] > 0, 1, 0)

        bars = bars.dropna(subset=FEATURE_COLS + ["return"])

        X = bars[FEATURE_COLS]
        y = bars["target"]

        return X, y

    def prepare_features_for_predict(self, bars):
        """Feature rows for live scoring — keeps the CURRENT bar.

        Unlike prepare_features(), this never builds the forward-looking label,
        so it doesn't drop the most recent bar (the one we're deciding on). It
        drops only rows whose features aren't warm yet."""
        bars = self._build_features(bars)
        bars = bars.dropna(subset=FEATURE_COLS)
        return bars[FEATURE_COLS]

    @staticmethod
    def compute_rsi(series, window=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0)

    def fit(self, bars):
        X, y = self.prepare_features(bars)
        if len(X) >= 10 and y.nunique() >= 2:
            self.model.fit(X, y)

    def predict(self, bars):
        X, _ = self.prepare_features(bars)
        if len(X) == 0:
            return np.array([])
        return self.model.predict(X)

    def predict_proba(self, bars):
        X, _ = self.prepare_features(bars)
        if len(X) == 0:
            return np.array([])
        try:
            proba = self.model.predict_proba(X)
            return proba
        except Exception:
            return np.array([])

    def safe_predict_proba(self, X) -> float:
        """Return P(class=1) for the last row of X, defaulting to 0.5 on any error."""
        try:
            if not hasattr(self.model, "classes_"):
                return 0.5
            proba = self.model.predict_proba(X)
            if len(proba) == 0:
                return 0.5
            row = proba[-1]
            if len(row) < 2:
                return 0.5
            return float(row[1])
        except Exception:
            return 0.5