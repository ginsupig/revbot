"""Retrain RevBot's persisted ML scorer with the current sklearn version.

Uses pooled daily history across the neutral universe. Features and next-bar
labels are built within each symbol (never across symbol boundaries), then
pooled in chronological order for time-series calibration. The artifact is
written atomically so a running service can never observe a partial pickle.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from dotenv import load_dotenv
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

from reversion_bot.ml import FEATURE_COLS, MLSignalLearner
from reversion_bot.universe import broad_reversion_universe
from run_real_backtest import fetch_alpaca_bars_batch


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "state" / "performance" / "ml_model.pkl"
META_PATH = MODEL_PATH.with_suffix(".meta.json")


def _chronological_splits(dates: pd.Series, n_splits: int = 3):
    unique = pd.Series(sorted(pd.unique(dates)))
    folds = []
    for k in range(1, n_splits + 1):
        train_end = int(len(unique) * k / (n_splits + 1))
        test_end = int(len(unique) * (k + 1) / (n_splits + 1))
        train_dates = set(unique.iloc[:train_end])
        test_dates = set(unique.iloc[train_end:test_end])
        train_idx = dates.index[dates.isin(train_dates)].to_numpy()
        test_idx = dates.index[dates.isin(test_dates)].to_numpy()
        if len(train_idx) and len(test_idx):
            folds.append((train_idx, test_idx))
    if len(folds) < 2:
        raise RuntimeError("Insufficient distinct dates for chronological calibration")
    return folds


def main() -> int:
    load_dotenv(ROOT / ".env")
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=3 * 365)
    symbols = broad_reversion_universe()
    bars_by_symbol = fetch_alpaca_bars_batch(
        symbols,
        start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "1Day",
    )

    feature_builder = MLSignalLearner(calibration=None)
    parts = []
    for symbol, bars in bars_by_symbol.items():
        if bars is None or len(bars) < 80 or "date" not in bars.columns:
            continue
        X, y = feature_builder.prepare_features(bars)
        if len(X) < 30:
            continue
        part = X.copy()
        part["target"] = y.to_numpy()
        part["date"] = pd.to_datetime(bars.loc[X.index, "date"], utc=True).to_numpy()
        part["symbol"] = symbol
        parts.append(part)

    if not parts:
        raise RuntimeError("No usable training data returned")
    pooled = pd.concat(parts, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    X = pooled[FEATURE_COLS]
    y = pooled["target"].astype(int)
    if y.nunique() < 2:
        raise RuntimeError("Training labels contain only one class")

    splits = _chronological_splits(pooled["date"])
    base = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model = CalibratedClassifierCV(base, method="sigmoid", cv=splits)
    model.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix="ml_model_", suffix=".pkl", dir=MODEL_PATH.parent)
    os.close(fd)
    try:
        joblib.dump(model, tmp_name)
        os.replace(tmp_name, MODEL_PATH)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)

    META_PATH.write_text(json.dumps({
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "symbols": int(pooled["symbol"].nunique()),
        "rows": int(len(pooled)),
        "start": str(pooled["date"].min()),
        "end": str(pooled["date"].max()),
        "positive_rate": float(y.mean()),
    }, indent=2), encoding="utf-8")
    print(META_PATH.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
