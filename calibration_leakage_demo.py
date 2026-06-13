"""ML calibration: demonstrate fold leakage, then fix it.

The bot's MLSignalLearner wraps a RandomForest in CalibratedClassifierCV(cv=3).
For classifiers sklearn defaults that to STRATIFIED K-fold: folds are drawn
without regard to time. The features are rolling-window transforms and the
label is shift(-1) next-bar direction, so temporally adjacent rows land in
opposite folds. A train row at t+1 shares almost its entire feature window with
a test row at t — the forest can 'recognize the neighborhood' rather than learn
anything predictive.

Protocol: generate PURE-NOISE random walks (zero drift => true AUC is 0.50 by
construction). Any measured skill is leakage. Compare:

  A) shuffled/stratified folds  (what the current calibration sees internally)
  B) TimeSeriesSplit folds      (honest, past->future only)

Then patch the learner to calibrate with TimeSeriesSplit and compare the live
probabilities both versions emit on a true out-of-time holdout.
"""
import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, cross_val_score

from reversion_bot.ml import MLSignalLearner


def make_noise_ohlcv(n=1500, seed=0):
    """Driftless random walk: the next-bar direction is unpredictable."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.004, n)          # zero drift
    close = 100 * np.cumprod(1 + rets)
    high = close * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.002, n)))
    open_ = np.r_[close[0], close[:-1]]
    vol = rng.integers(50_000, 150_000, n).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": vol})


def measured_auc(X, y, cv, seeds=range(5)):
    aucs = []
    for s in seeds:
        rf = RandomForestClassifier(n_estimators=100, random_state=s, n_jobs=-1)
        aucs.append(cross_val_score(rf, X, y, cv=cv, scoring="roc_auc").mean())
    return float(np.mean(aucs)), float(np.std(aucs))


def part1_leakage():
    print("=" * 72)
    print("PART 1 — apparent skill on PURE NOISE (true AUC = 0.500)")
    print("=" * 72)
    learner = MLSignalLearner(calibration=None)
    rows = []
    for seed in range(5):
        df = make_noise_ohlcv(seed=seed)
        X, y = learner.prepare_features(df)
        a_shuf, _ = measured_auc(X, y, StratifiedKFold(3, shuffle=True, random_state=0), seeds=[0])
        a_time, _ = measured_auc(X, y, TimeSeriesSplit(3), seeds=[0])
        rows.append((seed, a_shuf, a_time))
        print(f"  walk #{seed}:  shuffled-fold AUC = {a_shuf:.3f}   "
              f"TimeSeriesSplit AUC = {a_time:.3f}")
    arr = np.array(rows)
    print(f"\n  MEAN:      shuffled-fold AUC = {arr[:,1].mean():.3f}   "
          f"TimeSeriesSplit AUC = {arr[:,2].mean():.3f}")
    print("  -> skill above 0.500 under shuffled folds on driftless noise is")
    print("     pure temporal leakage; TimeSeriesSplit reports ~no skill (honest).")
    return arr[:, 1].mean(), arr[:, 2].mean()


def build_calibrated(cv):
    base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    return CalibratedClassifierCV(base, method="sigmoid", cv=cv)


def part2_fix():
    print()
    print("=" * 72)
    print("PART 2 — live-probability behavior: current vs fixed calibration")
    print("=" * 72)
    learner = MLSignalLearner(calibration=None)
    cur_dev, fix_dev, cur_brier, fix_brier = [], [], [], []
    for seed in range(5):
        df = make_noise_ohlcv(n=1500, seed=100 + seed)
        X, y = learner.prepare_features(df)
        # true out-of-time split: train on first 70%, score the last 30%
        cut = int(len(X) * 0.7)
        X_tr, y_tr, X_te, y_te = X.iloc[:cut], y.iloc[:cut], X.iloc[cut:], y.iloc[cut:]

        cur = build_calibrated(cv=3)                    # current: stratified
        fix = build_calibrated(cv=TimeSeriesSplit(3))   # fixed: chronological
        cur.fit(X_tr, y_tr)
        fix.fit(X_tr, y_tr)

        p_cur = cur.predict_proba(X_te)[:, 1]
        p_fix = fix.predict_proba(X_te)[:, 1]
        cur_dev.append(np.abs(p_cur - 0.5).mean())
        fix_dev.append(np.abs(p_fix - 0.5).mean())
        cur_brier.append(np.mean((p_cur - y_te) ** 2))
        fix_brier.append(np.mean((p_fix - y_te) ** 2))

    print(f"  mean |p - 0.5| on unseen future bars (overconfidence):")
    print(f"     current (stratified cal): {np.mean(cur_dev):.4f}")
    print(f"     fixed  (TimeSeriesSplit): {np.mean(fix_dev):.4f}")
    print(f"  Brier score on unseen future bars (lower = better calibrated):")
    print(f"     current (stratified cal): {np.mean(cur_brier):.5f}")
    print(f"     fixed  (TimeSeriesSplit): {np.mean(fix_brier):.5f}")
    print(f"     coin-flip reference:      0.25000")
    print("  -> on noise the FIXED version stays nearer 0.5 (it has learned it")
    print("     knows nothing); the current version is more overconfident, and")
    print("     that confidence feeds 25% of every entry score.")


if __name__ == "__main__":
    part1_leakage()
    part2_fix()
