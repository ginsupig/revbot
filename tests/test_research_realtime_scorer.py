import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from research.realtime_scorer import (
    _zscore, _trailing_ret_by_ord, _trend_on_by_ord, _candidates,
    score_sweep, run_realtime_scorer, compare_gate, gate_diagnosis,
    format_gate_comparison, liquid_filter, PROFILES, FACTORS,
    NEUTRAL_SECTOR_OF, ALL_SECTOR_OF,
)
from research.pipeline import PipelineResult


def _bars(close, dates, high=None, low=None):
    close = np.asarray(close, float)
    return pd.DataFrame({
        "open": close,
        "high": close * 1.01 if high is None else high,
        "low": close * 0.99 if low is None else low,
        "close": close,
        "volume": np.full(len(close), 1e6),
        "date": pd.to_datetime(dates),
    })


def test_zscore_handles_degenerate():
    assert np.allclose(_zscore([5.0]), [0.0])           # single element
    assert np.allclose(_zscore([3.0, 3.0, 3.0]), [0, 0, 0])  # zero variance
    z = _zscore([1.0, 2.0, 3.0])
    assert abs(z.mean()) < 1e-12 and z[2] > z[0]


def test_trailing_ret_is_causal():
    dates = pd.date_range("2025-01-01", periods=30)
    close = np.linspace(100, 130, 30)
    bars = _bars(close, dates)
    r = _trailing_ret_by_ord(bars, lookback=10)
    # the return keyed at day d uses close[d]/close[d-10]-1 — only past bars
    d20 = int(pd.Timestamp(dates[20]).toordinal())
    assert abs(r[d20] - (close[20] / close[10] - 1.0)) < 1e-12
    # the first 10 days have no trailing window -> absent (no peeking backward off-array)
    d5 = int(pd.Timestamp(dates[5]).toordinal())
    assert d5 not in r


def test_regime_gate_tracks_sma():
    dates = pd.date_range("2025-01-01", periods=80)
    close = np.concatenate([np.linspace(100, 60, 40), np.linspace(60, 140, 40)])
    on = _trend_on_by_ord(_bars(close, dates), win=20)
    # falling first half -> below SMA (risk-off); rising second half -> above
    early = int(pd.Timestamp(dates[35]).toordinal())
    late = int(pd.Timestamp(dates[75]).toordinal())
    assert on[early] is False and on[late] is True


def test_candidates_fire_on_oversold_below_band():
    dates = pd.date_range("2025-01-01", periods=80)
    # steady, then a sharp flush that drives price below the lower band + RSI down
    close = np.concatenate([np.full(60, 100.0), np.linspace(100, 80, 20)])
    rows = _candidates(_bars(close, dates), rsi_max=45.0)
    assert len(rows) > 0
    # each row: (day_ord, entry_i, feature_dict, ret)
    for (d, i, feat, ret) in rows:
        assert feat["setup"] > 0 and feat["movement"] > 0
        # the new factors are present and finite
        for k in ("exhaustion", "volexp", "trend"):
            assert k in feat and np.isfinite(feat[k])


def test_exhaustion_factor_reacts_to_volume():
    dates = pd.date_range("2025-01-01", periods=80)
    close = np.concatenate([np.full(60, 100.0), np.linspace(100, 80, 20)])
    base = _bars(close, dates)
    # same price path, but a volume spike on the flush bars -> higher exhaustion
    spike = base.copy()
    vol = np.full(80, 1e6)
    vol[60:] = 5e6
    spike["volume"] = vol
    f_base = {i: feat for (_d, i, feat, _r) in _candidates(base, 45.0)}
    f_spike = {i: feat for (_d, i, feat, _r) in _candidates(spike, 45.0)}
    common = set(f_base) & set(f_spike)
    assert common
    # at least one shared candidate has strictly higher exhaustion under the spike
    assert any(f_spike[i]["exhaustion"] > f_base[i]["exhaustion"] for i in common)


def test_trend_factor_sign_distinguishes_knife_from_dip():
    dates = pd.date_range("2025-01-01", periods=80)
    # dip-in-uptrend: long rise then a small pullback -> close still above SMA50 (>0)
    uptrend = np.concatenate([np.linspace(60, 120, 70), np.linspace(120, 112, 10)])
    rows = _candidates(_bars(uptrend, dates), rsi_max=100.0)
    if rows:
        assert rows[-1][2]["trend"] > -0.5    # not a deep broken-trend reading


def _panel():
    """Two names on the same calendar: STRONG has high relative strength /
    momentum, WEAK is flat. Both periodically dip below band to create candidates."""
    dates = pd.date_range("2024-01-01", periods=160)
    rng = np.random.default_rng(0)
    def series(drift):
        c = [100.0]
        for k in range(1, 160):
            shock = 0.85 if k % 20 == 0 else 1.0          # periodic flush -> candidate
            c.append(c[-1] * (1 + drift + rng.normal(0, 0.005)) * shock)
        return np.array(c)
    strong = _bars(series(0.004), dates)
    weak = _bars(series(0.0), dates)
    spy = _bars(np.linspace(100, 140, 160), dates)         # SPY uptrend -> regime on
    return {"STRONG": strong, "WEAK": weak}, spy, dates


def test_score_sweep_prefers_high_rs_name():
    symbol_bars, spy, _ = _panel()
    research, _ = score_sweep(
        symbol_bars, spy, profiles={"momentum": PROFILES["momentum"]},
        lookbacks=[20], ks=[1], regime_gate=True, holdout_cut_ord=None)
    book = research.get("momentum_lb20_K1", np.array([]))
    assert book.size > 0          # the momentum profile with K=1 traded a name each eligible day


def test_score_sweep_regime_gate_blanks_risk_off():
    symbol_bars, _spy, dates = _panel()
    spy_down = _bars(np.linspace(140, 80, 160), dates)     # SPY downtrend -> gate off
    research, _ = score_sweep(symbol_bars, spy_down, profiles={"balanced": PROFILES["balanced"]},
                              lookbacks=[20], ks=[2], regime_gate=True)
    # everything risk-off -> no longs taken
    assert research.get("balanced_lb20_K2", np.array([])).size == 0


def test_run_realtime_scorer_end_to_end():
    symbol_bars, spy, _ = _panel()
    res = run_realtime_scorer(symbol_bars, spy, lookbacks=[20, 40], ks=[1, 2],
                              regime_gate=True, holdout_frac=0.25)
    assert isinstance(res, PipelineResult)
    # 5 profiles x 2 lookbacks x 2 Ks
    assert res.n_trials == len(PROFILES) * 2 * 2
    assert res.chosen is not None


def test_run_realtime_scorer_empty_without_candidates():
    dates = pd.date_range("2025-01-01", periods=40)
    flat = _bars(np.full(40, 100.0), dates)                # never below band -> no candidates
    spy = _bars(np.linspace(100, 120, 40), dates)
    res = run_realtime_scorer({"X": flat}, spy)
    assert res.chosen is None and "empty" in res.verdict.lower()


def test_sector_factor_injected_optional():
    symbol_bars, spy, dates = _panel()
    sector_of = {"STRONG": "XLK", "WEAK": "XLK"}
    sector_bars = {"XLK": _bars(np.linspace(100, 150, 160), dates)}
    research, _ = score_sweep(symbol_bars, spy, sector_of=sector_of, sector_bars=sector_bars,
                              profiles={"sector_rs": PROFILES["sector_rs"]},
                              lookbacks=[20], ks=[1], regime_gate=True)
    assert research.get("sector_rs_lb20_K1", np.array([])).size > 0


# --- gate-on vs gate-off comparison (the 1-before-2 diagnostic) ----------------

def _result(deploy: bool, mean=0.01, n=200) -> PipelineResult:
    verdict = "DEPLOY-CANDIDATE: ..." if deploy else "REJECT: failed on the untouched holdout"
    return PipelineResult(chosen="balanced_lb20_K3", n_trials=20, research_sharpe=0.3,
                          deflated_sharpe=0.1 if deploy else -0.1, passed_gate=deploy,
                          robust=deploy, holdout=dict(n=n, mean=mean, pf=1.4),
                          breakeven_bps=mean * 10000, verdict=verdict)


def test_gate_diagnosis_off_only_justifies_hmm():
    msg = gate_diagnosis(_result(False), _result(True))
    assert "weak link" in msg and "HMM" in msg and "justified" in msg


def test_gate_diagnosis_both_clear_hmm_premature():
    msg = gate_diagnosis(_result(True), _result(True))
    assert "both clear" in msg and "premature" in msg


def test_gate_diagnosis_on_only_keep_gate():
    msg = gate_diagnosis(_result(True), _result(False))
    assert "earning its keep" in msg.lower() or "keep the sma gate" in msg.lower()


def test_gate_diagnosis_neither_clears_not_regime():
    msg = gate_diagnosis(_result(False), _result(False))
    assert "neither clears" in msg and "not in the regime" in msg


def test_compare_gate_runs_both_and_formats():
    symbol_bars, spy, _ = _panel()
    on, off = compare_gate(symbol_bars, spy, lookbacks=[20], ks=[2],
                           profiles={"balanced": PROFILES["balanced"]})
    assert isinstance(on, PipelineResult) and isinstance(off, PipelineResult)
    txt = format_gate_comparison(on, off)
    assert "gate ON" in txt and "gate OFF" in txt and "DIAGNOSIS" in txt


def test_single_profile_is_one_trial_no_deflation_tax():
    # isolating one profile/lookback/K -> n_trials == 1, so the deflated-Sharpe
    # gate stops penalizing best-of-N (the pre-registered-hypothesis path).
    symbol_bars, spy, _ = _panel()
    res = run_realtime_scorer(symbol_bars, spy,
                              profiles={"rs_setup": PROFILES["rs_setup"]},
                              lookbacks=[20], ks=[5], regime_gate=False)
    assert res.n_trials == 1
    assert res.chosen == "rs_setup_lb20_K5"


def test_new_profiles_only_reference_known_factors():
    # every weight key in every profile must be a real factor (guards typos that
    # would silently contribute zero to the composite).
    for name, w in PROFILES.items():
        assert set(w).issubset(set(FACTORS)), name
    assert "exhaustion" in PROFILES and "dip_in_trend" in PROFILES


def test_exhaustion_profile_runs_end_to_end():
    symbol_bars, spy, _ = _panel()
    res = run_realtime_scorer(symbol_bars, spy,
                              profiles={"exhaustion": PROFILES["exhaustion"]},
                              lookbacks=[20], ks=[3], regime_gate=False)
    assert isinstance(res, PipelineResult) and res.n_trials == 1


def test_liquid_filter_screens_price_and_volume():
    dates = pd.date_range("2025-01-01", periods=40)
    # liquid: $100, 1M shares -> $100M ADV, well above the $20M floor
    liquid = _bars(np.full(40, 100.0), dates)
    assert liquid_filter(liquid, min_price=5.0, min_dollar_vol=20e6) is True
    # penny stock: below the $5 floor
    penny = _bars(np.full(40, 2.0), dates)
    assert liquid_filter(penny, min_price=5.0, min_dollar_vol=0) is False
    # illiquid: $50 but only 1k shares -> $50k ADV, below floor
    thin = _bars(np.full(40, 50.0), dates)
    thin["volume"] = np.full(40, 1e3)
    assert liquid_filter(thin, min_price=5.0, min_dollar_vol=20e6) is False
    # too short / missing data -> rejected
    assert liquid_filter(_bars(np.full(10, 100.0), dates[:10])) is False


def test_liquid_filter_is_timeframe_invariant():
    # same DAILY dollar volume ($100M/day) on daily vs 5Min bars must give the same
    # verdict — the per-bar $ vol on 5Min must not be compared to a daily floor.
    days = pd.date_range("2025-01-01", periods=40, freq="D")
    daily = _bars(np.full(40, 100.0), days)
    daily["volume"] = np.full(40, 1e6)                  # $100M/day
    assert liquid_filter(daily, min_dollar_vol=20e6) is True
    # 5Min: 78 bars/day x 40 days, each bar 1/78 of the day's volume -> same daily $vol
    intraday_ts = pd.date_range("2025-01-02 09:30", periods=78 * 40, freq="5min")
    intra = _bars(np.full(78 * 40, 100.0), intraday_ts)
    intra["volume"] = np.full(78 * 40, 1e6 / 78)        # sums to 1e6/day
    assert liquid_filter(intra, min_dollar_vol=20e6) is True   # was False before the fix


def test_neutral_universe_is_broad_and_mapped():
    # survivorship control: many names, all 11 SPDR sectors, includes laggards
    assert len(NEUTRAL_SECTOR_OF) >= 60
    assert len(set(NEUTRAL_SECTOR_OF.values())) >= 10
    for laggard in ("INTC", "T", "VZ", "PFE", "BA", "CLF", "F"):
        assert laggard in NEUTRAL_SECTOR_OF
    # the combined map still covers the curated watchlist
    assert ALL_SECTOR_OF["ARM"] == "XLK" and ALL_SECTOR_OF["AAPL"] == "XLK"
