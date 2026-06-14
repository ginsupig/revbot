import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from reversion_bot.market_regime import (
    is_risk_off,
    suppress_longs_if_risk_off,
    resolve_sector_regime,
    suppress_longs_by_sector,
)


def _bars(values):
    return pd.DataFrame({"close": values})


def test_risk_off_true_when_below_trend():
    # Steady decline -> last close sits below its EMA -> risk-off.
    bars = _bars(np.linspace(120, 100, 80))
    assert is_risk_off(bars, ema_length=50) is True


def test_risk_on_when_above_trend():
    # Steady climb -> last close above its EMA -> risk-on.
    bars = _bars(np.linspace(100, 120, 80))
    assert is_risk_off(bars, ema_length=50) is False


def test_fail_open_on_thin_or_missing_data():
    assert is_risk_off(None, ema_length=50) is False
    assert is_risk_off(_bars([100, 101, 102]), ema_length=50) is False          # too few rows
    assert is_risk_off(pd.DataFrame({"open": [1, 2, 3]}), ema_length=2) is False  # no close col


def test_suppress_drops_longs_keeps_shorts():
    cands = [
        {"symbol": "AAA", "go_long": True},
        {"symbol": "BBB", "go_short": True},
        {"symbol": "CCC", "go_long": True},
    ]
    kept, dropped = suppress_longs_if_risk_off(cands, risk_off=True)
    assert [c["symbol"] for c in kept] == ["BBB"]
    assert [c["symbol"] for c in dropped] == ["AAA", "CCC"]


def test_suppress_noop_when_risk_on():
    cands = [{"symbol": "AAA", "go_long": True}]
    kept, dropped = suppress_longs_if_risk_off(cands, risk_off=False)
    assert kept == cands and dropped == []


# --- sector-aware regime ----------------------------------------------------

def _falling(n=80):
    return _bars(np.linspace(120, 100, n))


def _rising(n=80):
    return _bars(np.linspace(100, 120, n))


def test_resolve_sector_regime_per_etf_and_dedup():
    fetched = []

    def fetch(etf):
        fetched.append(etf)
        return _falling() if etf == "SMH" else _rising()

    # semis_ai -> SMH (falling -> risk-off), software/financials -> XLK/XLF (rising).
    regime = resolve_sector_regime(["semis_ai", "software", "financials"], fetch)
    assert regime == {"semis_ai": True, "software": False, "financials": False}
    # Each distinct ETF fetched exactly once (deduped).
    assert sorted(fetched) == ["SMH", "XLF", "XLK"]


def test_resolve_sector_regime_fail_open_on_fetch_error():
    def fetch(etf):
        raise RuntimeError("data hiccup")

    regime = resolve_sector_regime(["semis_ai"], fetch)
    assert regime == {"semis_ai": False}        # error -> risk-on (fail-open)


def test_resolve_skips_sectors_without_etf():
    # An unmapped/proxy-less sector is simply omitted (caller falls back to SPY).
    regime = resolve_sector_regime([None, "not_a_sector"], lambda etf: _falling())
    assert regime == {}


def test_suppress_by_sector_drops_only_the_weak_sector():
    # NVDA/AMD are semis (risk-off); JPM is financials (risk-on); shorts kept.
    cands = [
        {"symbol": "NVDA", "go_long": True},
        {"symbol": "JPM", "go_long": True},
        {"symbol": "AMD", "go_short": True},
    ]
    kept, dropped = suppress_longs_by_sector(
        cands, {"semis_ai": True, "financials": False}, fallback_risk_off=False
    )
    assert {c["symbol"] for c in dropped} == {"NVDA"}
    assert {c["symbol"] for c in kept} == {"JPM", "AMD"}


def test_combine_or_folds_in_broad_benchmark():
    # SPY risk-off (fallback) suppresses even a healthy-sector long under "or".
    cands = [{"symbol": "JPM", "go_long": True}]
    kept, dropped = suppress_longs_by_sector(
        cands, {"financials": False}, fallback_risk_off=True, combine="or"
    )
    assert [c["symbol"] for c in dropped] == ["JPM"]


def test_combine_sector_ignores_broad_for_mapped_names():
    # "sector" mode: a healthy sector trades even when SPY is risk-off...
    cands = [{"symbol": "JPM", "go_long": True}, {"symbol": "ASTS", "go_long": True}]
    kept, dropped = suppress_longs_by_sector(
        cands, {"financials": False}, fallback_risk_off=True, combine="sector"
    )
    # JPM (mapped, healthy sector) kept; ASTS (unmapped) falls back to SPY -> dropped.
    assert {c["symbol"] for c in kept} == {"JPM"}
    assert {c["symbol"] for c in dropped} == {"ASTS"}


def test_unmapped_symbol_uses_fallback_under_or():
    cands = [{"symbol": "ASTS", "go_long": True}]
    kept, dropped = suppress_longs_by_sector(cands, {}, fallback_risk_off=True)
    assert [c["symbol"] for c in dropped] == ["ASTS"]
    kept, dropped = suppress_longs_by_sector(cands, {}, fallback_risk_off=False)
    assert [c["symbol"] for c in kept] == ["ASTS"]


# --- classify_regime_series ------------------------------------------------

from reversion_bot.market_regime import classify_regime_series


def _ohlc_from_close(close):
    close = np.asarray(close, dtype=float)
    return pd.DataFrame({
        "open": close, "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": np.full(len(close), 1e6),
    })


def test_classify_regime_columns_and_empty():
    out = classify_regime_series(None)
    assert list(out.columns) == ["trend_down", "adx_high", "vol_up"]
    assert len(out) == 0
    out2 = classify_regime_series(_ohlc_from_close(np.linspace(100, 90, 120)))
    assert set(out2.columns) == {"trend_down", "adx_high", "vol_up"}
    assert out2.dtypes.apply(lambda d: d == bool).all()


def test_classify_regime_trend_down_flag():
    # A sustained decline: the last bar sits below its 50-EMA -> trend_down True.
    down = classify_regime_series(_ohlc_from_close(np.linspace(120, 80, 150)))
    assert bool(down["trend_down"].iloc[-1]) is True
    # A sustained rise: last bar above its EMA -> trend_down False.
    up = classify_regime_series(_ohlc_from_close(np.linspace(80, 120, 150)))
    assert bool(up["trend_down"].iloc[-1]) is False


def test_classify_regime_vol_up_on_expansion():
    # Flat price, but the bar RANGE ramps up at the end -> ATR% rises above its
    # trailing mean -> vol_up True on the last bar (and not during the calm part).
    n = 150
    close = np.full(n, 100.0)
    spread = np.concatenate([np.full(120, 0.5), np.linspace(0.5, 6.0, 30)])
    df = pd.DataFrame({
        "open": close, "high": close + spread, "low": close - spread,
        "close": close, "volume": np.full(n, 1e6),
    })
    out = classify_regime_series(df)
    assert bool(out["vol_up"].iloc[-1]) is True       # expanding at the end
    assert bool(out["vol_up"].iloc[100]) is False      # calm in the middle
