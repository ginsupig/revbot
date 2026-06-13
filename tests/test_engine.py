import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

from reversion_bot.config import ReversionConfig
from reversion_bot.engine import ReversionEngine


def make_df(n=120):
    idx = pd.date_range('2026-01-01', periods=n, freq='min')
    base = np.linspace(100, 101, n)
    noise = np.random.default_rng(7).normal(0, 0.15, n)
    close = base + noise
    open_ = close + np.random.default_rng(8).normal(0, 0.05, n)
    high = np.maximum(open_, close) + 0.10
    low = np.minimum(open_, close) - 0.10
    volume = np.full(n, 50000)
    return pd.DataFrame({
        'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume,
    }, index=idx)


def test_calculate_indicators_adds_columns():
    engine = ReversionEngine(ReversionConfig(min_history=60))
    df = engine.calculate_indicators(make_df())
    for col in ['sma', 'lb1', 'lb2', 'ri', 'rsi', 'adx', 'atr', 'vwap', 'trend_ema']:
        assert col in df.columns


def test_wait_when_no_valid_setup():
    engine = ReversionEngine(ReversionConfig(min_history=60))
    df = engine.calculate_indicators(make_df())
    decision = engine.get_decision(df, symbol='SPY')
    assert decision.signal == 'WAIT'


def make_overbought_df(n=160):
    """A ranging oscillation that ends on a swing high poking above the upper
    band — the mirror of the oversold dip the long side looks for."""
    idx = pd.date_range('2026-01-01', periods=n, freq='min')
    t = np.arange(n)
    # Several gentle cycles keep ADX low (range regime, passes the safety gate),
    # and the phase is chosen so the final bar sits near a peak.
    close = 100 + 3.0 * np.sin(2 * np.pi * t / 25.0)
    close[-1] = close.max() + 0.5  # final push above ub1
    open_ = close - 0.05           # green-ish bars; bearish_close not required by default
    high = np.maximum(open_, close) + 0.10
    low = np.minimum(open_, close) - 0.10
    volume = np.full(n, 60000)
    return pd.DataFrame({
        'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume,
    }, index=idx)


def test_short_signal_on_overbought_when_enabled():
    engine = ReversionEngine(ReversionConfig(min_history=60, enable_shorts=True))
    df = engine.calculate_indicators(make_overbought_df())
    decision = engine.get_decision(df, symbol='SPY')
    assert decision.signal == 'SHORT_REVERSION'
    assert decision.ub1 is not None and decision.close >= decision.ub1


def test_no_short_when_disabled():
    engine = ReversionEngine(ReversionConfig(min_history=60, enable_shorts=False))
    df = engine.calculate_indicators(make_overbought_df())
    decision = engine.get_decision(df, symbol='SPY')
    assert decision.signal != 'SHORT_REVERSION'


def make_falling_knife_df(n=160):
    """A strong, steady decline: high ADX + very low RSI — a falling knife the
    long-reversion must NOT try to catch."""
    idx = pd.date_range('2026-01-01', periods=n, freq='min')
    base = np.linspace(130, 100, n)              # relentless downtrend
    noise = np.random.default_rng(3).normal(0, 0.05, n)
    close = base + noise
    open_ = close + 0.10                          # each bar closes below its open
    high = np.maximum(open_, close) + 0.05
    low = np.minimum(open_, close) - 0.05
    volume = np.full(n, 60000)
    return pd.DataFrame({
        'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume,
    }, index=idx)


def test_falling_knife_blocks_long_entry():
    engine = ReversionEngine(ReversionConfig(min_history=60))
    df = engine.calculate_indicators(make_falling_knife_df())
    row = df.iloc[-1]
    # Precondition: this fixture really is an extended oversold downtrend.
    assert row['adx'] >= 50 and row['rsi'] <= 30
    decision = engine.get_decision(df, symbol='SPY')
    assert decision.signal == 'WAIT'
    assert decision.reason == 'Downtrend_Too_Extended'


def test_short_bias_relaxes_overbought_threshold():
    # Normal thresholds are set so high that nothing ever counts as overbought;
    # the risk-off thresholds are low. So a short only appears under short_bias.
    cfg = ReversionConfig(
        min_history=60, enable_shorts=True,
        rsi_min=99.0, ri_short_threshold=9.0,           # normal: unreachable
        risk_off_rsi_min=30.0, risk_off_ri_short_threshold=0.0,  # risk-off: easy
    )
    engine = ReversionEngine(cfg)
    df = engine.calculate_indicators(make_overbought_df())

    assert engine.get_decision(df, symbol='SPY', short_bias=False).signal != 'SHORT_REVERSION'
    assert engine.get_decision(df, symbol='SPY', short_bias=True).signal == 'SHORT_REVERSION'


def make_oversold_dip_df(n=160):
    """A ranging oscillation (keeps ADX low so the extreme falling-knife veto
    does NOT fire) that ends on a sharp swing low poking well below the lower
    band AND well under the 50-bar trend EMA — exactly the 'extended dip' the
    per-symbol trend filter is meant to refuse."""
    idx = pd.date_range('2026-01-01', periods=n, freq='min')
    t = np.arange(n)
    close = 100 + 3.0 * np.sin(2 * np.pi * t / 25.0)
    close[-1] = close.min() - 4.0          # deep final dip: below lb1 and <<trend_ema
    open_ = close + 0.05
    high = np.maximum(open_, close) + 0.10
    low = np.minimum(open_, close) - 0.10
    volume = np.full(n, 60000)
    return pd.DataFrame({
        'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume,
    }, index=idx)


def test_trend_filter_vetoes_extended_dip_when_enabled():
    df = make_oversold_dip_df()
    # Precondition: not an *extreme* downtrend, so Downtrend_Too_Extended (the
    # only per-symbol guard that was active before) must NOT be what stops it.
    enriched = ReversionEngine(ReversionConfig(min_history=60)).calculate_indicators(df)
    row = enriched.iloc[-1]
    assert not (row['adx'] >= 50 and row['rsi'] <= 30)
    assert row['close'] < row['trend_ema'] * 0.965   # genuinely extended below trend

    # Filter OFF: the dip is a valid long-reversion candidate.
    off = ReversionEngine(ReversionConfig(min_history=60, use_trend_filter=False))
    assert off.get_decision(off.calculate_indicators(df), symbol='WDC').signal == 'LONG_REVERSION'

    # Filter ON (now the default): the same extended dip is refused.
    on = ReversionEngine(ReversionConfig(min_history=60, use_trend_filter=True))
    decision = on.get_decision(on.calculate_indicators(df), symbol='WDC')
    assert decision.signal == 'WAIT'
    assert decision.reason == 'Higher_Timeframe_Trend_Too_Weak'


def test_trend_filter_band_pct_widens_tolerance():
    # A wide enough band lets the same extended dip back through -> proves the
    # knob actually controls the veto distance.
    df = make_oversold_dip_df()
    wide = ReversionEngine(ReversionConfig(
        min_history=60, use_trend_filter=True, trend_filter_band_pct=0.20))
    assert wide.get_decision(wide.calculate_indicators(df), symbol='WDC').signal == 'LONG_REVERSION'
