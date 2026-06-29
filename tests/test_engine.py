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
    """A ranging oscillation that ends on a sharp multi-bar rip pushing RSI and
    RI into genuinely overbought territory (RSI >= 65, RI >= 0.5), with the
    FINAL bar closing back below the upper band — a classic overbought rejection.

    The ramp pushes price above ub1 over several bars (building overbought RSI/RI),
    then the last bar closes back inside the band (reject_ub1 confirmation) and
    is bearish (close <= open). The indicators reflect the multi-bar rip, not
    just one bar, so the AND gate on RI AND RSI both fire.
    """
    idx = pd.date_range('2026-01-01', periods=n, freq='min')
    t = np.arange(n)
    # Gentle cycles to keep ADX moderate, with a strong ramp at the end.
    close = 100 + 3.0 * np.sin(2 * np.pi * t / 25.0)
    # Steep ramp: 15 bars pushing strongly up to build overbought RSI/RI
    for i in range(n - 15, n - 1):
        close[i] = close[n - 16] + (i - (n - 16)) * 0.8
    # Final bar: pull back BELOW where ub1 will be (~4-5 below the peak)
    # but still well above the SMA so it stays in the "overbought zone" from
    # the engine's perspective (close >= ub1 on the last row).
    # The key: we need close >= ub1 (in short zone) but the rejection is that
    # prev was ABOVE prev_ub1 and now we're still above ub1 but dropping.
    # Actually, the reject_ub1 needs close < ub1. So the final bar must be
    # above ub1 enough to be in the short zone but the reject just needs a
    # downtick from prev.
    close[-1] = close[-2] - 0.3
    open_ = close.copy()
    open_[-1] = close[-1] + 0.15   # final bar: close < open = bearish
    open_[:-1] = close[:-1] - 0.05
    high = np.maximum(open_, close) + 0.10
    low = np.minimum(open_, close) - 0.10
    volume = np.full(n, 60000)
    return pd.DataFrame({
        'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume,
    }, index=idx)


def test_short_signal_on_overbought_when_enabled():
    # use_trend_filter is off here: this test is about short-enablement, not the
    # higher-timeframe trend gate (which has its own tests).
    engine = ReversionEngine(ReversionConfig(
        min_history=60, enable_shorts=True, use_trend_filter=False))
    df = engine.calculate_indicators(make_overbought_df())
    decision = engine.get_decision(df, symbol='SPY')
    assert decision.signal == 'SHORT_REVERSION'


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
    # Normal thresholds are set so high that nothing ever counts as overbought
    # (AND gate requires BOTH unreachable); the risk-off thresholds are easy.
    # So a short only appears under short_bias.
    cfg = ReversionConfig(
        min_history=60, enable_shorts=True, use_trend_filter=False,  # isolate short_bias
        rsi_min=99.0, ri_short_threshold=9.0,           # normal: unreachable (AND)
        risk_off_rsi_min=30.0, risk_off_ri_short_threshold=0.0,  # risk-off: easy (AND still passes)
    )
    engine = ReversionEngine(cfg)
    df = engine.calculate_indicators(make_overbought_df())

    assert engine.get_decision(df, symbol='SPY', short_bias=False).signal != 'SHORT_REVERSION'
    assert engine.get_decision(df, symbol='SPY', short_bias=True).signal == 'SHORT_REVERSION'


def make_oversold_dip_df(n=160):
    """A ranging oscillation (keeps ADX low so the extreme falling-knife veto
    does NOT fire) that ends on a multi-bar drop producing genuinely oversold
    RSI (<= 35) and RI (<= -0.5), well below the lower band AND the 50-bar
    trend EMA — exactly the 'extended dip' the per-symbol trend filter should
    refuse."""
    idx = pd.date_range('2026-01-01', periods=n, freq='min')
    t = np.arange(n)
    close = 100 + 3.0 * np.sin(2 * np.pi * t / 25.0)
    # Multi-bar steep drop at the end to push RSI and RI into oversold under AND gate
    for i in range(n - 12, n):
        close[i] = close[n - 13] - (i - (n - 13)) * 0.7
    close[-1] = close[-2] - 1.5   # final deep dip: below lb1 and << trend_ema
    open_ = close + 0.15           # red bars (open > close)
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
    # Use OR gate + loose RSI to isolate the trend filter test from oversold gate changes.
    # Use OR gate + loose RSI + high adx_max to isolate the trend filter test.
    off = ReversionEngine(ReversionConfig(min_history=60, use_trend_filter=False, oversold_gate="or", rsi_max=48.0, adx_max=50.0))
    assert off.get_decision(off.calculate_indicators(df), symbol='WDC').signal == 'LONG_REVERSION'

    # Filter ON (now the default): the same extended dip is refused.
    on = ReversionEngine(ReversionConfig(min_history=60, use_trend_filter=True, oversold_gate="or", rsi_max=48.0, adx_max=50.0))
    decision = on.get_decision(on.calculate_indicators(df), symbol='WDC')
    assert decision.signal == 'WAIT'
    assert decision.reason == 'Higher_Timeframe_Trend_Too_Weak'


def test_trend_filter_band_pct_widens_tolerance():
    # A wide enough band lets the same extended dip back through -> proves the
    # knob actually controls the veto distance.
    df = make_oversold_dip_df()
    wide = ReversionEngine(ReversionConfig(
        min_history=60, use_trend_filter=True, trend_filter_band_pct=0.20, oversold_gate="or", rsi_max=48.0, adx_max=50.0))
    assert wide.get_decision(wide.calculate_indicators(df), symbol='WDC').signal == 'LONG_REVERSION'
