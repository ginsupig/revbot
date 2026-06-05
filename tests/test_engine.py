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
