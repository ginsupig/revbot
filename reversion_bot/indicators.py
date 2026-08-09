from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_volatility(df: pd.DataFrame, length: int = 14) -> pd.Series:
    close = df['close'].astype(float)
    return close.rolling(window=length, min_periods=length).std(ddof=0)


def calculate_momentum(df: pd.DataFrame, length: int = 10) -> pd.Series:
    close = df['close'].astype(float)
    return close - close.shift(length)


def calculate_macd(df: pd.DataFrame, fast_length: int = 12, slow_length: int = 26, signal_length: int = 9):
    close = df['close'].astype(float)
    ema_fast = close.ewm(span=fast_length, adjust=False, min_periods=fast_length).mean()
    ema_slow = close.ewm(span=slow_length, adjust=False, min_periods=slow_length).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_length, adjust=False, min_periods=signal_length).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist


def calculate_bollinger_bands(df: pd.DataFrame, length: int = 20, num_std: float = 2.0):
    close = df['close'].astype(float)
    sma = close.rolling(window=length, min_periods=length).mean()
    std = close.rolling(window=length, min_periods=length).std(ddof=0)
    upper_band = sma + num_std * std
    lower_band = sma - num_std * std
    return sma, upper_band, lower_band


def calculate_rsi(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Wilder RSI with explicit handling for one-sided and flat windows.

    Conventions used here:
    - no losses (avg_loss == 0, avg_gain > 0) -> RSI 100
    - no gains  (avg_gain == 0, avg_loss > 0) -> RSI 0
    - flat tape (avg_gain == avg_loss == 0)   -> RSI 50 (neutral)
    """
    close = df['close'].astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    loss_zero = avg_loss == 0.0
    gain_zero = avg_gain == 0.0
    mask = loss_zero | gain_zero
    rsi = rsi.mask(
        mask,
        np.select(
            [loss_zero & ~gain_zero, gain_zero & ~loss_zero, gain_zero & loss_zero],
            [100.0, 0.0, 50.0],
            default=np.nan,
        ),
    )
    return rsi.fillna(50.0)


def calculate_adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    close = df['close'].astype(float)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1 / length, adjust=False, min_periods=length
    ).mean() / atr.replace(0.0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1 / length, adjust=False, min_periods=length
    ).mean() / atr.replace(0.0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx = dx.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    return adx.fillna(0.0)


def calculate_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    close = df['close'].astype(float)
    tr = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """Session-reset VWAP: cumsum restarts at each calendar date."""
    typical = (df['high'].astype(float) + df['low'].astype(float) + df['close'].astype(float)) / 3.0
    vol = df['volume'].astype(float)
    pv = typical * vol

    # Determine per-row date for grouping
    if hasattr(df.index, 'date'):
        dates = pd.Series(df.index.date, index=df.index)
    else:
        dates = pd.Series([0] * len(df), index=df.index)

    cum_pv = pv.groupby(dates).cumsum()
    cum_vol = vol.groupby(dates).cumsum()
    return cum_pv / cum_vol.replace(0.0, np.nan)


def calculate_ema(series: pd.Series, length: int) -> pd.Series:
    return series.astype(float).ewm(span=length, adjust=False, min_periods=length).mean()


def normalized_reversion_index(close: pd.Series, length: int) -> pd.Series:
    mom = close.astype(float).diff(1)
    numerator = mom.rolling(length, min_periods=length).sum()
    denominator = mom.abs().rolling(length, min_periods=length).sum()
    return numerator / denominator.replace(0.0, np.nan)
