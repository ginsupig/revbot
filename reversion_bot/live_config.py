"""Live revbot configuration, built from the environment.

main.py builds its strategy and risk config here, and backtests that claim to
judge the live bot (research/regime_gate_ab.py) build theirs here too, so the
two cannot drift apart (fleet rule: backtests share code with live).
"""
from __future__ import annotations

import os

from .config import ReversionConfig, RiskConfig


def parse_bool(value: str, default: bool = False) -> bool:
    """Parse an env-style boolean, falling back to `default` for empty or
    unrecognized values. The old version returned False for anything outside
    the truthy set — so a blanked line (`USE_TRAILING_STOP=`) or a typo
    ("Ture") silently disabled default-ON safety features (regime gate,
    trailing stop, carryover flatten, trend filter).
    """
    if value is None:
        return default
    s = str(value).strip().lower()
    if s == "":
        return default
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    print(f"[CONFIG] Unrecognized boolean {value!r} — using default={default}.")
    return default


def strategy_config_from_env(lookback: int) -> ReversionConfig:
    return ReversionConfig(
        min_history=lookback,
        min_dollar_volume=float(os.getenv("MIN_DOLLAR_VOLUME", 750000.0)),
        min_price=float(os.getenv("MIN_PRICE", 2.0)),
        max_spread_bps=float(os.getenv("MAX_SPREAD_BPS", 40.0)),
        require_spread_data=parse_bool(os.getenv("REQUIRE_SPREAD_DATA", "True"), default=True),
        # Bollinger band entry zone (lb1/lb2 = sma -/+ band_std * std).
        band_length=int(os.getenv("BAND_LENGTH", 20)),
        band_std_1=float(os.getenv("BAND_STD_1", 1.0)),
        band_std_2=float(os.getenv("BAND_STD_2", 2.0)),
        # Indicator lookbacks.
        ri_length=int(os.getenv("RI_LENGTH", 20)),
        rsi_length=int(os.getenv("RSI_LENGTH", 14)),
        adx_length=int(os.getenv("ADX_LENGTH", 14)),
        trend_ema_length=int(os.getenv("TREND_EMA_LENGTH", 50)),
        # Long oversold gates.
        ri_threshold=float(os.getenv("RI_THRESHOLD", -0.5)),
        rsi_max=float(os.getenv("RSI_MAX", 40.0)),
        oversold_gate=os.getenv("OVERSOLD_GATE", "and"),
        adx_max=float(os.getenv("ADX_MAX", 40.0)),
        adx_hard_max=float(os.getenv("ADX_HARD_MAX", 50.0)),
        rsi_hard_max=float(os.getenv("RSI_HARD_MAX", 70.0)),
        max_vwap_extension_pct=float(os.getenv("MAX_VWAP_EXTENSION_PCT", 0.012)),
        require_reclaim_lb1=parse_bool(os.getenv("REQUIRE_RECLAIM_LB1", "False")),
        require_bullish_close=parse_bool(os.getenv("REQUIRE_BULLISH_CLOSE", "False")),
        require_volume_expansion=parse_bool(os.getenv("REQUIRE_VOLUME_EXPANSION", "False")),
        use_vwap_filter=parse_bool(os.getenv("USE_VWAP_FILTER", "False")),
        use_trend_filter=parse_bool(os.getenv("USE_TREND_FILTER", "True")),
        trend_filter_band_pct=float(os.getenv("TREND_FILTER_BAND_PCT", 0.02)),
        loss_reentry_cooldown_minutes=int(os.getenv("LOSS_REENTRY_COOLDOWN_MINUTES", 90)),
        # Volume confirmation.
        volume_lookback=int(os.getenv("VOLUME_LOOKBACK", 20)),
        volume_multiplier_min=float(os.getenv("VOLUME_MULTIPLIER_MIN", 1.0)),
        # Trend-fail strategy.
        trendfail_window=int(os.getenv("TRENDFAIL_WINDOW", 20)),
        trendfail_threshold=float(os.getenv("TRENDFAIL_THRESHOLD", 0.005)),
        # Short side (mirror) gates. Off by default: OOS backtest showed shorts
        # bleeding on daily large-caps (8% WR). Enable with ENABLE_SHORTS=True
        # paired with USE_MARKET_REGIME_FILTER + FAVOR_SHORTS_IN_RISK_OFF to
        # only short in confirmed downtrends.
        enable_shorts=parse_bool(os.getenv("ENABLE_SHORTS", "False"), default=False),
        ri_short_threshold=float(os.getenv("RI_SHORT_THRESHOLD", 0.5)),
        rsi_min=float(os.getenv("RSI_MIN", 65.0)),
        rsi_hard_min=float(os.getenv("RSI_HARD_MIN", 30.0)),
        risk_off_rsi_min=float(os.getenv("RISK_OFF_RSI_MIN", 55.0)),
        risk_off_ri_short_threshold=float(os.getenv("RISK_OFF_RI_SHORT_THRESHOLD", 0.30)),
        require_short_reject=parse_bool(os.getenv("REQUIRE_SHORT_REJECT", "True"), default=True),
        require_short_bearish_close=parse_bool(
            os.getenv("REQUIRE_SHORT_BEARISH_CLOSE", "True"), default=True
        ),
        short_trend_filter_band_pct=float(os.getenv("SHORT_TREND_FILTER_BAND_PCT", 0.0)),
    )


def risk_config_from_env() -> RiskConfig:
    return RiskConfig(
        risk_per_trade_pct=float(os.getenv("RISK_PER_TRADE_PCT", 0.005)),
        # MegaEmpire fleet: fall back to the overwatch per-position cap so orders fit the paper guard.
        max_position_value_pct=float(os.getenv("MAX_POSITION_VALUE_PCT", os.getenv("MAX_POSITION_PCT", 0.15))),
        min_rr=float(os.getenv("MIN_RR", 1.5)),
        # Per-style ATR stop/target multiples.
        stop_atr_multiple=float(os.getenv("STOP_ATR_MULTIPLE", 1.20)),
        target_atr_multiple=float(os.getenv("TARGET_ATR_MULTIPLE", 3.50)),
        trail_atr_multiple=float(os.getenv("TRAIL_ATR_MULTIPLE", 1.50)),
        trend_stop_atr_multiple=float(os.getenv("TREND_STOP_ATR_MULTIPLE", 1.20)),
        trend_target_atr_multiple=float(os.getenv("TREND_TARGET_ATR_MULTIPLE", 3.00)),
        trendfail_stop_atr_multiple=float(os.getenv("TRENDFAIL_STOP_ATR_MULTIPLE", 1.10)),
        trendfail_target_atr_multiple=float(os.getenv("TRENDFAIL_TARGET_ATR_MULTIPLE", 2.20)),
        atr_length=int(os.getenv("ATR_LENGTH", 14)),
        atr_floor_pct=float(os.getenv("ATR_FLOOR_PCT", 0.0035)),
    )
