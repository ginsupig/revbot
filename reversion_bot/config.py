from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ReversionConfig:
    trendfail_window: int = 20
    trendfail_threshold: float = 0.005
    # Bars a frozen breakout level may sit unviolated before the breakout is
    # accepted as genuine (no fade). Tunable: walkforward scan shows the
    # fade-activity vs trend-bleed tradeoff lives on this axis.
    trendfail_confirmation_window: int = 3
    band_length: int = 20
    band_std_1: float = 1.0
    band_std_2: float = 2.0
    ri_length: int = 20
    ri_threshold: float = -0.5

    adx_length: int = 14
    adx_max: float = 40.0
    adx_hard_max: float = 50.0

    rsi_length: int = 14
    rsi_max: float = 48.0
    rsi_hard_max: float = 70.0

    # --- Short side (mean-reversion mirror) ---------------------------------
    # When enabled, the engine also emits SHORT_REVERSION on overbought rips:
    # the exact mirror of the long oversold-dip entry. Thresholds are reflected
    # around the neutral midpoints (RI 0, RSI 50). rsi_hard_min guards against
    # shorting into a capitulation that is *also* strongly trending (the mirror
    # of rsi_hard_max, which blocks longs into a blow-off top).
    enable_shorts: bool = True
    ri_short_threshold: float = 0.5
    rsi_min: float = 52.0
    rsi_hard_min: float = 30.0
    # Relaxed short-entry thresholds used only when the market is risk-off AND
    # favor-shorts mode is on: in a confirmed downtrend, fade rips more eagerly
    # (a lower overbought bar) since bounces tend to fail. Applied via the
    # short_bias flag threaded from main.py through the engine.
    risk_off_rsi_min: float = 45.0
    risk_off_ri_short_threshold: float = 0.25

    min_history: int = 160

    require_reclaim_lb1: bool = False
    require_bullish_close: bool = False
    require_volume_expansion: bool = False

    volume_lookback: int = 20
    volume_multiplier_min: float = 1.0

    use_vwap_filter: bool = False
    max_vwap_extension_pct: float = 0.012

    use_trend_filter: bool = True
    trend_ema_length: int = 50
    # How far below the trend EMA a long-reversion entry is still allowed (and,
    # mirrored, how far above for a short). A dip-buy more than this fraction
    # under the trend is judged too extended — buying a falling knife — and
    # vetoed with reason "Higher_Timeframe_Trend_Too_Weak". 0.02 = 2%.
    # Tuned from 0.035: a 0.02 vs 0.035 A/B on two non-overlapping 20-day
    # windows had 0.02 cutting/flipping the universe loss on both while 0.035
    # was marginal-to-harmful (see trend_filter_ab_backtest.py).
    trend_filter_band_pct: float = 0.02
    # Loss-aware re-entry brake: after a CLOSED losing trade on a symbol, block
    # a new long-reversion entry in it for this many minutes. Unlike the plain
    # symbol cooldown (a fixed timer that re-arms regardless of result), this
    # only triggers after a loss — so it stops the bot re-buying a name that
    # just stopped it out and is still falling, without slowing winners.
    # 0 = off. Longer than symbol_cooldown_minutes (30) by design.
    loss_reentry_cooldown_minutes: int = 90

    max_spread_bps: float = 40.0
    min_dollar_volume: float = 750_000.0
    min_price: float = 5.0


@dataclass(frozen=True)
class RiskConfig:
    risk_per_trade_pct: float = 0.005
    max_position_value_pct: float = 0.20

    stop_atr_multiple: float = 1.20
    target_atr_multiple: float = 3.00
    # Trailing stop (long): ratchet the stop up to high_water - trail_atr_multiple*ATR.
    # Validated (execution_tuning_backtest.py) as the load-bearing edge — it lets
    # winners run while holding the win rate, vs a fixed target that caps/gives back.
    # 0 disables. Paired with the wider 3.0 target (target is the backstop the trail
    # rides toward). Gated live by USE_TRAILING_STOP.
    trail_atr_multiple: float = 1.50

    trend_stop_atr_multiple: float = 1.20
    trend_target_atr_multiple: float = 3.00

    trendfail_stop_atr_multiple: float = 1.10
    trendfail_target_atr_multiple: float = 2.20

    min_rr: float = 1.00

    atr_length: int = 14
    atr_floor_pct: float = 0.0035

    round_lot: int = 1
    min_qty: int = 1
    min_position_value: float = 100.0


@dataclass(frozen=True)
class ExecutionConfig:
    slippage_bps_buffer: float = 5.0
    limit_entry_offset_bps: float = 8.0
    use_limit_entry: bool = False
    tif: str = "day"
    paper: bool = True
    base_url: str = ""
    # HTTP connection-pool size for the Alpaca REST session. The bot evaluates
    # the whole universe concurrently, so the requests default of 10 overflows
    # and logs "Connection pool is full, discarding connection" every cycle.
    conn_pool_maxsize: int = 32
    # Opt-in: route the order path through the maintained `alpaca-py` SDK
    # (TradingClient) instead of the EOL `alpaca_trade_api` REST client. Default
    # OFF = legacy path, zero behavior change. Flip on (env USE_ALPACA_PY) to
    # paper-test the new SDK before it becomes the default.
    use_alpaca_py: bool = False
    # Opt-in breakout exit: actively market-close a LONG when price reaches the
    # upper linear-regression channel ("sell the breakout"), instead of waiting
    # for the fixed bracket target — which the A/B showed clips winners. The
    # broker-side stop + (wide) target stay as the safety net underneath. Default
    # OFF = today's static-bracket behavior. Pair with a WIDE TARGET_ATR_MULTIPLE
    # so the bracket target acts as a backstop the active exit beats.
    use_channel_exit: bool = False
    channel_exit_threshold: float = 0.80
    channel_lookback: int = 80
    channel_k: float = 2.0


@dataclass(frozen=True)
class PerformanceConfig:
    state_dir: str = "state/performance"
    enable_adaptive_threshold: bool = True
    min_samples: int = 20
    max_threshold_adj: float = 0.05
    # Persist the fitted ML model so a daily restart doesn't cold-start it (the
    # model is in-memory only otherwise, so ML contributes nothing until it
    # re-fits on live data — and in a one-directional tape it may never fit).
    # Loaded at startup if present and fresh; re-saved after each retrain. A
    # stale/incompatible load is self-healing: a failed predict invalidates it
    # and the retrain-until-fit logic rebuilds it.
    persist_ml_model: bool = True
    ml_model_filename: str = "ml_model.pkl"
    ml_model_max_age_hours: float = 168.0  # 7 days; older -> ignore, cold-start


@dataclass(frozen=True)
class PortfolioConfig:
    max_open_positions: int = 4
    max_trades_per_cycle: int = 2
    max_total_exposure_pct: float = 0.95
    max_positions_per_bucket: int = 2

    max_portfolio_heat_pct: float = 0.02
    max_daily_new_positions: int = 12
    max_reversion_positions: int = 3
    max_trend_positions: int = 2
    max_trendfail_positions: int = 1
    max_positions_per_regime: int = 3

    drawdown_pause_pct: float = 0.025
    reduce_size_after_drawdown_pct: float = 0.01
    reduced_risk_multiplier: float = 0.60

    symbol_cooldown_minutes: int = 30
    # After an entry, block re-entry in the OPPOSITE direction on the same
    # symbol for this many minutes (whipsaw guard: short->stop->long churn).
    # 0 = off. The plain symbol cooldown still blocks all re-entry separately.
    direction_flip_cooldown_minutes: int = 0
