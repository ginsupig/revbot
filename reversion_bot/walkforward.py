import numpy as np
import pandas as pd
from reversion_bot.engine import ReversionEngine
from reversion_bot.config import ReversionConfig
from reversion_bot.autotune import AutoTuner
from reversion_bot.analytics import profit_factor, sharpe_ratio, max_drawdown
from reversion_bot.cost_model import passes_cost_gate, in_cooldown
from reversion_bot.exhaustion import exhaustion_short_signal, relative_volume
from reversion_bot.indicators import calculate_atr

# --- Live-execution constants (mirror risk.py / main.py) ---
STOP_ATR_MULTIPLE = 1.20        # risk.py RiskConfig.stop_atr_multiple
TARGET_ATR_MULTIPLE = 2.00      # risk.py RiskConfig.target_atr_multiple
ATR_FLOOR_PCT = 0.0035          # risk.py RiskConfig.atr_floor_pct
SLIPPAGE_PCT = 0.0008           # 8 bps round-trip friction on leveraged ETFs
PERIODS_PER_YEAR_5MIN = 78 * 252  # 5-min RTH bars: correct Sharpe annualization
MORNING_BLACKOUT_HOUR_CT = 9    # is_morning_blackout(): block entries before 9:00 CT
EOD_LIQUIDATION_HOUR_CT = 14    # liquidate_all_positions(): force flat at 14:50 CT
EOD_LIQUIDATION_MINUTE_CT = 50


def _ct_hour_minute(index_like) -> tuple[pd.Series, pd.Series]:
    """Return (hour, minute) in America/Chicago for a datetime-like index/series.

    Accepts tz-aware or tz-naive timestamps; naive are assumed UTC (Alpaca bars).
    """
    ts = pd.to_datetime(index_like, utc=True)
    if isinstance(ts, pd.Series):
        ct = ts.dt.tz_convert("America/Chicago")
        return ct.dt.hour, ct.dt.minute
    ct = ts.tz_convert("America/Chicago")
    return ct.hour, ct.minute


def mean_reversion_strategy(df, **kwargs):
    """Entry signal generator + REALISTIC exit simulation.

    Entries use the ReversionEngine LONG_REVERSION decision. Exits mirror the
    live bracket: dynamic ATR stop/target, morning-blackout entry filter, and
    EOD liquidation at 14:50 CT. Round-trip friction is applied per trade.

    Cost/churn controls (popped before building the engine config; all default
    to the prior behavior so existing results are unchanged):
        reentry_cooldown_bars : block re-entry for N bars after an exit (live
            parity: 30-min cooldown == 6 bars on 5-min data). Default 0 (off).
        min_reward_cost_ratio : skip entries whose bracket reward doesn't clear
            cost by this multiple. Default 0.0 (off).
        cost_pct : round-trip friction applied to each trade. Default 8 bps.
    """
    reentry_cooldown_bars = int(kwargs.pop("reentry_cooldown_bars", 0))
    min_reward_cost_ratio = float(kwargs.pop("min_reward_cost_ratio", 0.0))
    cost_pct = float(kwargs.pop("cost_pct", SLIPPAGE_PCT))

    config_defaults = dict(
        band_length=20,
        band_std_1=1.0,
        band_std_2=2.0,
        min_history=2,
        ri_threshold=-0.5,
        rsi_max=48.0,
        adx_max=40.0,
        min_price=0.0,
        min_dollar_volume=0.0,
        require_reclaim_lb1=False,
        require_bullish_close=False,
        require_volume_expansion=False,
        use_vwap_filter=False,
        use_trend_filter=False,
    )
    config_defaults.update(kwargs)
    engine = ReversionEngine(ReversionConfig(**config_defaults))
    enriched = engine.calculate_indicators(df)

    # Resolve a Central-Time hour/minute for every bar (Tasks 2).
    if "date" in enriched.columns:
        hour_ct, minute_ct = _ct_hour_minute(enriched["date"])
        hour_ct = pd.Series(np.asarray(hour_ct), index=enriched.index)
        minute_ct = pd.Series(np.asarray(minute_ct), index=enriched.index)
    elif isinstance(enriched.index, pd.DatetimeIndex):
        h, m = _ct_hour_minute(enriched.index)
        hour_ct = pd.Series(np.asarray(h), index=enriched.index)
        minute_ct = pd.Series(np.asarray(m), index=enriched.index)
    else:
        # No timestamps available -> time filters disabled (signals only).
        hour_ct = pd.Series(np.full(len(enriched), 10), index=enriched.index)
        minute_ct = pd.Series(np.zeros(len(enriched), dtype=int), index=enriched.index)

    enriched["signal"] = 0
    enriched["position"] = 0
    enriched["strategy_return"] = 0.0

    in_trade = False
    entry_price = stop_price = target_price = 0.0
    prev_close = None
    last_exit_idx = None

    for i in range(len(enriched)):
        row = enriched.iloc[i]
        close = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        h = int(hour_ct.iloc[i])
        m = int(minute_ct.iloc[i])

        bar_return = 0.0

        if in_trade:
            # --- EXIT LOGIC: ATR bracket + EOD liquidation (Tasks 1 & 2) ---
            eod = (h > EOD_LIQUIDATION_HOUR_CT) or (
                h == EOD_LIQUIDATION_HOUR_CT and m >= EOD_LIQUIDATION_MINUTE_CT
            )
            exit_price = None
            if low <= stop_price:
                exit_price = stop_price               # stop first (conservative)
            elif high >= target_price:
                exit_price = target_price
            elif eod:
                exit_price = close                    # forced EOD flat

            if exit_price is not None:
                # Round-trip return from entry to exit, minus friction (Task 3).
                bar_return = (exit_price / entry_price - 1.0) - cost_pct
                in_trade = False
                last_exit_idx = i
                enriched.at[enriched.index[i], "position"] = 0
            else:
                bar_return = (close / prev_close - 1.0) if prev_close else 0.0
                enriched.at[enriched.index[i], "position"] = 1
        else:
            # --- ENTRY LOGIC: engine signal + morning blackout (Task 2) ---
            morning_blackout = h < MORNING_BLACKOUT_HOUR_CT
            eod = (h > EOD_LIQUIDATION_HOUR_CT) or (
                h == EOD_LIQUIDATION_HOUR_CT and m >= EOD_LIQUIDATION_MINUTE_CT
            )
            cooldown = in_cooldown(i, last_exit_idx, reentry_cooldown_bars)
            if not morning_blackout and not eod and not cooldown:
                # get_decision reads only the last row (current bar) and the
                # row before it (for reclaim/bullish checks). Passing the whole
                # history slice made the backtest O(n^2) in pandas slicing; a
                # 2-row window is semantically identical (verified by
                # tests/test_walkforward_window_equivalence.py) and O(n).
                dec = engine.get_decision(enriched.iloc[max(0, i - 1): i + 1])
                if dec.signal == "LONG_REVERSION":
                    atr = float(dec.atr or 0.0)
                    atr = max(atr, close * ATR_FLOOR_PCT)   # ATR floor (risk.py)
                    entry_price = close
                    stop_price = round(entry_price - atr * STOP_ATR_MULTIPLE, 4)
                    target_price = round(entry_price + atr * TARGET_ATR_MULTIPLE, 4)
                    # Skip entries too small to clear round-trip cost (churn).
                    cost_ok = passes_cost_gate(
                        entry_price, target_price, cost_pct, min_reward_cost_ratio
                    )
                    if stop_price < entry_price < target_price and cost_ok:
                        in_trade = True
                        enriched.at[enriched.index[i], "signal"] = 1
                        enriched.at[enriched.index[i], "position"] = 1

        enriched.at[enriched.index[i], "strategy_return"] = bar_return
        prev_close = close

    enriched["market_return"] = enriched["close"].pct_change()
    enriched["pnl"] = enriched["strategy_return"]
    return enriched


def short_exhaustion_strategy(df, **kwargs):
    """SHORT entries on the bearish-exhaustion signal + realistic exit sim.

    Mirror of mean_reversion_strategy for the short side: entries fire when
    exhaustion_short_signal prints (higher high on fading RVOL); exits use an
    inverted ATR bracket (stop ABOVE entry, target BELOW) plus EOD liquidation,
    with per-trade friction. Returns the frame with a ``pnl`` column so the same
    profit_factor / sharpe / max_drawdown scorers and AutoTuner apply.

    Signal params (all tunable in a walkforward grid):
        hh_lookback, rvol_lookback, rvol_max, require_divergence, cmf_max,
        cmf_period
    Quality filters (default off → prior behavior):
        min_extension_atr : only fire when close is at least this many ATR above
            its ext_ema_span EMA (a real blow-off, not a marginal high).
        confirm_mode : require the NEXT bar to confirm the rollover before
            entering. "none" (default) = enter on the signal bar; "weak" =
            next close < signal close; "medium" = next close < signal low;
            "strong" = medium AND next-bar RVOL > 1 (sellers stepped in).
    Exit / cost params:
        stop_atr_multiple, target_atr_multiple, atr_length, cost_pct,
        borrow_cost_pct, reentry_cooldown_bars, min_reward_cost_ratio
    """
    hh_lookback = int(kwargs.pop("hh_lookback", 20))
    rvol_lookback = int(kwargs.pop("rvol_lookback", 20))
    rvol_max = float(kwargs.pop("rvol_max", 1.0))
    require_divergence = bool(kwargs.pop("require_divergence", True))
    cmf_max = kwargs.pop("cmf_max", None)
    cmf_max = None if cmf_max is None else float(cmf_max)
    cmf_period = int(kwargs.pop("cmf_period", 21))
    min_extension_atr = float(kwargs.pop("min_extension_atr", 0.0))
    ext_ema_span = int(kwargs.pop("ext_ema_span", 50))
    confirm_mode = str(kwargs.pop("confirm_mode", "none")).lower()
    stop_mult = float(kwargs.pop("stop_atr_multiple", STOP_ATR_MULTIPLE))
    target_mult = float(kwargs.pop("target_atr_multiple", TARGET_ATR_MULTIPLE))
    atr_length = int(kwargs.pop("atr_length", 14))
    cost_pct = float(kwargs.pop("cost_pct", SLIPPAGE_PCT))
    # Extra per-trade friction specific to shorting (locate/borrow fee + worse
    # short fills). Added on top of cost_pct so backtests can haircut shorts
    # realistically; default 0 leaves prior results unchanged.
    borrow_cost_pct = float(kwargs.pop("borrow_cost_pct", 0.0))
    cost_pct += borrow_cost_pct
    reentry_cooldown_bars = int(kwargs.pop("reentry_cooldown_bars", 0))
    min_reward_cost_ratio = float(kwargs.pop("min_reward_cost_ratio", 0.0))

    enriched = df.copy()
    atr_series = calculate_atr(enriched, atr_length)
    rvol_series = relative_volume(enriched["volume"], rvol_lookback)
    signal_series = exhaustion_short_signal(
        enriched,
        hh_lookback=hh_lookback,
        rvol_lookback=rvol_lookback,
        rvol_max=rvol_max,
        require_divergence=require_divergence,
        cmf_max=cmf_max,
        cmf_period=cmf_period,
    )

    # Extension gate: only fade a push that is genuinely stretched above trend.
    if min_extension_atr > 0:
        close_s = enriched["close"].astype(float)
        ema = close_s.ewm(span=ext_ema_span, adjust=False).mean()
        extended = (close_s - ema) >= (min_extension_atr * atr_series)
        signal_series = signal_series & extended.fillna(False)

    if "date" in enriched.columns:
        hour_ct, minute_ct = _ct_hour_minute(enriched["date"])
        hour_ct = pd.Series(np.asarray(hour_ct), index=enriched.index)
        minute_ct = pd.Series(np.asarray(minute_ct), index=enriched.index)
    elif isinstance(enriched.index, pd.DatetimeIndex):
        h, m = _ct_hour_minute(enriched.index)
        hour_ct = pd.Series(np.asarray(h), index=enriched.index)
        minute_ct = pd.Series(np.asarray(m), index=enriched.index)
    else:
        hour_ct = pd.Series(np.full(len(enriched), 10), index=enriched.index)
        minute_ct = pd.Series(np.zeros(len(enriched), dtype=int), index=enriched.index)

    enriched["signal"] = 0
    enriched["position"] = 0
    enriched["strategy_return"] = 0.0

    in_trade = False
    entry_price = stop_price = target_price = 0.0
    prev_close = None
    last_exit_idx = None
    # Confirmation: a candidate at bar i waits for bar i+1 to confirm the
    # rollover. pending_idx holds the unconfirmed candidate bar.
    pending_idx = None

    def _confirmed(sig_idx: int, cur_close: float, cur_rvol: float) -> bool:
        sig_low = float(enriched["low"].iloc[sig_idx])
        sig_close = float(enriched["close"].iloc[sig_idx])
        if confirm_mode == "weak":
            return cur_close < sig_close
        if confirm_mode == "medium":
            return cur_close < sig_low
        if confirm_mode == "strong":
            return cur_close < sig_low and (pd.notna(cur_rvol) and cur_rvol > 1.0)
        return True

    for i in range(len(enriched)):
        row = enriched.iloc[i]
        close = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        h = int(hour_ct.iloc[i])
        m = int(minute_ct.iloc[i])

        bar_return = 0.0

        if in_trade:
            eod = (h > EOD_LIQUIDATION_HOUR_CT) or (
                h == EOD_LIQUIDATION_HOUR_CT and m >= EOD_LIQUIDATION_MINUTE_CT
            )
            # On a short the adverse move is UP: check the stop (above) first.
            exit_price = None
            if high >= stop_price:
                exit_price = stop_price
            elif low <= target_price:
                exit_price = target_price
            elif eod:
                exit_price = close

            if exit_price is not None:
                # Short P&L: profit when exit is below entry. Friction per trip.
                bar_return = (entry_price / exit_price - 1.0) - cost_pct
                in_trade = False
                last_exit_idx = i
                enriched.at[enriched.index[i], "position"] = 0
            else:
                # Held: short gains when price falls bar-over-bar.
                bar_return = (prev_close / close - 1.0) if prev_close else 0.0
                enriched.at[enriched.index[i], "position"] = -1
        else:
            morning_blackout = h < MORNING_BLACKOUT_HOUR_CT
            eod = (h > EOD_LIQUIDATION_HOUR_CT) or (
                h == EOD_LIQUIDATION_HOUR_CT and m >= EOD_LIQUIDATION_MINUTE_CT
            )
            cooldown = in_cooldown(i, last_exit_idx, reentry_cooldown_bars)
            tradeable = not morning_blackout and not eod and not cooldown

            # Decide whether THIS bar opens a short: either a candidate that needs
            # no confirmation, or a prior candidate confirmed by this bar.
            enter = False
            if tradeable:
                if confirm_mode == "none":
                    enter = bool(signal_series.iloc[i])
                elif pending_idx is not None and i == pending_idx + 1:
                    enter = _confirmed(pending_idx, close, float(rvol_series.iloc[i]))

            # A candidate fired on this bar but is awaiting next-bar confirmation.
            if confirm_mode != "none" and bool(signal_series.iloc[i]):
                pending_idx = i
            elif confirm_mode != "none" and pending_idx is not None and i > pending_idx:
                pending_idx = None   # candidate expires if not confirmed next bar

            if enter:
                atr = float(atr_series.iloc[i]) if pd.notna(atr_series.iloc[i]) else 0.0
                atr = max(atr, close * ATR_FLOOR_PCT)
                entry_price = close
                stop_price = round(entry_price + atr * stop_mult, 4)     # above
                target_price = round(entry_price - atr * target_mult, 4)  # below
                # Short reward clears cost? reward = (entry - target) / entry.
                reward_pct = (entry_price - target_price) / entry_price if entry_price else 0.0
                cost_ok = min_reward_cost_ratio <= 0 or reward_pct >= cost_pct * min_reward_cost_ratio
                if target_price < entry_price < stop_price and target_price > 0 and cost_ok:
                    in_trade = True
                    pending_idx = None
                    enriched.at[enriched.index[i], "signal"] = -1
                    enriched.at[enriched.index[i], "position"] = -1

        enriched.at[enriched.index[i], "strategy_return"] = bar_return
        prev_close = close

    enriched["market_return"] = enriched["close"].pct_change()
    enriched["pnl"] = enriched["strategy_return"]
    return enriched


def evaluate_fixed_params(df, params, n_splits=3):
    """Out-of-sample metrics for ONE fixed param set (no per-symbol tuning).

    Scores a symbol the way the live bot actually trades it — with the shared,
    aggregated params — rather than with its own bespoke best params. This is
    what the allowlist should gate on: a name can win under its own optimum yet
    lose under the median params every symbol is forced to use.
    """
    from sklearn.model_selection import TimeSeriesSplit

    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics = []
    for fold, (_train_idx, test_idx) in enumerate(tscv.split(df)):
        res = mean_reversion_strategy(df.iloc[test_idx], **params)
        metrics.append({
            "fold": fold + 1,
            "profit_factor": profit_factor(res),
            "sharpe": sharpe_ratio(res, periods_per_year=PERIODS_PER_YEAR_5MIN),
            "max_drawdown": max_drawdown(res),
        })
    return metrics


def run_walkforward_backtest(df, param_grid=None, n_splits=5):
    # Task 4: focus the grid on deep-oversold RI + VWAP extension, the dimensions
    # that matter for fast 3x leveraged assets (Bollinger width is too lagging).
    if param_grid is None:
        param_grid = {
            "ri_threshold": [-1.0, -0.5, 0.0, 0.5],
            "use_vwap_filter": [True, False],
            "max_vwap_extension_pct": [0.012, 0.020, 0.030],
            "rsi_max": [40.0, 48.0, 55.0],
            "band_length": [20],
        }
    tuner = AutoTuner(mean_reversion_strategy, df, param_grid)
    best_params, best_score = tuner.tune(profit_factor, n_splits=n_splits)
    print(f"Best Params: {best_params}, Best Profit Factor: {best_score:.2f}")
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []
    oos_metrics = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
        test = df.iloc[test_idx]
        res = mean_reversion_strategy(test, **best_params)
        results.append(res)
        pf = profit_factor(res)
        sr = sharpe_ratio(res, periods_per_year=PERIODS_PER_YEAR_5MIN)
        dd = max_drawdown(res)
        oos_metrics.append({"fold": fold + 1, "profit_factor": pf, "sharpe": sr, "max_drawdown": dd})
        print(f"Fold {fold + 1}: PF={pf:.2f}, Sharpe={sr:.2f}, MaxDD={dd:.4f}")
    if oos_metrics:
        avg_pf = np.mean([m["profit_factor"] for m in oos_metrics])
        avg_sr = np.mean([m["sharpe"] for m in oos_metrics])
        avg_dd = np.mean([m["max_drawdown"] for m in oos_metrics])
        print(f"\nOut-of-sample summary: Avg PF={avg_pf:.2f}, Avg Sharpe={avg_sr:.2f}, Avg MaxDD={avg_dd:.4f}")
    return results, oos_metrics, best_params


def run_exhaustion_walkforward(df, param_grid=None, n_splits=5, verbose=True, fixed_kwargs=None):
    """Walk-forward the short-exhaustion strategy: tune in-sample, score OOS.

    Same protocol as run_walkforward_backtest but drives short_exhaustion_strategy
    so the higher-high-on-fading-RVOL edge can be validated per symbol across the
    universe before it's ever wired live.

    fixed_kwargs: strategy params held CONSTANT across the whole sweep (e.g. a
    borrow-cost haircut). Folded into the grid as single-value lists so they flow
    through both the tuner and the OOS scoring without expanding the search.
    """
    from sklearn.model_selection import TimeSeriesSplit

    if param_grid is None:
        param_grid = {
            "rvol_max": [0.8, 1.0, 1.2],
            "hh_lookback": [10, 20],
            "require_divergence": [True, False],
            "target_atr_multiple": [2.0, 3.0],
        }
    if fixed_kwargs:
        param_grid = {**param_grid, **{k: [v] for k, v in fixed_kwargs.items()}}
    tuner = AutoTuner(short_exhaustion_strategy, df, param_grid)
    best_params, best_score = tuner.tune(profit_factor, n_splits=n_splits)
    if verbose:
        print(f"Best Params: {best_params}, Best Profit Factor: {best_score:.2f}")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    results, oos_metrics = [], []
    for fold, (_train_idx, test_idx) in enumerate(tscv.split(df)):
        res = short_exhaustion_strategy(df.iloc[test_idx], **best_params)
        results.append(res)
        pf = profit_factor(res)
        sr = sharpe_ratio(res, periods_per_year=PERIODS_PER_YEAR_5MIN)
        dd = max_drawdown(res)
        oos_metrics.append({"fold": fold + 1, "profit_factor": pf, "sharpe": sr, "max_drawdown": dd})
        if verbose:
            print(f"Fold {fold + 1}: PF={pf:.2f}, Sharpe={sr:.2f}, MaxDD={dd:.4f}")
    if verbose and oos_metrics:
        avg_pf = np.mean([m["profit_factor"] for m in oos_metrics])
        avg_sr = np.mean([m["sharpe"] for m in oos_metrics])
        avg_dd = np.mean([m["max_drawdown"] for m in oos_metrics])
        print(f"\nOut-of-sample summary: Avg PF={avg_pf:.2f}, Avg Sharpe={avg_sr:.2f}, Avg MaxDD={avg_dd:.4f}")
    return results, oos_metrics, best_params
