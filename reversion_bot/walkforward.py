import numpy as np
import pandas as pd
from reversion_bot.engine import ReversionEngine
from reversion_bot.config import ReversionConfig
from reversion_bot.autotune import AutoTuner
from reversion_bot.analytics import profit_factor, sharpe_ratio, max_drawdown

# --- Live-execution constants (mirror risk.py / main.py) ---
STOP_ATR_MULTIPLE = 1.20        # risk.py RiskConfig.stop_atr_multiple
TARGET_ATR_MULTIPLE = 2.00      # risk.py RiskConfig.target_atr_multiple
ATR_FLOOR_PCT = 0.0035          # risk.py RiskConfig.atr_floor_pct
SLIPPAGE_PCT = 0.0008           # 8 bps round-trip friction on leveraged ETFs
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
    """
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
                bar_return = (exit_price / entry_price - 1.0) - SLIPPAGE_PCT
                in_trade = False
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
            if not morning_blackout and not eod:
                dec = engine.get_decision(enriched.iloc[: i + 1])
                if dec.signal == "LONG_REVERSION":
                    atr = float(dec.atr or 0.0)
                    atr = max(atr, close * ATR_FLOOR_PCT)   # ATR floor (risk.py)
                    entry_price = close
                    stop_price = round(entry_price - atr * STOP_ATR_MULTIPLE, 4)
                    target_price = round(entry_price + atr * TARGET_ATR_MULTIPLE, 4)
                    if stop_price < entry_price < target_price:
                        in_trade = True
                        enriched.at[enriched.index[i], "signal"] = 1
                        enriched.at[enriched.index[i], "position"] = 1

        enriched.at[enriched.index[i], "strategy_return"] = bar_return
        prev_close = close

    enriched["market_return"] = enriched["close"].pct_change()
    enriched["pnl"] = enriched["strategy_return"]
    return enriched


def run_walkforward_backtest(df, param_grid=None, n_splits=5):
    # Task 4: focus the grid on deep-oversold RI + VWAP extension, the dimensions
    # that matter for fast 3x leveraged assets (Bollinger width is too lagging).
    if param_grid is None:
        param_grid = {
            "ri_threshold": [-1.5, -1.0, -0.75, -0.5],
            "use_vwap_filter": [True],
            "max_vwap_extension_pct": [0.008, 0.012, 0.018, 0.025],
            "rsi_max": [30.0, 40.0],
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
        sr = sharpe_ratio(res)
        dd = max_drawdown(res)
        oos_metrics.append({"fold": fold + 1, "profit_factor": pf, "sharpe": sr, "max_drawdown": dd})
        print(f"Fold {fold + 1}: PF={pf:.2f}, Sharpe={sr:.2f}, MaxDD={dd:.4f}")
    if oos_metrics:
        avg_pf = np.mean([m["profit_factor"] for m in oos_metrics])
        avg_sr = np.mean([m["sharpe"] for m in oos_metrics])
        avg_dd = np.mean([m["max_drawdown"] for m in oos_metrics])
        print(f"\nOut-of-sample summary: Avg PF={avg_pf:.2f}, Avg Sharpe={avg_sr:.2f}, Avg MaxDD={avg_dd:.4f}")
    return results, oos_metrics, best_params
