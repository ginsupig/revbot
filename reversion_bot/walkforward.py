import numpy as np
import pandas as pd
from reversion_bot.engine import ReversionEngine
from reversion_bot.config import ReversionConfig
from reversion_bot.autotune import AutoTuner
from reversion_bot.analytics import profit_factor, sharpe_ratio, max_drawdown

def mean_reversion_strategy(df, **kwargs):
    # Set loose defaults, but allow override from kwargs
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
        use_trend_filter=False
    )
    config_defaults.update(kwargs)
    engine = ReversionEngine(ReversionConfig(**config_defaults))
    enriched = engine.calculate_indicators(df)
    # Simple signal: 1 if LONG_REVERSION, else 0
    enriched['signal'] = 0
    enriched['position'] = 0

    in_trade = False
    entry_price = None

    for i in range(len(enriched)):
        sub = enriched.iloc[:i+1]
        row = enriched.iloc[i]
        close = float(row['close'])
        sma = float(row['sma']) if pd.notna(row['sma']) else None
        lb2 = float(row['lb2']) if pd.notna(row['lb2']) else None

        if in_trade:
            # Exit when price reverts to SMA or stop-loss triggers below lb2
            stop_hit = lb2 is not None and close < lb2
            mean_reclaimed = sma is not None and close >= sma
            if mean_reclaimed or stop_hit:
                in_trade = False
                entry_price = None
        else:
            dec = engine.get_decision(sub)
            if dec.signal == 'LONG_REVERSION':
                enriched.at[enriched.index[i], 'signal'] = 1
                in_trade = True
                entry_price = close

        enriched.at[enriched.index[i], 'position'] = 1 if in_trade else 0

    enriched['market_return'] = enriched['close'].pct_change()
    enriched['strategy_return'] = enriched['market_return'] * enriched['position'].shift(1)
    enriched['pnl'] = enriched['strategy_return']
    return enriched

def run_walkforward_backtest(df, param_grid=None, n_splits=5):
    if param_grid is None:
        param_grid = {'band_length': [20], 'band_std_1': [1.0], 'band_std_2': [2.0]}
    tuner = AutoTuner(mean_reversion_strategy, df, param_grid)
    best_params, best_score = tuner.tune(profit_factor, n_splits=n_splits)
    print(f"Best Params: {best_params}, Best Profit Factor: {best_score:.2f}")
    # Run walkforward with best params
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []
    oos_metrics = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        res = mean_reversion_strategy(test, **best_params)
        results.append(res)
        pf = profit_factor(res)
        sr = sharpe_ratio(res)
        dd = max_drawdown(res)
        oos_metrics.append({'fold': fold+1, 'profit_factor': pf, 'sharpe': sr, 'max_drawdown': dd})
        print(f"Fold {fold+1}: PF={pf:.2f}, Sharpe={sr:.2f}, MaxDD={dd:.4f}")
    # Out-of-sample summary
    if oos_metrics:
        avg_pf = np.mean([m['profit_factor'] for m in oos_metrics])
        avg_sr = np.mean([m['sharpe'] for m in oos_metrics])
        avg_dd = np.mean([m['max_drawdown'] for m in oos_metrics])
        print(f"\nOut-of-sample summary: Avg PF={avg_pf:.2f}, Avg Sharpe={avg_sr:.2f}, Avg MaxDD={avg_dd:.4f}")
    return results, oos_metrics, best_params

# Example usage:
# df = pd.read_csv('your_data.csv')
# run_walkforward_backtest(df)
