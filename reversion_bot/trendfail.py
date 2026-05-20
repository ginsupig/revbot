import pandas as pd
import numpy as np

def trend_fail_continue_strategy(data, window=20, confirmation_window=3, threshold=0.005):
    # Ensure no lookahead bias: all calculations use only past data
    data = data.copy()
    data['high_roll'] = data['high'].rolling(window=window).max()
    data['low_roll'] = data['low'].rolling(window=window).min()
    data['breakout_up'] = (data['close'] > data['high_roll'].shift(1) * (1 + threshold))
    data['breakout_down'] = (data['close'] < data['low_roll'].shift(1) * (1 - threshold))
    data['fakeout_up'] = False
    data['fakeout_down'] = False
    for i in range(window + confirmation_window, len(data)):
        if data['breakout_up'].iloc[i]:
            confirm = all(data['close'].iloc[i-j] < data['high_roll'].iloc[i-j] for j in range(1, confirmation_window+1))
            if confirm:
                data.at[data.index[i], 'fakeout_up'] = True
        if data['breakout_down'].iloc[i]:
            confirm = all(data['close'].iloc[i-j] > data['low_roll'].iloc[i-j] for j in range(1, confirmation_window+1))
            if confirm:
                data.at[data.index[i], 'fakeout_down'] = True
    data['signal'] = 0
    data.loc[data['fakeout_up'], 'signal'] = -1
    data.loc[data['fakeout_down'], 'signal'] = 1
    data['position'] = data['signal'].replace(0, np.nan).ffill().fillna(0)
    data['market_return'] = data['close'].pct_change()
    stop_loss_pct = 0.01
    take_profit_pct = 0.02
    data['strategy_return'] = 0.0
    for i in range(1, len(data)):
        pos = data['position'].iloc[i-1]
        if pos == 0:
            continue
        entry = data['close'].iloc[i-1]
        price = data['close'].iloc[i]
        ret = (price - entry) / entry * pos
        if pos > 0:
            sl = entry * (1 - stop_loss_pct)
            tp = entry * (1 + take_profit_pct)
            if data['low'].iloc[i] <= sl:
                ret = -stop_loss_pct
            elif data['high'].iloc[i] >= tp:
                ret = take_profit_pct
        elif pos < 0:
            sl = entry * (1 + stop_loss_pct)
            tp = entry * (1 - take_profit_pct)
            if data['high'].iloc[i] >= sl:
                ret = -stop_loss_pct
            elif data['low'].iloc[i] <= tp:
                ret = take_profit_pct
        data.at[data.index[i], 'strategy_return'] = ret
    data['pnl'] = data['strategy_return']
    return data
