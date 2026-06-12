"""Failed-breakout (fakeout) fade strategy — corrected implementation.

WHAT THE OLD VERSION GOT WRONG
The original 'confirmation' required the bars BEFORE a breakout to have closed
below the rolling max of highs — true ~100% of the time (a close almost never
exceeds the running max of highs). Every breakout was therefore labeled a
'fakeout' on its own breakout bar, so the model faded every breakout and sat
net short through clean uptrends (see tests/test_trendfail_vacuous_confirmation).

WHAT A FAKEOUT ACTUALLY IS
A breakout that FAILS: price pushes through a prior extreme, then falls back
through that level shortly after. The information arrives only when the
failure happens — so the signal must fire on the failure bar, never on the
breakout bar.

TIMING / NO-LOOKAHEAD CONTRACT
- Breakout detected at bar b against the rolling extreme of bars [b-window, b-1]
  (shift(1), so the breakout bar never raises its own level).
- The breakout LEVEL is frozen at bar b.
- If within the next `confirmation_window` bars a close crosses back through
  the frozen level, the fakeout signal prints on THAT bar (-1 after an upside
  failure, +1 after a downside failure).
- If the move holds through the window, the breakout is treated as genuine and
  no fade is taken. Past signals are never revised by future data
  (tested in test_trendfail_fixed.py::test_no_lookahead).

Output columns are a superset of the old version (signal / position /
strategy_return / pnl preserved) so service.py and walkforward.py work
unchanged.
"""
import numpy as np
import pandas as pd


def trend_fail_continue_strategy(
    data,
    window=20,
    confirmation_window=3,
    threshold=0.005,
    stop_loss_pct=0.01,
    take_profit_pct=0.02,
    max_hold_bars=None,
):
    data = data.copy()
    close = data["close"].astype(float)
    high = data["high"].astype(float)
    low = data["low"].astype(float)

    # Prior-bar rolling extremes: bar i's level is built from bars [i-window, i-1].
    high_roll = high.rolling(window=window, min_periods=window).max().shift(1)
    low_roll = low.rolling(window=window, min_periods=window).min().shift(1)
    data["high_roll"] = high_roll
    data["low_roll"] = low_roll
    data["breakout_up"] = close > high_roll * (1.0 + threshold)
    data["breakout_down"] = close < low_roll * (1.0 - threshold)

    n = len(data)
    fakeout_up = np.zeros(n, dtype=bool)    # upside breakout FAILED -> short
    fakeout_down = np.zeros(n, dtype=bool)  # downside breakout FAILED -> long

    closes = close.to_numpy()
    b_up = data["breakout_up"].fillna(False).to_numpy()
    b_dn = data["breakout_down"].fillna(False).to_numpy()
    hr = high_roll.to_numpy()
    lr = low_roll.to_numpy()

    # Track at most one pending breakout per direction (newest wins).
    pend_up_level = pend_dn_level = None
    pend_up_expiry = pend_dn_expiry = -1

    for i in range(n):
        c = closes[i]

        # 1) Resolve pending breakouts FIRST using bar i's close.
        if pend_up_level is not None:
            if c < pend_up_level:            # fell back through the level: FAKEOUT
                fakeout_up[i] = True
                pend_up_level = None
            elif i >= pend_up_expiry:        # held the level: genuine breakout
                pend_up_level = None
        if pend_dn_level is not None:
            if c > pend_dn_level:
                fakeout_down[i] = True
                pend_dn_level = None
            elif i >= pend_dn_expiry:
                pend_dn_level = None

        # 2) Then register any NEW breakout on bar i (its failure can only be
        #    observed from bar i+1 on, so this ordering enforces no-lookahead).
        if b_up[i] and np.isfinite(hr[i]):
            pend_up_level = hr[i]
            pend_up_expiry = i + confirmation_window
        if b_dn[i] and np.isfinite(lr[i]):
            pend_dn_level = lr[i]
            pend_dn_expiry = i + confirmation_window

    data["fakeout_up"] = fakeout_up
    data["fakeout_down"] = fakeout_down

    data["signal"] = 0
    data.loc[data["fakeout_up"], "signal"] = -1
    data.loc[data["fakeout_down"], "signal"] = 1

    # Position management: enter on the signal bar's close, then bracket-managed
    # (stop/target/optional time stop) instead of the old hold-forever ffill.
    max_hold = int(max_hold_bars) if max_hold_bars else confirmation_window * 4
    position = np.zeros(n)
    strategy_return = np.zeros(n)
    sig = data["signal"].to_numpy()
    highs = high.to_numpy()
    lows = low.to_numpy()

    pos = 0
    entry = 0.0
    bars_held = 0
    for i in range(1, n):
        if pos != 0:
            bars_held += 1
            price = closes[i]
            ret = (price - entry) / entry * pos
            exited = False
            if pos > 0:
                sl, tp = entry * (1 - stop_loss_pct), entry * (1 + take_profit_pct)
                if lows[i] <= sl:
                    ret, exited = -stop_loss_pct, True      # stop first: conservative
                elif highs[i] >= tp:
                    ret, exited = take_profit_pct, True
            else:
                sl, tp = entry * (1 + stop_loss_pct), entry * (1 - take_profit_pct)
                if highs[i] >= sl:
                    ret, exited = -stop_loss_pct, True
                elif lows[i] <= tp:
                    ret, exited = take_profit_pct, True
            if not exited and bars_held >= max_hold:
                exited = True                                # time stop at close
            strategy_return[i] = ret if exited else 0.0
            if not exited:
                # mark-to-market accrues only at exit; carry the open position
                strategy_return[i] = 0.0
                position[i] = pos
            else:
                pos = 0
                position[i] = 0
        if pos == 0 and sig[i] != 0:
            pos = int(sig[i])
            entry = closes[i]
            bars_held = 0
            position[i] = pos

    data["position"] = position
    data["market_return"] = close.pct_change()
    data["strategy_return"] = strategy_return
    data["pnl"] = data["strategy_return"]
    return data
