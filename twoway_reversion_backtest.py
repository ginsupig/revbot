"""Two-way mean-reversion backtest: long oversold dips + short overbought rips.

Exercises the full engine with the tightened AND-gated settings on real Alpaca
data. Two non-overlapping windows (research discipline), median exits plus
symmetric ATR stops/trailing stops mirroring the live bot, cost modeled as round-trip
slippage. Reports per-symbol and aggregate metrics for both sides.
"""
from __future__ import annotations
import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from reversion_bot.engine import ReversionEngine
from reversion_bot.config import ReversionConfig
from reversion_bot.analytics import profit_factor, sharpe_ratio, max_drawdown
from reversion_bot.indicators import calculate_atr
from run_real_backtest import fetch_alpaca_bars

# --- Constants mirroring live risk.py ---
STOP_ATR_MULT = 1.20
TRAIL_ATR_MULT = 1.50
ATR_FLOOR_PCT = 0.0035
COST_PCT = 0.0010          # 10 bps round-trip (slippage + spread)
SHORT_EXTRA_COST = 0.0005  # 5 bps extra for borrow/locate on shorts

# --- Universe: validated neutral large-caps ---
UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMD", "META", "GOOGL", "AMZN", "TSLA",
    "JPM", "BAC", "GS", "UNH", "JNJ", "XOM", "CVX", "HD",
    "PG", "KO", "CAT", "BA", "FCX", "CLF", "OXY", "NEE",
]

# Two non-overlapping windows (research discipline)
WINDOW_1 = ("2024-06-01", "2025-06-01")   # 12 months
WINDOW_2 = ("2025-06-01", "2026-06-01")   # 12 months (OOS)


def fetch_bars(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    try:
        bars = fetch_alpaca_bars(symbol, start, end, "1Day")
        if bars is None or len(bars) < 60:
            return None
        return bars
    except Exception as e:
        print(f"  [SKIP] {symbol}: {e}")
        return None


def _risk_off_dates(spy: pd.DataFrame, ema_length: int = 50) -> set:
    close = spy["close"].astype(float)
    ema = close.ewm(span=ema_length, adjust=False).mean()
    decisive = close < ema * 0.995
    confirmed = (close < ema).rolling(3).sum() >= 3
    dates = pd.to_datetime(spy["date"], utc=True).dt.date
    return set(dates[decisive | confirmed])


def simulate_twoway(df: pd.DataFrame, engine: ReversionEngine,
                    cost_pct: float = COST_PCT, risk_off_dates: set | None = None):
    """Simulate long and short reversion entries to the rolling center."""
    enriched = engine.calculate_indicators(df)
    n = len(enriched)
    close = enriched["close"].to_numpy(float)
    high = enriched["high"].to_numpy(float)
    low = enriched["low"].to_numpy(float)
    atr = enriched["atr"].to_numpy(float)
    sma = enriched["sma"].to_numpy(float)
    dates = pd.to_datetime(enriched["date"], utc=True).dt.date.to_numpy()

    returns = np.zeros(n)
    sides = np.zeros(n)     # +1 long, -1 short
    in_trade = False
    entry_price = stop_price = target_price = 0.0
    direction = 0
    high_water = low_water = 0.0
    trail_distance = 0.0
    trades = []

    for i in range(1, n):
        if in_trade:
            exit_price = None
            if direction == 1:  # long
                high_water = max(high_water, high[i])
                stop_price = max(stop_price, high_water - trail_distance)
                if low[i] <= stop_price:
                    exit_price = stop_price
                elif np.isfinite(sma[i]) and high[i] >= sma[i]:
                    exit_price = sma[i]
            else:  # short
                low_water = min(low_water, low[i])
                stop_price = min(stop_price, low_water + trail_distance)
                if high[i] >= stop_price:
                    exit_price = stop_price
                elif np.isfinite(sma[i]) and low[i] <= sma[i]:
                    exit_price = sma[i]

            if exit_price is not None:
                if direction == 1:
                    ret = (exit_price / entry_price - 1.0) - cost_pct
                else:
                    ret = (1.0 - exit_price / entry_price) - cost_pct - SHORT_EXTRA_COST
                returns[i] = ret
                trades.append({"side": direction, "ret": ret})
                in_trade = False
                direction = 0
            else:
                sides[i] = direction

        if not in_trade:
            # Need at least 2 rows for the engine to check prev bar
            if i < 1:
                continue
            window = enriched.iloc[max(0, i - 1): i + 1]
            dec = engine.get_decision(window)

            if dec.signal in ("LONG_REVERSION", "SHORT_REVERSION"):
                if dec.signal == "SHORT_REVERSION" and dates[i] not in (risk_off_dates or set()):
                    continue
                a = float(atr[i]) if np.isfinite(atr[i]) else 0.0
                a = max(a, close[i] * ATR_FLOOR_PCT)

                if dec.signal == "LONG_REVERSION":
                    entry_price = close[i]
                    stop_price = round(entry_price - a * STOP_ATR_MULT, 4)
                    target_price = round(float(dec.sma or 0.0), 4)
                    direction = 1
                    high_water = high[i]
                else:  # SHORT_REVERSION
                    entry_price = close[i]
                    stop_price = round(entry_price + a * STOP_ATR_MULT, 4)
                    target_price = round(float(dec.sma or 0.0), 4)
                    direction = -1
                    low_water = low[i]

                trail_distance = a * TRAIL_ATR_MULT

                if direction == 1 and stop_price < entry_price < target_price:
                    in_trade = True
                    sides[i] = direction
                elif direction == -1 and target_price < entry_price < stop_price and target_price > 0:
                    in_trade = True
                    sides[i] = direction
                else:
                    direction = 0

    result = pd.DataFrame({
        "strategy_return": returns,
        "pnl": returns,
        "side": sides,
    })
    return result, trades


def score(result_df, trades, label=""):
    """Compute and print metrics."""
    pf = profit_factor(result_df)
    sr = sharpe_ratio(result_df)
    dd = max_drawdown(result_df)
    n_trades = len(trades)
    long_trades = [t for t in trades if t["side"] == 1]
    short_trades = [t for t in trades if t["side"] == -1]
    n_long = len(long_trades)
    n_short = len(short_trades)
    win_rate = (sum(1 for t in trades if t["ret"] > 0) / n_trades * 100) if n_trades else 0
    avg_ret = (np.mean([t["ret"] for t in trades]) * 100) if n_trades else 0
    long_wr = (sum(1 for t in long_trades if t["ret"] > 0) / n_long * 100) if n_long else 0
    short_wr = (sum(1 for t in short_trades if t["ret"] > 0) / n_short * 100) if n_short else 0
    long_avg = (np.mean([t["ret"] for t in long_trades]) * 100) if n_long else 0
    short_avg = (np.mean([t["ret"] for t in short_trades]) * 100) if n_short else 0

    return {
        "label": label, "pf": pf, "sharpe": sr, "maxdd": dd,
        "trades": n_trades, "longs": n_long, "shorts": n_short,
        "win%": win_rate, "avg_ret%": avg_ret,
        "long_wr%": long_wr, "short_wr%": short_wr,
        "long_avg%": long_avg, "short_avg%": short_avg,
    }


def run_window(symbols, start, end, engine, window_name):
    """Run the backtest on all symbols for one window."""
    print(f"\n{'='*72}")
    print(f"  {window_name}: {start} -> {end}")
    print(f"{'='*72}")

    all_returns = []
    all_trades = []
    per_symbol = []
    spy = fetch_bars("SPY", start, end)
    if spy is None:
        print("  No SPY data; refusing to simulate shorts without regime state.")
        return None, None
    risk_off_dates = _risk_off_dates(spy)

    for sym in symbols:
        bars = fetch_bars(sym, start, end)
        if bars is None:
            continue
        result, trades = simulate_twoway(bars, engine, risk_off_dates=risk_off_dates)
        all_returns.append(result)
        all_trades.extend([{**t, "symbol": sym} for t in trades])
        if trades:
            s = score(result, trades, sym)
            per_symbol.append(s)

    if not all_returns:
        print("  No data fetched. Check API credentials.")
        return None, None

    # Aggregate
    agg = pd.concat(all_returns, ignore_index=True)
    agg_score = score(agg, all_trades, f"AGGREGATE ({window_name})")

    # Print per-symbol table
    print(f"\n{'sym':<8}{'PF':>6}{'Sharpe':>8}{'trades':>7}{'L':>4}{'S':>4}{'win%':>7}{'avg%':>7}{'L_wr%':>7}{'S_wr%':>7}")
    print("-" * 72)
    for s in sorted(per_symbol, key=lambda x: x["pf"], reverse=True):
        print(f"{s['label']:<8}{s['pf']:6.2f}{s['sharpe']:8.2f}{s['trades']:7d}"
              f"{s['longs']:4d}{s['shorts']:4d}{s['win%']:7.1f}{s['avg_ret%']:7.2f}"
              f"{s['long_wr%']:7.1f}{s['short_wr%']:7.1f}")

    # Print aggregate
    a = agg_score
    print(f"\n--- {a['label']} ---")
    print(f"  Profit Factor : {a['pf']:.2f}")
    print(f"  Sharpe Ratio  : {a['sharpe']:.2f}")
    print(f"  Max Drawdown  : {a['maxdd']:.4f}")
    print(f"  Total Trades  : {a['trades']} (L={a['longs']}, S={a['shorts']})")
    print(f"  Win Rate      : {a['win%']:.1f}%")
    print(f"  Avg Trade     : {a['avg_ret%']:.3f}%")
    print(f"  Long  WR/Avg  : {a['long_wr%']:.1f}% / {a['long_avg%']:.3f}%")
    print(f"  Short WR/Avg  : {a['short_wr%']:.1f}% / {a['short_avg%']:.3f}%")

    return agg_score, per_symbol


def main():
    load_dotenv()

    configs = {
        "LONG-ONLY (tightened)": ReversionConfig(
            min_history=2, min_price=0.0, min_dollar_volume=0.0,
            rsi_max=35.0, ri_threshold=-0.5,
            enable_shorts=False,
            require_reclaim_lb1=True, require_bullish_close=True,
            use_trend_filter=False, use_vwap_filter=False,
        ),
        "TWO-WAY (live gates, shorts opt-in)": ReversionConfig(
            min_history=2, min_price=0.0, min_dollar_volume=0.0,
            rsi_max=40.0, ri_threshold=-0.5,
            rsi_min=65.0, ri_short_threshold=0.5,
            enable_shorts=True,
            require_reclaim_lb1=False, require_bullish_close=False,
            require_short_reject=True, require_short_bearish_close=True,
            short_trend_filter_band_pct=0.0,
            use_trend_filter=True, use_vwap_filter=False,
        ),
        "LONG-ONLY (old loose OR)": ReversionConfig(
            min_history=2, min_price=0.0, min_dollar_volume=0.0,
            rsi_max=48.0, ri_threshold=-0.5,
            enable_shorts=False,
            require_reclaim_lb1=False, require_bullish_close=False,
            use_trend_filter=False, use_vwap_filter=False,
        ),
    }

    for name, cfg in configs.items():
        engine = ReversionEngine(cfg)
        print(f"\n{'#'*72}")
        print(f"  CONFIG: {name}")
        print(f"  RSI long<={cfg.rsi_max}, shorts={'ON' if cfg.enable_shorts else 'OFF'}")
        if cfg.enable_shorts:
            print(f"  RSI short>={cfg.rsi_min}, RI short>={cfg.ri_short_threshold}")
        print(f"  Confirmations: reclaim={cfg.require_reclaim_lb1}, bullish_close={cfg.require_bullish_close}")
        print(f"  Exits: stop={STOP_ATR_MULT}xATR, trail={TRAIL_ATR_MULT}xATR, target=rolling SMA{cfg.band_length}")
        print(f"{'#'*72}")

        w1_score, _ = run_window(UNIVERSE, *WINDOW_1, engine, f"{name} W1 (IS)")
        w2_score, _ = run_window(UNIVERSE, *WINDOW_2, engine, f"{name} W2 (OOS)")

        if w1_score and w2_score:
            print(f"\n  {'':>20}{'W1 (IS)':>12}{'W2 (OOS)':>12}{'delta':>10}")
            for key in ["pf", "sharpe", "win%", "avg_ret%", "trades", "longs", "shorts"]:
                v1, v2 = w1_score[key], w2_score[key]
                fmt = ".2f" if isinstance(v1, float) else "d"
                d = v2 - v1
                print(f"  {key:>18}{v1:>12{fmt}}{v2:>12{fmt}}{d:>+10{fmt}}")

            if w2_score["pf"] >= 1.10 and w2_score["sharpe"] > 0:
                print(f"\n  VERDICT ({name}): OOS PF >= 1.10. Edge holds.")
            elif w2_score["pf"] >= 1.0:
                print(f"\n  VERDICT ({name}): OOS PF ~1.0. Marginal.")
            else:
                print(f"\n  VERDICT ({name}): OOS PF < 1.0. No durable edge.")


if __name__ == "__main__":
    main()
