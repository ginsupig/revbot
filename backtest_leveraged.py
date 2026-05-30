"""Offline backtest of the leveraged-ETF universe using free Yahoo Finance data.

Runs the repo's mean-reversion signal (reversion_bot.walkforward.mean_reversion_strategy)
over each ticker and reports per-ticker + equal-weight portfolio metrics. No Alpaca
credentials required. Daily bars are a proxy for the bot's intended 5Min intraday
signal, so treat these as a directional read on edge, not exact live behavior.
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

from reversion_bot.walkforward import mean_reversion_strategy
from reversion_bot.analytics import profit_factor, sharpe_ratio, max_drawdown

UNIVERSE = ["TQQQ", "SOXL", "TECL", "NVDL", "TSLL", "AAPU", "METU", "GGLL", "MSFU", "AMZU"]

# Strategy params mirror the bot's ReversionConfig defaults.
PARAMS = dict(band_length=20, band_std_1=1.0, band_std_2=2.0, rsi_max=48.0, adx_max=40.0)


def load(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame | None:
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    if df is None or len(df) == 0:
        return None
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    df = df.reset_index().rename(columns={"Date": "date", "Datetime": "date"})
    df = df[["date", "open", "high", "low", "close", "volume"]].dropna().reset_index(drop=True)
    return df


def trade_stats(res: pd.DataFrame) -> tuple[int, float]:
    """Count entries and win rate over completed holding stretches."""
    pos = res["position"].fillna(0).to_numpy()
    sig = (res["signal"] == 1).to_numpy()
    entries = int(sig.sum())
    # Win rate by entry-to-exit blocks
    rets, cur, holding = [], 0.0, False
    sr = res["strategy_return"].fillna(0).to_numpy()
    for i in range(len(res)):
        if pos[i] == 1:
            cur += sr[i]; holding = True
        elif holding:
            rets.append(cur); cur = 0.0; holding = False
    if holding:
        rets.append(cur)
    win = (np.mean([r > 0 for r in rets]) * 100) if rets else float("nan")
    return entries, win


def run(period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    rows, curves = [], {}
    for t in UNIVERSE:
        df = load(t, period, interval)
        if df is None or len(df) < 60:
            rows.append(dict(ticker=t, bars=0 if df is None else len(df), note="insufficient data"))
            continue
        res = mean_reversion_strategy(df, **PARAMS)
        res["strategy_return"] = res["strategy_return"].fillna(0)
        eq = (1 + res["strategy_return"]).cumprod()
        curves[t] = pd.Series(eq.to_numpy(), index=df["date"])
        bh = df["close"].iloc[-1] / df["close"].iloc[0] - 1
        entries, win = trade_stats(res)
        rows.append(dict(
            ticker=t, bars=len(df), trades=entries, win_rate=round(win, 1),
            strat_ret=round((eq.iloc[-1] - 1) * 100, 1),
            buyhold_ret=round(bh * 100, 1),
            pf=round(profit_factor(res), 2),
            sharpe=round(sharpe_ratio(res), 2),
            max_dd=round(max_drawdown(res) * 100, 1),
        ))
    table = pd.DataFrame(rows)

    # Equal-weight portfolio: average strategy return across tickers per date.
    if curves:
        port = pd.concat([c.pct_change().fillna(0) for c in curves.values()], axis=1).mean(axis=1)
        port_eq = (1 + port).cumprod()
        port_res = pd.DataFrame({"strategy_return": port.to_numpy(), "pnl": port.to_numpy()})
        table = pd.concat([table, pd.DataFrame([dict(
            ticker="PORTFOLIO(eq-wt)", bars=len(port_eq), trades=int(table["trades"].sum()),
            win_rate=np.nan,
            strat_ret=round((port_eq.iloc[-1] - 1) * 100, 1), buyhold_ret=np.nan,
            pf=round(profit_factor(port_res), 2), sharpe=round(sharpe_ratio(port_res), 2),
            max_dd=round(max_drawdown(port_res) * 100, 1),
        )])], ignore_index=True)
        curves["PORTFOLIO"] = port_eq

    # Save equity-curve chart.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 6))
        for name, c in curves.items():
            lw = 3 if name == "PORTFOLIO" else 1
            ax.plot(c.index, c.to_numpy(), label=name, linewidth=lw,
                    color="black" if name == "PORTFOLIO" else None)
        ax.set_title(f"Leveraged universe — mean-reversion equity curves ({interval}, {period})")
        ax.set_ylabel("Growth of $1 (strategy)"); ax.legend(ncol=2, fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig("backtest_equity.png", dpi=120)
        print("saved chart -> backtest_equity.png")
    except Exception as e:
        print(f"(chart skipped: {e})")

    return table


if __name__ == "__main__":
    import sys
    period = sys.argv[1] if len(sys.argv) > 1 else "2y"
    interval = sys.argv[2] if len(sys.argv) > 2 else "1d"
    tbl = run(period, interval)
    pd.set_option("display.width", 140, "display.max_columns", 20)
    print(f"\n=== Leveraged universe backtest ({interval} bars, {period}) ===")
    print(tbl.to_string(index=False))
