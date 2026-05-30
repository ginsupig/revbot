"""Parameter sweep for the leveraged universe with an honest train/test split.

Optimizes portfolio Sharpe on the first ~70% of history, then reports the
winning params on the held-out last ~30% alongside the current defaults.
Daily free-data proxy (no costs), same caveats as backtest_leveraged.py.
"""
from __future__ import annotations
import warnings, itertools
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

from reversion_bot.walkforward import mean_reversion_strategy
from reversion_bot.analytics import profit_factor, sharpe_ratio, max_drawdown

UNIVERSE = ["TQQQ", "SOXL", "TECL", "NVDL", "TSLL", "AAPU", "METU", "GGLL", "MSFU", "AMZU"]
DEFAULTS = dict(band_length=20, band_std_1=1.0, band_std_2=2.0, rsi_max=48.0, adx_max=40.0)

GRID = {
    "band_length": [15, 20, 30],
    "band_std_2": [1.5, 2.0, 2.5],
    "rsi_max": [40.0, 48.0, 55.0],
}


def load(t):
    df = yf.download(t, period="2y", interval="1d", progress=False, auto_adjust=True)
    if df is None or len(df) == 0:
        return None
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    df = df.reset_index().rename(columns={"Date": "date"})
    return df[["date", "open", "high", "low", "close", "volume"]].dropna().reset_index(drop=True)


DATA = {t: load(t) for t in UNIVERSE}
DATA = {t: d for t, d in DATA.items() if d is not None and len(d) > 120}


def portfolio_metrics(params, which):
    """which='train' (first 70%) or 'test' (last 30%). Returns (pf, sharpe, dd, ret%, trades)."""
    rets, trades = [], 0
    for t, d in DATA.items():
        cut = int(len(d) * 0.70)
        seg = d.iloc[:cut] if which == "train" else d.iloc[cut:]
        seg = seg.reset_index(drop=True)
        p = {**DEFAULTS, **params}
        res = mean_reversion_strategy(seg, **p)
        sr = res["strategy_return"].fillna(0)
        sr.index = seg["date"]
        rets.append(sr)
        trades += int((res["signal"] == 1).sum())
    port = pd.concat(rets, axis=1).mean(axis=1).fillna(0)
    pr = pd.DataFrame({"strategy_return": port.to_numpy(), "pnl": port.to_numpy()})
    eq = (1 + port).cumprod()
    return profit_factor(pr), sharpe_ratio(pr), max_drawdown(pr) * 100, (eq.iloc[-1] - 1) * 100, trades


def main():
    keys = list(GRID)
    combos = [dict(zip(keys, v)) for v in itertools.product(*GRID.values())]
    print(f"Sweeping {len(combos)} combos over {len(DATA)} tickers (train=first 70%)...\n")

    scored = []
    for c in combos:
        pf, sh, dd, ret, tr = portfolio_metrics(c, "train")
        scored.append((sh, pf, dd, ret, tr, c))
    scored.sort(reverse=True, key=lambda x: x[0])

    print("=== Top 5 by TRAIN Sharpe ===")
    print(f"{'sharpe':>7}{'pf':>6}{'maxDD':>8}{'ret%':>8}{'trades':>8}  params")
    for sh, pf, dd, ret, tr, c in scored[:5]:
        print(f"{sh:7.2f}{pf:6.2f}{dd:8.1f}{ret:8.1f}{tr:8d}  {c}")

    best = scored[0][5]

    print("\n=== Held-out TEST (last 30%): default vs tuned ===")
    print(f"{'config':<24}{'sharpe':>7}{'pf':>6}{'maxDD':>8}{'ret%':>8}{'trades':>8}")
    for label, params in [("default", {}), (f"tuned {best}", best)]:
        pf, sh, dd, ret, tr = portfolio_metrics(params, "test")
        print(f"{label:<24}{sh:7.2f}{pf:6.2f}{dd:8.1f}{ret:8.1f}{tr:8d}")


if __name__ == "__main__":
    main()
