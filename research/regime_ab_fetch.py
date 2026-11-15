"""Cache the bars research/regime_gate_ab.py needs (same fetch + feed as live).

Keys are read in-process from the file named by --env (the fleet's credential
file) and never written anywhere.
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path(__file__).resolve().parents[1] / "state/regime_ab"
SYMBOLS = "AMD,NVDA,SMCI,META,ARM,CRWD,FCX,OXY,CLF,HIMS,SPY".split(",")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True)
    p.add_argument("--prefix", default="REVBOT_ALPACA")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="2026-11-14")
    a = p.parse_args()
    from dotenv import dotenv_values
    v = dotenv_values(a.env)
    key, secret = v[a.prefix + "_KEY"], v[a.prefix + "_SECRET"]
    for k in ("APCA_API_KEY_ID", "ALPACA_API_KEY"):
        os.environ[k] = key
    for k in ("APCA_API_SECRET_KEY", "ALPACA_SECRET_KEY", "ALPACA_API_SECRET"):
        os.environ[k] = secret
    os.environ.setdefault("ALPACA_DATA_FEED", "iex")
    from run_real_backtest import fetch_alpaca_bars

    OUT.mkdir(parents=True, exist_ok=True)
    spy = fetch_alpaca_bars("SPY", "2023-01-01", a.end, "1Day")
    spy.to_pickle(OUT / "SPY_1Day.pkl"); print("SPY 1Day", len(spy), flush=True)
    years = [(f"{y}-01-01", f"{y}-12-31") for y in range(int(a.start[:4]), int(a.end[:4]) + 1)]
    years[0] = (a.start, years[0][1]); years[-1] = (years[-1][0], a.end)
    for s in SYMBOLS:
        path = OUT / f"{s}_5Min.pkl"
        if path.exists():
            continue
        parts = [fetch_alpaca_bars(s, x, y, "5Min") for x, y in years]
        df = (pd.concat([x for x in parts if x is not None and len(x)])
              .drop_duplicates("date").sort_values("date").reset_index(drop=True))
        df.to_pickle(path); print(s, len(df), df["date"].iloc[0], df["date"].iloc[-1], flush=True)


if __name__ == "__main__":
    main()
