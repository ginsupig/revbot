"""A/B: does live's SPY market-regime gate (USE_MARKET_REGIME_FILTER) help revbot?

Judges the LIVE bot, so it is built from live's code (fleet rule):
  * config      reversion_bot.live_config.strategy_config_from_env / risk_config_from_env
                (the functions main.py itself calls) + symbol_params.json per-symbol configs
  * signals     ReversionService.evaluate_symbol on the same TRADE_LOOKBACK-bar window
                live passes, once per completed 5-minute bar inside live's entry hours;
                an entry is a payload with go_long and the service's position_plan
  * gate        reversion_bot.market_regime.is_risk_off with live's MARKET_REGIME_EMA /
                REGIME_BUFFER_PCT / REGIME_CONFIRM_BARS, on what live's 1Day SPY fetch
                returns mid-session: prior daily closes + today's partial bar (SPY's
                price at the signal bar)
  * execution   research.audit_reversion.simulate with the service's plans: marketable
                limit at LIMIT_ENTRY_OFFSET_BPS, bracket stop/target, TRAIL_ATR_MULTIPLE
                trail, median exit, 15:50 ET flatten, 30/90-minute re-entry cooldowns

Known gaps vs live (they apply to BOTH arms, so they blur but do not bias the A/B):
  * no historical quotes: the spread gate is off (REQUIRE_SPREAD_DATA=False)
  * one position per symbol; the cross-symbol governor (max positions, heat, daily
    count, drawdown pause) is not modelled
  * the adaptive threshold has no outcome history (live's also has almost none)

The ML probability is stubbed to 0.5: in the service it is reported but never
reaches go_long (the router and the mean-reversion gate score ignore it), and
tests/test_live_config_shared.py fails if that stops being true. It is 75% of
the replay's runtime.

    python -m research.regime_gate_ab --start 2024-01-01 --end 2026-11-14
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reversion_bot.live_config import parse_bool, risk_config_from_env, strategy_config_from_env
from reversion_bot.market_regime import is_risk_off

DATA = ROOT / "state/regime_ab"
NY = "America/New_York"
DEFAULT_SYMBOLS = "AMD,NVDA,SMCI,META,ARM,CRWD,FCX,OXY,CLF,HIMS"

# The fleet's revbot env (config/paper_runtimes.json + child_env), minus the
# spread gate, which needs quotes the history does not have.
FLEET_ENV = {
    "MIN_DOLLAR_VOLUME": "40000",
    "MAX_POSITION_PCT": "0.045",
    "REQUIRE_SPREAD_DATA": "False",
}


def _load(symbol: str) -> pd.DataFrame:
    df = pd.read_pickle(DATA / f"{symbol}_5Min.pkl")
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.sort_values("date").reset_index(drop=True)


def _entry_window(ts_completed_local) -> bool:
    """Live evaluates after the 9:30-10:00 ET blackout and before the EOD entry
    cutoff; the simulator only fills entries 10:00 <= t < 15:40 ET."""
    m = ts_completed_local.hour * 60 + ts_completed_local.minute
    return 600 <= m < 940


def replay(args) -> dict:
    """Live service decisions for one symbol: {bar: (plan, atr, close)}."""
    symbol, start, end = args
    from reversion_bot.config import PerformanceConfig
    from reversion_bot.service import ReversionService
    from reversion_bot.symbol_params import build_symbol_configs, load_symbol_params

    lookback = int(os.getenv("TRADE_LOOKBACK", 160))
    scfg, rcfg = strategy_config_from_env(lookback), risk_config_from_env()
    params = ROOT / os.getenv("SYMBOL_PARAMS_FILE", "symbol_params.json")
    tmp = Path(tempfile.mkdtemp(prefix=f"regab_{symbol}_"))
    service = ReversionService(
        scfg, rcfg,
        PerformanceConfig(
            state_dir=str(tmp),
            enable_adaptive_threshold=parse_bool(os.getenv("PERF_ENABLE_ADAPTIVE_THRESHOLD", "True"), default=True),
            min_samples=int(os.getenv("PERF_MIN_SAMPLES", 20)),
            max_threshold_adj=float(os.getenv("PERF_MAX_THRESHOLD_ADJ", 0.05)),
        ),
        log_file=str(tmp / "service.log"),
        symbol_configs=build_symbol_configs(load_symbol_params(str(params)), scfg),
        min_trade_score=float(os.getenv("MIN_TRADE_SCORE", 0.45)),
    )
    import logging
    service.logger.setLevel(logging.WARNING)
    service._get_ml_probability = lambda df: 0.5   # no effect on go_long (see module doc)

    bars = _load(symbol)
    times = pd.DatetimeIndex(bars["date"])
    done_local = (times + pd.Timedelta(minutes=5)).tz_convert(NY)
    t0, t1 = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    out = {}
    for i in range(lookback - 1, len(bars)):
        if not (t0 <= times[i] < t1) or not _entry_window(done_local[i]):
            continue
        window = bars.iloc[i - lookback + 1:i + 1].reset_index(drop=True)
        res = service.evaluate_symbol(symbol, window, 100_000.0)
        if res.get("go_long") and res.get("position_plan"):
            d = res["decision"]
            out[i] = (res["position_plan"], float(d.get("atr") or 0.0), float(d.get("close") or 0.0))
    return {"symbol": symbol, "signals": out}


def risk_off_at(spy_daily: pd.DataFrame, spy_5m: pd.DataFrame, ts) -> bool:
    """What live's evaluate_market_regime returns at signal time ts."""
    ema = int(os.getenv("MARKET_REGIME_EMA", 50))
    buffer_pct = float(os.getenv("REGIME_BUFFER_PCT", 0.005))
    confirm = int(os.getenv("REGIME_CONFIRM_BARS", 3))
    day = ts.tz_convert(NY).normalize().tz_localize(None)
    prior = spy_daily[spy_daily["day"] < day]
    upto = spy_5m[spy_5m["date"] <= ts]
    if upto.empty:
        return False
    closes = list(prior["close"].astype(float).tail(ema * 3)) + [float(upto["close"].iloc[-1])]
    return bool(is_risk_off(pd.DataFrame({"close": closes}), ema, buffer_pct=buffer_pct, confirm_bars=confirm))


def simulate_arm(symbol, signals, keep, start, end, cost):
    from research.audit_reversion import simulate
    bars = _load(symbol)
    band = int(os.getenv("BAND_LENGTH", 20))
    bars["sma"] = bars["close"].astype(float).rolling(band).mean()
    plans = {i: s[0] for i, s in signals.items() if keep(i)}
    decisions = {i: SimpleNamespace(atr=signals[i][1], close=signals[i][2]) for i in plans}
    rcfg = risk_config_from_env()
    r = simulate(bars, decisions, start, end, cost, details=True, plans=plans, risk_config=rcfg,
                 trail_atr_multiple=rcfg.trail_atr_multiple,
                 limit_entry_offset_bps=float(os.getenv("LIMIT_ENTRY_OFFSET_BPS", 8.0)))
    for t in r["trades"]:
        t["symbol"] = symbol
        t["time"] = bars["date"].iloc[t["entry_bar"]]
    return r["trades"]


def summarize(trades) -> dict:
    if not trades:
        return {"n": 0, "pf": None, "mean_bps": 0.0, "win": 0.0, "pnl": 0.0}
    pnl = np.array([t["pnl"] for t in trades]); bps = np.array([t["ret_bps"] for t in trades])
    w, l = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    return {"n": len(trades), "pf": round(float(w / l), 3) if l else None,
            "mean_bps": round(float(bps.mean()), 2), "win": round(float((pnl > 0).mean()), 3),
            "pnl": round(float(pnl.sum()), 2)}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="2026-11-14")
    p.add_argument("--symbols", default=os.getenv("TRADE_WATCHLIST", DEFAULT_SYMBOLS))
    p.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 4))
    p.add_argument("--costs", default="8,16")
    p.add_argument("--refresh", action="store_true", help="re-run the service replay")
    a = p.parse_args()
    for k, v in FLEET_ENV.items():
        os.environ.setdefault(k, v)
    symbols = [s.strip().upper() for s in a.symbols.split(",") if s.strip()]

    cache = DATA / f"replay_{a.start}_{a.end}.pkl"
    if cache.exists() and not a.refresh:
        replays = pickle.loads(cache.read_bytes())
    else:
        with ProcessPoolExecutor(a.workers) as pool:
            replays = {r["symbol"]: r["signals"] for r in pool.map(replay, [(s, a.start, a.end) for s in symbols])}
        cache.write_bytes(pickle.dumps(replays))
    print(f"signals: {{{', '.join(f'{s}: {len(v)}' for s, v in replays.items())}}}", flush=True)

    spy_d = pd.read_pickle(DATA / "SPY_1Day.pkl")
    spy_d["day"] = pd.to_datetime(spy_d["date"], utc=True).dt.tz_convert(NY).dt.normalize().dt.tz_localize(None)
    spy_5 = _load("SPY")
    gate = {}
    for s, sig in replays.items():
        dates = _load(s)["date"]
        for i in sig:
            # The order goes in after bar i completes; live fetches SPY then.
            gate[(s, i)] = risk_off_at(spy_d, spy_5, dates.iloc[i])

    report = {"window": [a.start, a.end], "env": FLEET_ENV,
              "signals": {s: len(v) for s, v in replays.items()},
              "risk_off_share_of_signals": round(float(np.mean(list(gate.values()))) if gate else 0.0, 3),
              "costs": {}}
    for cost in [float(c) for c in a.costs.split(",")]:
        arms = {"no_gate": [], "gate": []}
        for s, sig in replays.items():
            arms["no_gate"] += simulate_arm(s, sig, lambda i: True, a.start, a.end, cost)
            arms["gate"] += simulate_arm(s, sig, lambda i, s=s: not gate[(s, i)], a.start, a.end, cost)
        # The trades the gate removes, as they would have played out without it.
        cut = [t for t in arms["no_gate"] if gate[(t["symbol"], t["signal_bar"])]]
        kept = [t for t in arms["no_gate"] if not gate[(t["symbol"], t["signal_bar"])]]
        row = {"no_gate": summarize(arms["no_gate"]), "gate": summarize(arms["gate"]),
               "cut_by_gate": summarize(cut), "kept_by_gate": summarize(kept), "by_year": {}}
        years = sorted({t["time"].year for t in arms["no_gate"]})
        for y in years:
            row["by_year"][y] = {k: summarize([t for t in v if t["time"].year == y]) for k, v in
                                 {"no_gate": arms["no_gate"], "gate": arms["gate"], "cut_by_gate": cut}.items()}
        report["costs"][str(cost)] = row
        print(f"\n=== cost {cost:g} bps round trip ===")
        for k in ("no_gate", "gate", "cut_by_gate", "kept_by_gate"):
            print(f"  {k:<13} {row[k]}")
        for y, v in row["by_year"].items():
            print(f"  {y}: " + " | ".join(f"{k} n={x['n']} pf={x['pf']} {x['mean_bps']}bps" for k, x in v.items()))
    out = DATA / f"regime_gate_ab_{a.start}_{a.end}.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print("\nreport:", out)


if __name__ == "__main__":
    main()
