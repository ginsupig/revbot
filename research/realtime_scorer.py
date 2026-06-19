"""Real-time cross-sectional scorer: rank the WHOLE universe by today's conditions.

No allowlist, no per-name history requirement. Every name (1 or 1000) is scored
each day against the *current* regime and its relative strength / momentum /
sector strength / movement / setup depth, ranked cross-sectionally, and only the
top-K that clear the regime gate are traded. A brand-new name qualifies the
instant its conditions rank — nothing is preselected. Static is the failure mode
(CLAUDE.md: regime dominates; don't fight the tape with a static signal); this is
the dynamic replacement for the frozen allowlist.

Causal: every feature for day d is computed from data through d's close (the
at-close decision point), never the forward bars the trade will live through
(see research.leakage).

Factors — each cross-sectionally z-scored across that day's candidates, then
weighted by a profile:
  rs        relative strength vs SPY over `lookback`
  momentum  the name's own trailing return over `lookback`
  setup     reversion depth (lower_band - close)/ATR — deeper oversold dip = bigger
  movement  ATR/price — room to revert
  sector    sector-ETF trailing return (injected sector map; 0 if unavailable)
A regime gate (SPY above its own trend) blanks the long book when risk-off,
matching the live market_regime gate.

The weight profile, lookback and K form a parameter sweep -> per-config trade
returns split into a research window and an untouched holdout -> fed to
research.pipeline.evaluate_sweep. Scoring the universe this way does NOT preselect
winners: it makes the SCORING FUNCTION the object under test, which still must
clear the deflated-Sharpe + fold + holdout bar before it earns live wiring.
Research-only until then (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import itertools
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .engines.reference import EXIT
from .daily_ranker import setup_outcomes, day_ordinals
from .pipeline import evaluate_sweep, PipelineResult
from .run import format_report

from reversion_bot.indicators import calculate_atr, calculate_rsi, calculate_bollinger_bands

FACTORS = ("rs", "momentum", "setup", "movement", "sector")

# Weight profiles over the factors. The sweep tests which *combination* of live
# conditions ranks names best — that combination is what the gate validates.
PROFILES = {
    "balanced":   dict(rs=1.0, momentum=1.0, setup=1.0, movement=0.0, sector=1.0),
    "rs_setup":   dict(rs=1.0, momentum=0.0, setup=1.5, movement=0.0, sector=0.0),
    "momentum":   dict(rs=0.0, momentum=1.5, setup=0.5, movement=0.0, sector=0.0),
    "sector_rs":  dict(rs=1.0, momentum=0.0, setup=0.5, movement=0.0, sector=1.5),
    "setup_only": dict(rs=0.0, momentum=0.0, setup=1.0, movement=0.0, sector=0.0),
}


def _zscore(x: np.ndarray) -> np.ndarray:
    """Cross-sectional standardize; all-equal or single element -> zeros."""
    x = np.asarray(x, float)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def _trailing_ret_by_ord(bars, lookback) -> dict:
    """``{day_ord: trailing return over `lookback` bars}`` for one series (SPY/sector)."""
    close = bars["close"].astype(float).to_numpy()
    ords = day_ordinals(bars)
    out = {}
    for i in range(lookback, len(close)):
        if close[i - lookback] > 0:
            out[int(ords[i])] = close[i] / close[i - lookback] - 1.0
    return out


def _trend_on_by_ord(bars, win=50) -> dict:
    """``{day_ord: close > SMA(win)}`` — the SPY risk-on regime gate."""
    close = bars["close"].astype(float)
    sma = close.rolling(win).mean()
    ords = day_ordinals(bars)
    out = {}
    c, m = close.to_numpy(), sma.to_numpy()
    for i in range(len(c)):
        if not np.isnan(m[i]):
            out[int(ords[i])] = bool(c[i] > m[i])
    return out


def _candidates(bars, rsi_max: float) -> list:
    """Per-symbol entry candidates: bars below the lower band AND oversold (the
    reversion setup). Returns ``[(day_ord, entry_i, setup_depth, movement, ret)]``
    with the tuned-bracket outcome — lookback-independent features + the trade."""
    high = bars["high"].astype(float).to_numpy()
    low = bars["low"].astype(float).to_numpy()
    close = bars["close"].astype(float).to_numpy()
    atr = calculate_atr(bars).to_numpy()
    rsi = calculate_rsi(bars).to_numpy()
    _, _, lower = calculate_bollinger_bands(bars)
    lower = lower.to_numpy()
    ords = day_ordinals(bars)

    idx = []
    for i in range(len(close)):
        a = atr[i] if not np.isnan(atr[i]) else 0.0
        if a <= 0 or np.isnan(lower[i]) or np.isnan(rsi[i]):
            continue
        if close[i] < lower[i] and rsi[i] <= rsi_max:
            idx.append(i)
    outcomes = {e: r for (e, _x, r) in setup_outcomes(high, low, close, atr, idx, EXIT)}
    rows = []
    for i in idx:
        a = atr[i]
        rows.append((int(ords[i]), i, float((lower[i] - close[i]) / a),
                     float(a / close[i]), outcomes[i]))
    return rows


def _name_mom(bars) -> "tuple[np.ndarray, np.ndarray]":
    return bars["close"].astype(float).to_numpy(), day_ordinals(bars)


def score_sweep(symbol_bars, spy_bars, sector_of=None, sector_bars=None,
                profiles=PROFILES, lookbacks=(20, 60), ks=(3, 5),
                rsi_max=45.0, regime_gate=True, holdout_frac=0.25,
                holdout_cut_ord=None) -> "tuple[dict, dict]":
    """Sweep (profile, lookback, K) and return ``(research, holdout)`` per-config
    trade returns. Each day every candidate name is scored cross-sectionally on the
    current conditions; the regime gate blanks risk-off days; the top-K trade."""
    sector_of = sector_of or {}
    sector_bars = sector_bars or {}

    # Per-symbol candidate rows (lookback-independent) + close/ord for momentum.
    cand = {s: _candidates(b, rsi_max) for s, b in symbol_bars.items()}
    closes = {s: _name_mom(b) for s, b in symbol_bars.items()}

    # SPY + sector trailing returns / regime, precomputed per lookback / ord.
    spy_ret = {lb: _trailing_ret_by_ord(spy_bars, lb) for lb in lookbacks}
    regime = _trend_on_by_ord(spy_bars) if regime_gate else {}
    sec_ret = {lb: {etf: _trailing_ret_by_ord(b, lb) for etf, b in sector_bars.items()}
               for lb in lookbacks}

    def _name_ret(sym, i, lb):
        close, _o = closes[sym]
        return close[i] / close[i - lb] - 1.0 if i >= lb and close[i - lb] > 0 else 0.0

    research, holdout = defaultdict(list), defaultdict(list)
    for prof_name, lb, K in itertools.product(profiles, lookbacks, ks):
        w = profiles[prof_name]
        key = f"{prof_name}_lb{lb}_K{K}"
        # group candidates by day across the whole universe
        by_day = defaultdict(list)
        for sym, rows in cand.items():
            for (d, i, setup, movement, ret) in rows:
                by_day[d].append((sym, i, setup, movement, ret))
        for d in sorted(by_day):
            if regime_gate and not regime.get(d, False):
                continue                              # risk-off: no longs today
            group = by_day[d]
            mom = np.array([_name_ret(s, i, lb) for (s, i, _se, _mv, _r) in group])
            rs = np.array([m - spy_ret[lb].get(d, 0.0) for m in mom])
            setup = np.array([se for (_s, _i, se, _mv, _r) in group])
            movement = np.array([mv for (_s, _i, _se, mv, _r) in group])
            sector = np.array([sec_ret[lb].get(sector_of.get(s, ""), {}).get(d, 0.0)
                               for (s, _i, _se, _mv, _r) in group])
            comp = (w["rs"] * _zscore(rs) + w["momentum"] * _zscore(mom)
                    + w["setup"] * _zscore(setup) + w["movement"] * _zscore(movement)
                    + w["sector"] * _zscore(sector))
            order = np.argsort(-comp)
            for rank_pos in order[:K]:
                ret = group[rank_pos][4]
                bucket = (holdout if holdout_cut_ord is not None and d >= holdout_cut_ord
                          else research)
                bucket[key].append(ret)
    return ({k: np.asarray(v, float) for k, v in research.items()},
            {k: np.asarray(v, float) for k, v in holdout.items()})


def run_realtime_scorer(symbol_bars, spy_bars, sector_of=None, sector_bars=None,
                        profiles=PROFILES, lookbacks=(20, 60), ks=(3, 5),
                        rsi_max=45.0, regime_gate=True,
                        holdout_frac=0.25) -> PipelineResult:
    """End-to-end: score the universe daily, split a date holdout, sweep, gate."""
    all_ords = sorted(d for b in symbol_bars.values()
                      for (d, _i, _s, _m, _r) in _candidates(b, rsi_max))
    if not all_ords:
        return PipelineResult(None, 0, 0.0, 0.0, False, False, verdict="empty sweep")
    cut = all_ords[min(int(round(len(all_ords) * (1.0 - holdout_frac))), len(all_ords) - 1)]
    research, holdout = score_sweep(
        symbol_bars, spy_bars, sector_of, sector_bars, profiles, lookbacks, ks,
        rsi_max, regime_gate, holdout_cut_ord=cut)
    return evaluate_sweep(research, holdout)


def compare_gate(symbol_bars, spy_bars, sector_of=None, sector_bars=None,
                 profiles=PROFILES, lookbacks=(20, 60), ks=(3, 5), rsi_max=45.0,
                 holdout_frac=0.25) -> "tuple[PipelineResult, PipelineResult]":
    """Run the identical sweep with the SPY regime gate ON and OFF. The pair is the
    cheap diagnostic for whether the gate is the binding constraint (and thus whether
    an HMM regime upgrade is worth building) — see ``gate_diagnosis``."""
    on = run_realtime_scorer(symbol_bars, spy_bars, sector_of, sector_bars,
                             profiles, lookbacks, ks, rsi_max, True, holdout_frac)
    off = run_realtime_scorer(symbol_bars, spy_bars, sector_of, sector_bars,
                              profiles, lookbacks, ks, rsi_max, False, holdout_frac)
    return on, off


def gate_diagnosis(on: PipelineResult, off: PipelineResult) -> str:
    """One-line read of the gate-on vs gate-off comparison: does the regime gate
    bind, and is an HMM upgrade justified?"""
    on_ok = on.verdict.startswith("DEPLOY")
    off_ok = off.verdict.startswith("DEPLOY")
    if off_ok and not on_ok:
        return ("DIAGNOSIS: gate-off clears but gate-on does not -> the SMA regime gate is "
                "the weak link. An HMM (filtered, causal) regime upgrade is justified now.")
    if on_ok and off_ok:
        return ("DIAGNOSIS: both clear -> ship gate-on; a fancier (HMM) gate is premature "
                "optimization. The regime input is not the bottleneck.")
    if on_ok and not off_ok:
        return ("DIAGNOSIS: gate-on clears, gate-off does not -> the regime gate is already "
                "earning its keep. Keep the SMA gate; HMM is optional polish, not a blocker.")
    return ("DIAGNOSIS: neither clears -> the edge is not in the regime input. A fancier gate "
            "won't save it; diagnose the factors/weights (or conclude no edge) instead.")


def format_gate_comparison(on: PipelineResult, off: PipelineResult) -> str:
    def _line(tag, r):
        ho = r.holdout or {}
        return (f"  [{tag:>8}] {r.verdict}\n"
                f"             best={r.chosen} deflated={r.deflated_sharpe:+.3f} "
                f"breakeven={r.breakeven_bps:.1f}bps "
                f"holdout(n={ho.get('n', 0)} mean={ho.get('mean', 0) * 100:.3f}% "
                f"PF={ho.get('pf', 0):.2f})")
    return "\n".join([_line("gate ON", on), _line("gate OFF", off), "", gate_diagnosis(on, off)])


# Default S&P sector map for the live watchlist + the sector ETFs to fetch. Used
# only by the CLI; the core takes these injected so tests need no broker.
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLI", "XLC", "XLB", "XLP", "XLU"]
SECTOR_OF = {
    "ASTS": "XLC", "MU": "XLK", "AMD": "XLK", "NVDA": "XLK", "SMCI": "XLK",
    "META": "XLC", "APP": "XLK", "ARM": "XLK", "CRWD": "XLK", "MSFT": "XLK",
    "LRCX": "XLK", "FCX": "XLB", "OXY": "XLE", "CLF": "XLB", "HIMS": "XLV",
    "WDC": "XLK", "COIN": "XLF", "ADI": "XLK", "AMAT": "XLK", "TSLA": "XLY",
    "IONQ": "XLK",
}
UNIVERSE = list(SECTOR_OF)


def main() -> None:
    p = argparse.ArgumentParser(description="Real-time cross-sectional scorer (gated)")
    p.add_argument("symbols", nargs="*", help="symbols (overrides the default universe)")
    p.add_argument("--days", type=int, default=540)
    p.add_argument("--end", default=None)
    p.add_argument("--timeframe", default="1Day")
    p.add_argument("--lookbacks", type=int, nargs="*", default=[20, 60])
    p.add_argument("--ks", type=int, nargs="*", default=[3, 5])
    p.add_argument("--no-regime-gate", action="store_true")
    p.add_argument("--compare-gate", action="store_true",
                   help="run the identical sweep gate ON vs OFF and print the diagnosis")
    p.add_argument("--no-sector", action="store_true")
    p.add_argument("--holdout", type=float, default=0.25)
    args = p.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    from run_real_backtest import fetch_alpaca_bars   # guarded: needs alpaca + creds

    symbols = [s.upper() for s in args.symbols] if args.symbols else UNIVERSE
    end = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.now()
    start = (end - timedelta(days=args.days)).strftime("%Y-%m-%d")
    end_s = end.strftime("%Y-%m-%d")

    def _fetch(sym):
        try:
            b = fetch_alpaca_bars(sym, start, end_s, args.timeframe)
            return b if b is not None and len(b) >= 30 else None
        except Exception as e:
            print(f"  # {sym}: {e}")
            return None

    symbol_bars = {s: b for s in symbols if (b := _fetch(s)) is not None}
    spy_bars = _fetch("SPY")
    if spy_bars is None:
        print("SPY fetch failed — needed for regime + relative strength.")
        return
    sector_bars = None if args.no_sector else {e: b for e in SECTOR_ETFS
                                               if (b := _fetch(e)) is not None}
    n_cfg = len(PROFILES) * len(args.lookbacks) * len(args.ks)
    sec_of = None if args.no_sector else SECTOR_OF
    print(f"Real-time scorer | {len(symbol_bars)}/{len(symbols)} names | {start}->{end_s} | "
          f"{args.timeframe} | profiles={list(PROFILES)} lb={args.lookbacks} K={args.ks} "
          f"({n_cfg} configs) | holdout={args.holdout:.0%}"
          + (" | COMPARE gate on/off" if args.compare_gate else
             f" | regime_gate={not args.no_regime_gate}"))
    if args.compare_gate:
        on, off = compare_gate(symbol_bars, spy_bars, sec_of, sector_bars,
                               PROFILES, args.lookbacks, args.ks,
                               holdout_frac=args.holdout)
        print("\n" + format_gate_comparison(on, off))
    else:
        res = run_realtime_scorer(
            symbol_bars, spy_bars, sec_of, sector_bars, PROFILES, args.lookbacks,
            args.ks, regime_gate=not args.no_regime_gate, holdout_frac=args.holdout)
        print("\n" + format_report(res))


if __name__ == "__main__":
    main()
