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

# Factors, all cross-sectionally z-scored each day. Lookback-independent ones
# (setup/movement/exhaustion/volexp/trend) are computed at the entry bar in
# _candidates; lookback-dependent ones (rs/momentum/sector) in score_sweep.
#   setup       reversion depth (lower_band - close)/ATR — deeper oversold = bigger
#   movement    ATR/price — room to revert
#   exhaustion  |today's return| * relative volume — the flush intensity that
#               flagged the MU/ASTS/WDC dislocations manually
#   volexp      ATR / ATR.rolling(50) — volatility expansion (a real flush, not drift)
#   trend       close/SMA50 - 1 — POSITIVE rewards a dip still inside an uptrend and
#               penalizes one that has broken trend (the falling-knife guard)
#   rs          relative strength vs SPY ; momentum own trailing ; sector sector-ETF
FACTORS = ("rs", "momentum", "setup", "movement", "sector", "exhaustion", "volexp", "trend")

# Weight profiles over the factors (missing factor -> weight 0). The sweep tests
# which *combination* of live conditions ranks names best — that combination is
# what the gate validates. `exhaustion` follows the reviewer's factor model
# (hunt good companies temporarily mispriced); `dip_in_trend` is the falling-knife
# guard (deep dip but still above the 50-day).
PROFILES = {
    "balanced":     dict(rs=1.0, momentum=1.0, setup=1.0, sector=1.0),
    "rs_setup":     dict(rs=1.0, setup=1.5),
    "momentum":     dict(momentum=1.5, setup=0.5),
    "sector_rs":    dict(rs=1.0, setup=0.5, sector=1.5),
    "setup_only":   dict(setup=1.0),
    "exhaustion":   dict(exhaustion=0.30, rs=0.25, setup=0.20, volexp=0.15, sector=0.10),
    "dip_in_trend": dict(setup=0.40, trend=0.40, rs=0.20),
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


def _candidates(bars, rsi_max: float, entry_lag: int = 0) -> list:
    """Per-symbol entry candidates: bars below the lower band AND oversold (the
    reversion setup). Returns ``[(day_ord, entry_i, features, ret)]`` where
    ``features`` is the lookback-independent factor dict and ``ret`` the tuned-
    bracket outcome. All features use data through bar i only (causal).
    ``entry_lag=1`` fills at the next bar's open (intraday fill-realism check)."""
    high = bars["high"].astype(float).to_numpy()
    low = bars["low"].astype(float).to_numpy()
    close_s = bars["close"].astype(float)
    close = close_s.to_numpy()
    open_arr = bars["open"].astype(float).to_numpy() if "open" in bars.columns else close
    vol = bars["volume"].astype(float) if "volume" in bars.columns else pd.Series(np.ones(len(bars)))
    atr_s = calculate_atr(bars)
    atr = atr_s.to_numpy()
    rsi = calculate_rsi(bars).to_numpy()
    _, _, lower = calculate_bollinger_bands(bars)
    lower = lower.to_numpy()
    ords = day_ordinals(bars)

    # Exhaustion / volatility-expansion / trend context — all trailing, causal.
    ret = close_s.pct_change().to_numpy()
    rel_vol = (vol / vol.rolling(20).mean()).fillna(1.0).to_numpy()
    atr_exp = (atr_s / atr_s.rolling(50).mean()).fillna(1.0).to_numpy()
    sma50 = close_s.rolling(50).mean().to_numpy()

    idx = []
    for i in range(len(close)):
        a = atr[i] if not np.isnan(atr[i]) else 0.0
        if a <= 0 or np.isnan(lower[i]) or np.isnan(rsi[i]):
            continue
        if close[i] < lower[i] and rsi[i] <= rsi_max:
            idx.append(i)
    outcomes = {e: r for (e, _x, r) in
                setup_outcomes(high, low, close, atr, idx, EXIT, open_arr, entry_lag)}
    rows = []
    for i in idx:
        if i not in outcomes:                       # entry_lag ran off the end of the series
            continue
        a = atr[i]
        trend = (close[i] / sma50[i] - 1.0) if (not np.isnan(sma50[i]) and sma50[i] > 0) else 0.0
        feat = dict(
            setup=float((lower[i] - close[i]) / a),
            movement=float(a / close[i]),
            exhaustion=float(abs(ret[i] if not np.isnan(ret[i]) else 0.0) * rel_vol[i]),
            volexp=float(atr_exp[i]),
            trend=float(trend),
        )
        rows.append((int(ords[i]), i, feat, outcomes[i]))
    return rows


def _name_mom(bars) -> "tuple[np.ndarray, np.ndarray]":
    return bars["close"].astype(float).to_numpy(), day_ordinals(bars)


def score_sweep(symbol_bars, spy_bars, sector_of=None, sector_bars=None,
                profiles=PROFILES, lookbacks=(20, 60), ks=(3, 5),
                rsi_max=45.0, regime_gate=True, holdout_frac=0.25,
                holdout_cut_ord=None, entry_lag=0) -> "tuple[dict, dict]":
    """Sweep (profile, lookback, K) and return ``(research, holdout)`` per-config
    trade returns. Each day every candidate name is scored cross-sectionally on the
    current conditions; the regime gate blanks risk-off days; the top-K trade.
    ``entry_lag=1`` fills at the next bar's open (intraday fill-realism check)."""
    sector_of = sector_of or {}
    sector_bars = sector_bars or {}

    # Per-symbol candidate rows (lookback-independent) + close/ord for momentum.
    cand = {s: _candidates(b, rsi_max, entry_lag) for s, b in symbol_bars.items()}
    closes = {s: _name_mom(b) for s, b in symbol_bars.items()}

    # SPY + sector trailing returns / regime, precomputed per lookback / ord.
    spy_ret = {lb: _trailing_ret_by_ord(spy_bars, lb) for lb in lookbacks}
    regime = _trend_on_by_ord(spy_bars) if regime_gate else {}
    sec_ret = {lb: {etf: _trailing_ret_by_ord(b, lb) for etf, b in sector_bars.items()}
               for lb in lookbacks}

    def _name_ret(sym, i, lb):
        close, _o = closes[sym]
        return close[i] / close[i - lb] - 1.0 if i >= lb and close[i - lb] > 0 else 0.0

    # group candidates by day across the whole universe (once; lookback-independent)
    by_day = defaultdict(list)
    for sym, rows in cand.items():
        for (d, i, feat, ret) in rows:
            by_day[d].append((sym, i, feat, ret))

    research, holdout = defaultdict(list), defaultdict(list)
    for prof_name, lb, K in itertools.product(profiles, lookbacks, ks):
        w = profiles[prof_name]
        key = f"{prof_name}_lb{lb}_K{K}"
        for d in sorted(by_day):
            if regime_gate and not regime.get(d, False):
                continue                              # risk-off: no longs today
            group = by_day[d]
            mom = np.array([_name_ret(s, i, lb) for (s, i, _f, _r) in group])
            cols = {
                "rs": np.array([m - spy_ret[lb].get(d, 0.0) for m in mom]),
                "momentum": mom,
                "sector": np.array([sec_ret[lb].get(sector_of.get(s, ""), {}).get(d, 0.0)
                                    for (s, _i, _f, _r) in group]),
                "setup": np.array([f["setup"] for (_s, _i, f, _r) in group]),
                "movement": np.array([f["movement"] for (_s, _i, f, _r) in group]),
                "exhaustion": np.array([f["exhaustion"] for (_s, _i, f, _r) in group]),
                "volexp": np.array([f["volexp"] for (_s, _i, f, _r) in group]),
                "trend": np.array([f["trend"] for (_s, _i, f, _r) in group]),
            }
            comp = sum(w.get(f, 0.0) * _zscore(cols[f]) for f in FACTORS)
            order = np.argsort(-comp)
            for rank_pos in order[:K]:
                ret = group[rank_pos][3]
                bucket = (holdout if holdout_cut_ord is not None and d >= holdout_cut_ord
                          else research)
                bucket[key].append(ret)
    return ({k: np.asarray(v, float) for k, v in research.items()},
            {k: np.asarray(v, float) for k, v in holdout.items()})


def run_realtime_scorer(symbol_bars, spy_bars, sector_of=None, sector_bars=None,
                        profiles=PROFILES, lookbacks=(20, 60), ks=(3, 5),
                        rsi_max=45.0, regime_gate=True,
                        holdout_frac=0.25, entry_lag=0) -> PipelineResult:
    """End-to-end: score the universe daily, split a date holdout, sweep, gate.
    ``entry_lag=1`` fills at the next bar's open (intraday fill-realism check)."""
    cand_ords = [d for b in symbol_bars.values()
                 for (d, _i, _f, _r) in _candidates(b, rsi_max, entry_lag)]
    if not cand_ords:
        return PipelineResult(None, 0, 0.0, 0.0, False, False, verdict="empty sweep")
    # Split by CALENDAR day, not trade count: the boundary is a fixed fraction of
    # the unique trading days, so clustered flushes can't shrink/inflate the holdout
    # window or straddle a regime period (review fix).
    unique_days = sorted(set(cand_ords))
    cut = unique_days[min(int(round(len(unique_days) * (1.0 - holdout_frac))),
                          len(unique_days) - 1)]
    research, holdout = score_sweep(
        symbol_bars, spy_bars, sector_of, sector_bars, profiles, lookbacks, ks,
        rsi_max, regime_gate, holdout_cut_ord=cut, entry_lag=entry_lag)
    return evaluate_sweep(research, holdout)


def compare_gate(symbol_bars, spy_bars, sector_of=None, sector_bars=None,
                 profiles=PROFILES, lookbacks=(20, 60), ks=(3, 5), rsi_max=45.0,
                 holdout_frac=0.25, entry_lag=0) -> "tuple[PipelineResult, PipelineResult]":
    """Run the identical sweep with the SPY regime gate ON and OFF. The pair is the
    cheap diagnostic for whether the gate is the binding constraint (and thus whether
    an HMM regime upgrade is worth building) — see ``gate_diagnosis``."""
    on = run_realtime_scorer(symbol_bars, spy_bars, sector_of, sector_bars,
                             profiles, lookbacks, ks, rsi_max, True, holdout_frac, entry_lag)
    off = run_realtime_scorer(symbol_bars, spy_bars, sector_of, sector_bars,
                              profiles, lookbacks, ks, rsi_max, False, holdout_frac, entry_lag)
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

# Survivorship control (review #1): a broad, neutral large-cap universe spanning
# all 11 sectors and deliberately including laggards (INTC/T/VZ/PFE/BA/CLF/F...),
# not just today's momentum winners. The edge must come from RANKING, not from
# knowing the names. Still a static membership list -> NOT point-in-time S&P 500
# (that needs a historical-constituents feed; see PR notes); the liquidity filter
# below is the mechanical gate that should ultimately *define* the universe.
NEUTRAL_SECTOR_OF = {
    # XLK information technology
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AVGO": "XLK", "ORCL": "XLK",
    "CRM": "XLK", "CSCO": "XLK", "ACN": "XLK", "ADBE": "XLK", "AMD": "XLK",
    "INTC": "XLK", "QCOM": "XLK", "TXN": "XLK", "IBM": "XLK", "MU": "XLK",
    "AMAT": "XLK", "LRCX": "XLK", "ADI": "XLK",
    # XLC communication services
    "GOOGL": "XLC", "META": "XLC", "NFLX": "XLC", "DIS": "XLC", "CMCSA": "XLC",
    "T": "XLC", "VZ": "XLC", "TMUS": "XLC",
    # XLY consumer discretionary
    "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY", "MCD": "XLY", "NKE": "XLY",
    "LOW": "XLY", "SBUX": "XLY", "BKNG": "XLY", "F": "XLY",
    # XLP consumer staples
    "PG": "XLP", "KO": "XLP", "PEP": "XLP", "COST": "XLP", "WMT": "XLP",
    "MO": "XLP", "CL": "XLP", "MDLZ": "XLP",
    # XLF financials
    "JPM": "XLF", "BAC": "XLF", "WFC": "XLF", "GS": "XLF", "MS": "XLF",
    "C": "XLF", "AXP": "XLF", "SCHW": "XLF", "BLK": "XLF",
    # XLV health care
    "UNH": "XLV", "JNJ": "XLV", "LLY": "XLV", "PFE": "XLV", "MRK": "XLV",
    "ABBV": "XLV", "TMO": "XLV", "ABT": "XLV", "BMY": "XLV", "AMGN": "XLV",
    # XLE energy
    "XOM": "XLE", "CVX": "XLE", "COP": "XLE", "SLB": "XLE", "EOG": "XLE",
    "OXY": "XLE", "MPC": "XLE",
    # XLI industrials
    "CAT": "XLI", "BA": "XLI", "HON": "XLI", "UPS": "XLI", "GE": "XLI",
    "RTX": "XLI", "LMT": "XLI", "DE": "XLI", "UNP": "XLI",
    # XLB materials
    "LIN": "XLB", "FCX": "XLB", "NEM": "XLB", "APD": "XLB", "SHW": "XLB",
    "DOW": "XLB", "CLF": "XLB",
    # XLU utilities
    "NEE": "XLU", "DUK": "XLU", "SO": "XLU", "D": "XLU", "AEP": "XLU",
}
ALL_SECTOR_OF = {**NEUTRAL_SECTOR_OF, **SECTOR_OF}
UNIVERSES = {"curated": list(SECTOR_OF), "neutral": list(NEUTRAL_SECTOR_OF)}
UNIVERSE = UNIVERSES["curated"]


def liquid_filter(bars, min_price=5.0, min_dollar_vol=20e6) -> bool:
    """Mechanical universe gate (review #1): keep a name only if it clears a price
    and average *daily* dollar-volume floor over the window. A coarse liquidity
    screen, not point-in-time membership — but it lets the universe be *defined by
    filters* rather than handpicked. ``True`` = keep.

    Daily dollar volume is summed within each calendar day before averaging, so the
    threshold is timeframe-invariant: on 5Min bars it does NOT compare a per-bar
    figure against a daily floor (which silently cut the whole universe)."""
    if bars is None or len(bars) < 30 or "close" not in bars.columns:
        return False
    close = bars["close"].astype(float)
    vol = bars["volume"].astype(float) if "volume" in bars.columns else pd.Series(np.zeros(len(bars)))
    price_ok = float(close.iloc[-1]) >= min_price
    dollar = (close * vol).to_numpy()
    daily = pd.Series(dollar).groupby(day_ordinals(bars)).sum()   # per-calendar-day $ vol
    adv = float(daily.mean()) if len(daily) else 0.0
    return bool(price_ok and adv >= min_dollar_vol)


def main() -> None:
    p = argparse.ArgumentParser(description="Real-time cross-sectional scorer (gated)")
    p.add_argument("symbols", nargs="*", help="symbols (overrides --universe)")
    p.add_argument("--universe", choices=list(UNIVERSES), default="neutral",
                   help="curated (the 21-name watchlist) or neutral (broad large-cap "
                        "incl. laggards — the survivorship control). Default neutral.")
    p.add_argument("--days", type=int, default=540)
    p.add_argument("--end", default=None)
    p.add_argument("--timeframe", default="1Day")
    p.add_argument("--lookbacks", type=int, nargs="*", default=[20, 60])
    p.add_argument("--ks", type=int, nargs="*", default=[3, 5, 10, 20],
                   help="names traded per day. Bigger Ks ask 'does the top BUCKET "
                        "outperform?' not 'can I pick the one winner?' (review #4).")
    p.add_argument("--profiles", nargs="*", choices=list(PROFILES), default=None,
                   help="subset of weight profiles to sweep (default all). Isolate one "
                        "(e.g. --profiles exhaustion --lookbacks 20 --ks 10) to pre-register a "
                        "single hypothesis and shed the best-of-N deflation penalty.")
    p.add_argument("--no-regime-gate", action="store_true")
    p.add_argument("--entry-lag", type=int, default=0,
                   help="0 = fill at the signal bar's close; 1 = fill at the NEXT bar's open "
                        "(removes the same-bar bid-ask-bounce artifact — the intraday "
                        "fill-realism check that exposes a 5Min PF~10 mirage)")
    p.add_argument("--compare-gate", action="store_true",
                   help="run the identical sweep gate ON vs OFF and print the diagnosis")
    p.add_argument("--no-sector", action="store_true")
    p.add_argument("--min-price", type=float, default=5.0, help="liquidity filter floor")
    p.add_argument("--min-dollar-vol", type=float, default=20e6, help="avg $ volume floor")
    p.add_argument("--holdout", type=float, default=0.25)
    args = p.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    from run_real_backtest import fetch_alpaca_bars   # guarded: needs alpaca + creds

    symbols = [s.upper() for s in args.symbols] if args.symbols else UNIVERSES[args.universe]
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

    raw = {s: b for s in symbols if (b := _fetch(s)) is not None}
    symbol_bars = {s: b for s, b in raw.items()
                   if liquid_filter(b, args.min_price, args.min_dollar_vol)}
    dropped = len(raw) - len(symbol_bars)
    spy_bars = _fetch("SPY")
    if spy_bars is None:
        print("SPY fetch failed — needed for regime + relative strength.")
        return
    sector_bars = None if args.no_sector else {e: b for e in SECTOR_ETFS
                                               if (b := _fetch(e)) is not None}
    profiles = {k: PROFILES[k] for k in args.profiles} if args.profiles else PROFILES
    n_cfg = len(profiles) * len(args.lookbacks) * len(args.ks)
    sec_of = None if args.no_sector else ALL_SECTOR_OF
    print(f"Real-time scorer | universe={args.universe} | {len(symbol_bars)}/{len(symbols)} names "
          f"({dropped} cut by liquidity >=${args.min_price:g}/>=${args.min_dollar_vol/1e6:g}M) "
          f"| {start}->{end_s} | ")
    print(f"  {args.timeframe} | profiles={list(profiles)} lb={args.lookbacks} K={args.ks} "
          f"({n_cfg} configs) | holdout={args.holdout:.0%} | entry_lag={args.entry_lag}"
          + (" (next-bar open)" if args.entry_lag else " (signal-bar close)")
          + (" | COMPARE gate on/off" if args.compare_gate else
             f" | regime_gate={not args.no_regime_gate}"))
    if args.compare_gate:
        on, off = compare_gate(symbol_bars, spy_bars, sec_of, sector_bars,
                               profiles, args.lookbacks, args.ks,
                               holdout_frac=args.holdout, entry_lag=args.entry_lag)
        print("\n" + format_gate_comparison(on, off))
    else:
        res = run_realtime_scorer(
            symbol_bars, spy_bars, sec_of, sector_bars, profiles, args.lookbacks,
            args.ks, regime_gate=not args.no_regime_gate, holdout_frac=args.holdout,
            entry_lag=args.entry_lag)
        print("\n" + format_report(res))


if __name__ == "__main__":
    main()
