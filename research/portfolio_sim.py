"""Occupancy-realistic portfolio sim: what the candidate ACTUALLY draws down.

The gated sweep books every selected signal as an independent full-size trade —
occupancy-free, so on a day with 20 dips it holds 20 correlated falling knives at
once and reports a ~63% MC drawdown. The live bot never does that: it caps open
positions, holds one per symbol, and shares capital. This walks the calendar with
those constraints — at most ``max_positions`` open, filled by the scorer's daily
rank when a slot frees — and returns the real equity curve, max drawdown, and
Sharpe. The occupancy-free sweep overstates BOTH the return and the drawdown
(CLAUDE.md occupancy discipline); this is the honest read.

Two realism upgrades over the first cut:
  * mark-to-market drawdown — open positions are marked at their unrealized P&L
    every day (not only at exit), so an intra-hold trough is caught. The realized-
    basis DD understates the pain; the MTM DD is the number that matters.
  * sizing="risk" — size each trade by the live model (risk_pct of equity at an
    ATR-multiple stop, capped at max_pos_frac), not naive equal weight.

Research-only. CLI (needs alpaca + creds):
    python -m research.portfolio_sim --universe neutral --max-positions 4 --sizing risk
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd

from .daily_ranker import setup_outcomes, day_ordinals
from .engines.reference import EXIT
from .realtime_scorer import (
    _candidates, UNIVERSES, liquid_filter, _trailing_ret_by_ord, _zscore,
    FACTORS, PROFILES, ALL_SECTOR_OF, SECTOR_ETFS,
)

from reversion_bot.indicators import calculate_atr, calculate_rsi, calculate_bollinger_bands

ATR_FLOOR = 0.0035          # matches RiskConfig.atr_floor_pct


@dataclass(frozen=True)
class Entry:
    entry_ord: int
    exit_ord: int
    ret: float              # net bracket return (already minus cost)
    sym: str
    score: float            # oversold depth — the ranking key
    entry_price: float = 0.0
    atr_pct: float = 0.0     # ATR / entry_price at entry (for risk sizing + MTM)


def build_entries(symbol_bars, rsi_max=45.0) -> list:
    """``[Entry]`` across the universe — reversion candidates with bracket exit DAY,
    oversold-depth score, entry price and ATR% (for risk sizing / mark-to-market)."""
    out = []
    for sym, bars in symbol_bars.items():
        rows = _candidates(bars, rsi_max)            # (day_ord, entry_i, feat, ret)
        if not rows:
            continue
        ords = day_ordinals(bars)
        high = bars["high"].astype(float).to_numpy()
        low = bars["low"].astype(float).to_numpy()
        close = bars["close"].astype(float).to_numpy()
        atr = calculate_atr(bars).to_numpy()
        idx = [i for (_d, i, _f, _r) in rows]
        exit_ord = {e: int(ords[x]) for (e, x, _r) in setup_outcomes(high, low, close, atr, idx, EXIT)}
        for (d, i, feat, ret) in rows:
            price = float(close[i])
            a = float(atr[i]) if not np.isnan(atr[i]) else 0.0
            out.append(Entry(int(d), exit_ord[i], float(ret), sym, float(feat["setup"]),
                             price, max(a / price, ATR_FLOOR) if price > 0 else ATR_FLOOR))
    return out


def build_closes(symbol_bars) -> dict:
    """``{sym: {day_ord: close}}`` — for daily mark-to-market of open positions."""
    out = {}
    for sym, bars in symbol_bars.items():
        ords = day_ordinals(bars)
        close = bars["close"].astype(float).to_numpy()
        out[sym] = {int(o): float(c) for o, c in zip(ords, close)}
    return out


def _notional_frac(e: Entry, sizing, max_positions, risk_pct, stop_atr_mult, max_pos_frac) -> float:
    if sizing == "equal":
        return 1.0 / max_positions
    stop_pct = stop_atr_mult * e.atr_pct            # fractional stop distance
    return min(risk_pct / stop_pct, max_pos_frac) if stop_pct > 0 else max_pos_frac


def _curve_stats(realized, mtm, n_trades, n_skipped, sizing) -> dict:
    def _mdd(curve):
        eq = np.asarray(curve, float)
        if eq.size < 2:
            return 0.0
        peak = np.maximum.accumulate(eq)
        return float(((eq - peak) / peak).min())
    eq = np.asarray(realized, float)
    rets = np.diff(eq) / eq[:-1] if eq.size > 1 else np.array([0.0])
    sd = rets.std(ddof=0)
    return dict(
        sizing=sizing, n_trades=int(n_trades), n_skipped=int(n_skipped),
        max_drawdown=_mdd(realized), mtm_max_drawdown=_mdd(mtm),
        total_return=float(eq[-1] - 1.0) if eq.size else 0.0,
        sharpe=float(rets.mean() / sd) if sd > 0 else 0.0,
        final_equity=float(eq[-1]) if eq.size else 1.0,
        equity=eq.tolist(),
    )


def simulate_portfolio(entries, closes_by_ord=None, max_positions=4, sizing="equal",
                       risk_pct=0.005, stop_atr_mult=1.2, max_pos_frac=0.20,
                       max_total_exposure=None, max_portfolio_heat=None) -> dict:
    """Walk the calendar holding <= ``max_positions`` positions, one per symbol,
    filling free slots by score (desc). ``sizing`` 'equal' = equity/N, 'risk' =
    live ATR-risk model. With ``closes_by_ord`` the open book is marked to market
    daily so the drawdown reflects intra-hold troughs, not just realized exits.

    Live governor constraints, off (None) by default so unit tests are unaffected,
    set to live values by the CLI:
      * ``max_total_exposure`` — sum of open notionals as a fraction of equity
        (live MAX_TOTAL_EXPOSURE_PCT=0.95). Blocks the leverage a high cap x risk%
        cell would otherwise imply.
      * ``max_portfolio_heat`` — sum of open risk-to-stop as a fraction of equity
        (live MAX_PORTFOLIO_HEAT_PCT=0.02). This is why risk_pct can't be raised
        without raising the heat cap in lockstep: 4 x 1.5% = 6% heat >> 2%.
    """
    entries = [e if isinstance(e, Entry) else Entry(*e) for e in entries]
    if not entries:
        return dict(sizing=sizing, n_trades=0, n_skipped=0, max_drawdown=0.0,
                    mtm_max_drawdown=0.0, total_return=0.0, sharpe=0.0,
                    final_equity=1.0, equity=[1.0])
    if closes_by_ord:                                # mark every trading day
        days = sorted({o for m in closes_by_ord.values() for o in m})
    else:
        days = sorted({e.entry_ord for e in entries} | {e.exit_ord for e in entries})
    by_entry = {}
    for e in entries:
        by_entry.setdefault(e.entry_ord, []).append(e)

    equity = 1.0
    open_pos = []          # {exit, notional, ret, sym, entry_price, mark}
    realized_curve, mtm_curve = [], []
    n_trades = n_skipped = 0
    for d in days:
        # 1) close positions whose bracket exits on/before today
        keep = []
        for p in open_pos:
            if p["exit"] <= d:
                equity += p["notional"] * p["ret"]
            else:
                keep.append(p)
        open_pos = keep
        held = {p["sym"] for p in open_pos}
        # 2) fill free slots from today's candidates, highest score first
        for e in sorted(by_entry.get(d, []), key=lambda x: x.score, reverse=True):
            if len(open_pos) >= max_positions:
                n_skipped += 1
                continue
            if e.sym in held:
                n_skipped += 1
                continue
            nf = _notional_frac(e, sizing, max_positions, risk_pct, stop_atr_mult, max_pos_frac)
            notional = nf * equity
            pos_heat = nf * (stop_atr_mult * e.atr_pct) * equity   # dollar risk to stop
            # live governor caps: total exposure and portfolio heat
            if max_total_exposure is not None:
                if sum(p["notional"] for p in open_pos) + notional > max_total_exposure * equity:
                    n_skipped += 1
                    continue
            if max_portfolio_heat is not None:
                if sum(p["heat"] for p in open_pos) + pos_heat > max_portfolio_heat * equity:
                    n_skipped += 1
                    continue
            if e.exit_ord <= d:                      # same-day exit: realize at once
                equity += notional * e.ret
            else:
                open_pos.append(dict(exit=e.exit_ord, notional=notional, ret=e.ret,
                                     sym=e.sym, entry_price=e.entry_price, mark=0.0,
                                     heat=pos_heat))
                held.add(e.sym)
            n_trades += 1
        # 3) mark open book to market for the MTM curve
        unrealized = 0.0
        for p in open_pos:
            if closes_by_ord and p["entry_price"] > 0:
                c = closes_by_ord.get(p["sym"], {}).get(d)
                if c is not None:
                    p["mark"] = c / p["entry_price"] - 1.0
            unrealized += p["notional"] * p["mark"]
        realized_curve.append(equity)
        mtm_curve.append(equity + unrealized)
    return _curve_stats(realized_curve, mtm_curve, n_trades, n_skipped, sizing)


def run_portfolio_sim(symbol_bars, max_positions=4, rsi_max=45.0, sizing="equal") -> dict:
    return simulate_portfolio(build_entries(symbol_bars, rsi_max),
                              build_closes(symbol_bars), max_positions, sizing)


# Sizing frontier: equal-weight (the return ceiling, no vol scaling) vs ATR-risk at
# a rising risk budget. ATR-risk sizes notional ~ 1/ATR, so it underweights the
# high-vol deep-dip winners; raising risk_pct scales THEM up first (low-vol names
# are already pinned at max_pos_frac), tracing return vs drawdown to find the knee.
SIZING_GRID = [("equal", 0.0), ("risk", 0.005), ("risk", 0.0075),
               ("risk", 0.01), ("risk", 0.015), ("risk", 0.02)]


def compare_sizing(entries, closes_by_ord, max_positions=4, grid=SIZING_GRID,
                   stop_atr_mult=1.2, max_pos_frac=0.20,
                   max_total_exposure=None, max_portfolio_heat=None) -> list:
    """``[(label, stats)]`` for each (mode, risk_pct) in ``grid`` at a fixed cap —
    the return/drawdown frontier across sizing aggressiveness."""
    rows = []
    for mode, rp in grid:
        stats = simulate_portfolio(entries, closes_by_ord, max_positions, mode,
                                   rp or 0.005, stop_atr_mult, max_pos_frac,
                                   max_total_exposure, max_portfolio_heat)
        label = "equal" if mode == "equal" else f"risk@{rp*100:g}%"
        rows.append((label, stats))
    return rows


def _ret_dd(stats) -> float:
    dd = stats["mtm_max_drawdown"]
    return stats["total_return"] / abs(dd) if dd < 0 else float("inf")


def format_sizing_comparison(rows, max_positions: int) -> str:
    lines = [f"  sizing frontier @ max_positions={max_positions} (MTM drawdown, compounding)",
             f"  {'SIZING':<11}{'RETURN':>9}{'MTM_DD':>9}{'SHARPE':>8}{'RET/DD':>8}{'TRADES':>8}",
             "  " + "-" * 51]
    for label, s in rows:
        lines.append(f"  {label:<11}{s['total_return']*100:>8.1f}%{s['mtm_max_drawdown']*100:>8.1f}%"
                     f"{s['sharpe']:>8.3f}{_ret_dd(s):>8.1f}{s['n_trades']:>8}")
    lines += ["",
              "  equal = return ceiling (no vol scaling). Rising risk@% adds budget to the",
              "  high-vol deep-dip names ATR-risk underweights — watch where RET/DD peaks:",
              "  that knee recovers return without paying it all back in drawdown."]
    return "\n".join(lines)


def grid_sweep(entries, closes_by_ord, caps, risk_pcts, stop_atr_mult=1.2,
               max_pos_frac=0.20, max_total_exposure=None, max_portfolio_heat=None) -> dict:
    """``{(cap, risk_pct): stats}`` over the position-cap x risk-budget grid — the
    two knobs interact (more slots vs more per-trade budget both add exposure, but
    differently against the position cap), so the best risk/return point is 2-D.
    With the live exposure/heat caps passed, unreachable (leveraged/over-heat) cells
    are throttled exactly as the governor would."""
    out = {}
    for cap in caps:
        for rp in risk_pcts:
            out[(cap, rp)] = simulate_portfolio(entries, closes_by_ord, cap, "risk", rp,
                                                stop_atr_mult, max_pos_frac,
                                                max_total_exposure, max_portfolio_heat)
    return out


def _grid_matrix(title, caps, risk_pcts, grid, cell) -> str:
    head = "  cap\\risk" + "".join(f"{rp*100:>8g}%" for rp in risk_pcts)
    lines = [f"  {title}", head]
    for cap in caps:
        lines.append(f"  {cap:>7} " + "".join(f"{cell(grid[(cap, rp)]):>9}" for rp in risk_pcts))
    return "\n".join(lines)


def format_grid(grid, caps, risk_pcts) -> str:
    blocks = [
        _grid_matrix("RET/DD  (return / MTM drawdown — the decision metric)", caps, risk_pcts,
                     grid, lambda s: f"{_ret_dd(s):.1f}"),
        _grid_matrix("RETURN %", caps, risk_pcts, grid,
                     lambda s: f"{s['total_return']*100:.1f}"),
        _grid_matrix("MTM DRAWDOWN %", caps, risk_pcts, grid,
                     lambda s: f"{s['mtm_max_drawdown']*100:.1f}"),
    ]
    return ("\n\n".join(blocks) + "\n\n"
            "  With the live exposure (95%) + heat (2%) caps applied, unreachable cells are\n"
            "  throttled by the governor (heat binds first: 4 x 1.5% = 6% >> 2%, so high\n"
            "  risk% cells quietly hold fewer names). Pick the (cap, risk%) with the best\n"
            "  RET/DD that holds on BOTH windows — the robust, live-reachable cell. To run a\n"
            "  cap x risk% that needs more heat, you must raise --max-heat in lockstep.")


# --- factor-ranked daily experiment ------------------------------------------
# Momentum-free profiles (momentum fights a dip-buy). The question: does an
# exhaustion / quality-filtered reversion basket survive realistic execution
# (next-open fill + occupancy + heat + ATR-risk) on a multi-day hold?
EXPERIMENT_PROFILES = {
    "exhaustion": PROFILES["exhaustion"],          # exhaustion+rs+setup+volexp+sector
    "sector_rs":  PROFILES["sector_rs"],           # rs+sector+setup
    "rs":         dict(rs=1.0, setup=0.5),
    "movement":   dict(movement=1.0, setup=0.5),
    "setup_only": dict(setup=1.0),                 # CONTROL: no factor tilt — the base
                                                   # oversold-reversion. If the tilts flip
                                                   # sign across windows but this stays
                                                   # positive on both, the tilt is regime
                                                   # noise and the base is the real edge.
}


def build_factor_entries(symbol_bars, spy_bars, sector_of, sector_bars, profile,
                         hold, lookback=60, rsi_max=45.0, entry_lag=1) -> list:
    """``[Entry]`` ranked by the cross-sectional factor composite. For each oversold
    candidate: next-open entry (``entry_lag``), bracket exit at the given ``hold``,
    factors z-scored across that signal-day's names and weighted by ``profile``.
    Occupancy/heat/risk are applied later by ``simulate_portfolio``."""
    exit = dict(EXIT, hold=int(hold))
    sector_of = sector_of or {}
    spy_ret = _trailing_ret_by_ord(spy_bars, lookback) if spy_bars is not None else {}
    sec_ret = {etf: _trailing_ret_by_ord(b, lookback) for etf, b in (sector_bars or {}).items()}

    raw = []
    for sym, bars in symbol_bars.items():
        high = bars["high"].astype(float).to_numpy()
        low = bars["low"].astype(float).to_numpy()
        close_s = bars["close"].astype(float)
        close = close_s.to_numpy()
        openp = bars["open"].astype(float).to_numpy() if "open" in bars.columns else close
        vol = bars["volume"].astype(float) if "volume" in bars.columns else pd.Series(np.ones(len(bars)))
        atr_s = calculate_atr(bars)
        atr = atr_s.to_numpy()
        rsi = calculate_rsi(bars).to_numpy()
        _, _, lower = calculate_bollinger_bands(bars)
        lower = lower.to_numpy()
        ords = day_ordinals(bars)
        ret_pct = close_s.pct_change().to_numpy()
        rel_vol = (vol / vol.rolling(20).mean()).fillna(1.0).to_numpy()
        atr_exp = (atr_s / atr_s.rolling(50).mean()).fillna(1.0).to_numpy()
        sma50 = close_s.rolling(50).mean().to_numpy()
        n = len(close)

        idx = [i for i in range(n)
               if (not np.isnan(atr[i]) and atr[i] > 0 and not np.isnan(lower[i])
                   and not np.isnan(rsi[i]) and close[i] < lower[i] and rsi[i] <= rsi_max)]
        out_by_entry = {e: (x, r) for (e, x, r) in
                        setup_outcomes(high, low, close, atr, idx, exit, openp, entry_lag)}
        etf = sector_of.get(sym, "")
        for i in idx:
            if i not in out_by_entry:
                continue
            ei = i + entry_lag
            if ei >= n:
                continue
            x, r = out_by_entry[i]
            entry_price = float(openp[ei]) if entry_lag else float(close[ei])
            a = atr[i]
            mom = (close[i] / close[i - lookback] - 1.0) if (i >= lookback and close[i - lookback] > 0) else 0.0
            sig_d = int(ords[i])
            factors = dict(
                rs=mom - spy_ret.get(sig_d, 0.0), momentum=mom,
                setup=float((lower[i] - close[i]) / a), movement=float(a / entry_price),
                exhaustion=float(abs(ret_pct[i] if not np.isnan(ret_pct[i]) else 0.0) * rel_vol[i]),
                volexp=float(atr_exp[i]),
                trend=float((close[i] / sma50[i] - 1.0) if (not np.isnan(sma50[i]) and sma50[i] > 0) else 0.0),
                sector=sec_ret.get(etf, {}).get(sig_d, 0.0),
            )
            raw.append(dict(sig_d=sig_d, entry_ord=int(ords[ei]), exit_ord=int(ords[x]),
                            ret=float(r), sym=sym, entry_price=entry_price,
                            atr_pct=max(a / entry_price, ATR_FLOOR) if entry_price > 0 else ATR_FLOOR,
                            f=factors))

    by_day = defaultdict(list)                         # rank cross-sectionally per SIGNAL day
    for rec in raw:
        by_day[rec["sig_d"]].append(rec)
    entries = []
    for recs in by_day.values():
        z = {fac: _zscore(np.array([r["f"][fac] for r in recs])) for fac in FACTORS}
        comp = sum(profile.get(fac, 0.0) * z[fac] for fac in FACTORS)
        for rec, sc in zip(recs, comp):
            entries.append(Entry(rec["entry_ord"], rec["exit_ord"], rec["ret"], rec["sym"],
                                 float(sc), rec["entry_price"], rec["atr_pct"]))
    return entries


def factor_sweep(symbol_bars, spy_bars, sector_of=None, sector_bars=None,
                 profiles=EXPERIMENT_PROFILES, ks=(3, 5, 10), holds=(1, 2, 3, 5),
                 risks=(0.0075, 0.01), heat=0.02, lookback=60, entry_lag=1,
                 rsi_max=45.0) -> list:
    """Sweep (profile, hold, K, risk) under live occupancy/heat/risk with next-open
    fills. Returns ``[(label, stats)]`` sorted by RET/DD (best first)."""
    closes = build_closes(symbol_bars)
    rows = []
    for pname, w in profiles.items():
        for hold in holds:
            entries = build_factor_entries(symbol_bars, spy_bars, sector_of, sector_bars,
                                           w, hold, lookback, rsi_max, entry_lag)
            for K in ks:
                for risk in risks:
                    s = simulate_portfolio(entries, closes, K, "risk", risk, 1.2, 0.20,
                                           max_total_exposure=0.95, max_portfolio_heat=heat)
                    rows.append((f"{pname}_h{hold}d_K{K}_r{risk*100:g}%", s))
    rows.sort(key=lambda kv: _ret_dd(kv[1]), reverse=True)
    return rows


def format_factor_sweep(rows, top=20) -> str:
    lines = [f"  factor-ranked daily experiment (next-open fill, heat 2%, ATR-risk) — top {top} by RET/DD",
             f"  {'CONFIG':<26}{'RETURN':>9}{'MTM_DD':>9}{'RET/DD':>8}{'SHARPE':>8}{'TRADES':>8}",
             "  " + "-" * 68]
    for label, s in rows[:top]:
        lines.append(f"  {label:<26}{s['total_return']*100:>8.1f}%{s['mtm_max_drawdown']*100:>8.1f}%"
                     f"{_ret_dd(s):>8.1f}{s['sharpe']:>8.3f}{s['n_trades']:>8}")
    lines += ["",
              "  Returns are net of 8bps. 'Survives execution' = RET/DD and return hold up on",
              "  BOTH windows at the top of the table; a config that only ranks high on one is",
              "  window-luck. A genuine winner here is a real live-deployment candidate."]
    return "\n".join(lines)


def format_sim(stats: dict, max_positions: int) -> str:
    return "\n".join([
        f"  max_positions  : {max_positions} (one per symbol, {stats['sizing']} sizing, compounding)",
        f"  trades taken   : {stats['n_trades']}  (skipped {stats['n_skipped']} — slots full / dup symbol)",
        f"  total return   : {stats['total_return']*100:+.1f}%   final equity x{stats['final_equity']:.2f}",
        f"  max DD realized: {stats['max_drawdown']*100:.1f}%",
        f"  max DD mark-to-mkt: {stats['mtm_max_drawdown']*100:.1f}%   <- the honest worst-case",
        f"  daily Sharpe   : {stats['sharpe']:.3f}",
        "",
        "  MTM DD marks open positions every day (intra-hold troughs); it is the number",
        "  to size live risk against. Compare to the occupancy-free MC maxDD_p95 (~-63%).",
    ])


def main() -> None:
    p = argparse.ArgumentParser(description="Occupancy-realistic portfolio sim (research-only)")
    p.add_argument("symbols", nargs="*", help="symbols (overrides --universe)")
    p.add_argument("--universe", choices=list(UNIVERSES), default="neutral")
    p.add_argument("--days", type=int, default=540)
    p.add_argument("--end", default=None)
    p.add_argument("--timeframe", default="1Day")
    p.add_argument("--max-positions", type=int, nargs="*", default=[4],
                   help="open-position cap(s) to sweep (live default 4)")
    p.add_argument("--sizing", choices=["equal", "risk"], default="risk",
                   help="equal = equity/N; risk = live ATR-risk model (default)")
    p.add_argument("--compare-sizing", action="store_true",
                   help="sweep equal vs ATR-risk at rising risk_pct (the return/DD "
                        "frontier) at the first --max-positions cap")
    p.add_argument("--grid-sweep", action="store_true",
                   help="2-D sweep: --max-positions x --risk-grid, printing RET/DD, "
                        "return and drawdown matrices (the cap x budget tradeoff)")
    p.add_argument("--factor-sweep", action="store_true",
                   help="factor-ranked daily experiment: sweep profile x hold x K x risk "
                        "with next-open fills + occupancy/heat/risk (needs SPY + sector ETFs)")
    p.add_argument("--holds", type=int, nargs="*", default=[1, 2, 3, 5],
                   help="holding periods in bars/days for --factor-sweep")
    p.add_argument("--risks", type=float, nargs="*", default=[0.0075, 0.01],
                   help="risk_pct values for --factor-sweep")
    p.add_argument("--risk-grid", type=float, nargs="*",
                   default=[0.005, 0.0075, 0.01, 0.015, 0.02],
                   help="risk_pct values for --grid-sweep")
    p.add_argument("--risk-pct", type=float, default=0.005)
    p.add_argument("--stop-atr", type=float, default=1.2)
    p.add_argument("--max-pos-frac", type=float, default=0.20,
                   help="per-position cap (live MAX_POSITION_VALUE_PCT)")
    p.add_argument("--max-total-exposure", type=float, default=0.95,
                   help="sum-of-notionals cap (live MAX_TOTAL_EXPOSURE_PCT); 0 = off")
    p.add_argument("--max-heat", type=float, default=0.02,
                   help="portfolio heat cap = sum risk-to-stop (live MAX_PORTFOLIO_HEAT_PCT); 0 = off")
    p.add_argument("--rsi-max", type=float, default=45.0)
    p.add_argument("--min-price", type=float, default=5.0)
    p.add_argument("--min-dollar-vol", type=float, default=20e6)
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
    print(f"Portfolio sim | universe={args.universe} | {len(symbol_bars)}/{len(symbols)} names | "
          f"{start}->{end_s} | {args.timeframe} | sizing={args.sizing}")

    if args.factor_sweep:
        spy_bars = _fetch("SPY")
        sector_bars = {e: b for e in SECTOR_ETFS if (b := _fetch(e)) is not None}
        if spy_bars is None:
            print("SPY fetch failed — needed for rs/sector factors.")
            return
        print(f"  factor sweep | profiles={list(EXPERIMENT_PROFILES)} holds={args.holds} "
              f"K={args.max_positions} risks={args.risks} | heat<={args.max_heat:g} | "
              f"next-open fill\n")
        rows = factor_sweep(symbol_bars, spy_bars, ALL_SECTOR_OF, sector_bars,
                            ks=args.max_positions, holds=args.holds, risks=args.risks,
                            heat=(args.max_heat or None), rsi_max=args.rsi_max)
        print(format_factor_sweep(rows))
        return

    entries = build_entries(symbol_bars, args.rsi_max)
    closes = build_closes(symbol_bars)
    mte = args.max_total_exposure or None      # 0 -> off
    heat = args.max_heat or None
    print(f"  {len(entries)} reversion candidates across the window | "
          f"caps: exposure<={args.max_total_exposure:g} heat<={args.max_heat:g} (live governor)\n")
    if args.grid_sweep:
        grid = grid_sweep(entries, closes, args.max_positions, args.risk_grid,
                          stop_atr_mult=args.stop_atr, max_pos_frac=args.max_pos_frac,
                          max_total_exposure=mte, max_portfolio_heat=heat)
        print(format_grid(grid, args.max_positions, args.risk_grid))
        return
    if args.compare_sizing:
        n = args.max_positions[0]
        rows = compare_sizing(entries, closes, n, stop_atr_mult=args.stop_atr,
                              max_pos_frac=args.max_pos_frac,
                              max_total_exposure=mte, max_portfolio_heat=heat)
        print(format_sizing_comparison(rows, n))
        return
    for n in args.max_positions:
        stats = simulate_portfolio(entries, closes, n, args.sizing, args.risk_pct,
                                   args.stop_atr, args.max_pos_frac, mte, heat)
        print(format_sim(stats, n))
        print()


if __name__ == "__main__":
    main()
