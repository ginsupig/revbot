import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from research.run import run_gated_sweep, format_report, UNIVERSES
from research.pipeline import PipelineResult


def _bars_with_flush(n=400, seed=3):
    """Random walk with periodic sharp flushes that then bounce (gives the
    flush-bounce engine real trades to sweep)."""
    rng = np.random.default_rng(seed)
    close = [100.0]
    for d in range(1, n):
        if d % 25 == 0:
            close.append(close[-1] * 0.94)          # -6% flush
        elif d % 25 in (1, 2, 3):
            close.append(close[-1] * 1.02)          # bounce
        else:
            close.append(close[-1] * (1 + rng.normal(0, 0.004)))
    close = np.array(close)
    return pd.DataFrame({"open": close, "high": close * 1.005,
                         "low": close * 0.995, "close": close,
                         "volume": np.full(n, 1e6)})


def _stub_fetch(seed_base=0):
    """Injectable fetch_fn: deterministic synthetic bars per symbol, no broker."""
    def _fetch(sym, start, end, timeframe):
        return _bars_with_flush(seed=seed_base + sum(ord(c) for c in sym))
    return _fetch


def test_run_gated_sweep_returns_pipeline_result():
    res = run_gated_sweep(_stub_fetch(), ["MU", "ASTS", "WDC"],
                          "2024-01-01", "2025-01-01", "5Min",
                          flush_grid=[0.04, 0.06], rsi_grid=[50, 100],
                          drop_window=1, holdout_frac=0.25)
    assert isinstance(res, PipelineResult)
    # 2x2 grid -> 4 trials swept
    assert res.n_trials == 4
    assert res.chosen is not None
    # verdict is one of the known terminal strings
    assert res.verdict.startswith("DEPLOY-CANDIDATE") or res.verdict.startswith("REJECT")


def test_run_gated_sweep_skips_short_series():
    def _short_fetch(sym, start, end, timeframe):
        return _bars_with_flush(n=30)               # below min_bars=80
    res = run_gated_sweep(_short_fetch, ["MU"], "2024-01-01", "2025-01-01",
                          "5Min", flush_grid=[0.05], rsi_grid=[100], drop_window=1)
    # everything skipped -> empty sweep
    assert res.chosen is None
    assert "empty" in res.verdict.lower()


def test_run_gated_sweep_survives_fetch_errors():
    def _flaky(sym, start, end, timeframe):
        if sym == "BAD":
            raise RuntimeError("network down")
        return _bars_with_flush(seed=sum(ord(c) for c in sym))
    res = run_gated_sweep(_flaky, ["BAD", "MU", "ASTS"], "2024-01-01",
                          "2025-01-01", "5Min", flush_grid=[0.05],
                          rsi_grid=[100], drop_window=1)
    # BAD raised but the others still swept
    assert res.chosen is not None


def test_format_report_renders_evidence():
    res = run_gated_sweep(_stub_fetch(), ["MU", "ASTS"], "2024-01-01",
                          "2025-01-01", "5Min", flush_grid=[0.05],
                          rsi_grid=[100], drop_window=1)
    txt = format_report(res)
    assert "VERDICT" in txt
    assert "deflated" in txt
    assert "holdout" in txt
    assert "breakeven cost" in txt


def test_format_report_handles_empty():
    empty = PipelineResult(None, 0, 0.0, 0.0, False, False, verdict="empty sweep")
    assert "no trades" in format_report(empty)


def test_universes_well_formed():
    assert "keepers" in UNIVERSES and "watchlist" in UNIVERSES
    assert UNIVERSES["keepers"] == ["ARM", "CRWD", "MU", "LRCX"]
    assert all(isinstance(s, str) for u in UNIVERSES.values() for s in u)
