"""VectorBT fast-sweep TEMPLATE — run on a machine with `pip install vectorbt`.

Mirrors research.engines.reference but vectorized, so thousands of flush x RSI x
RVOL x ATR-exit combos run in seconds. Feed its per-config returns into
research.pipeline.evaluate_sweep so the conclusion stays gated (deflation +
walk-forward + holdout) no matter how big the grid. NOT imported by the test
suite (vectorbt isn't a sandbox dep); the import is guarded so the module loads.

Wiring sketch (fill in for your data):

    import vectorbt as vbt
    price = vbt.YFData.download(symbols).get("Close")   # or Alpaca bars
    # entries: N-bar flush + oversold; exits: ATR stop/target via from_signals
    rsi = vbt.RSI.run(price, window=rsi_len).rsi
    drop = price / price.shift(drop_window) - 1
    entries = (drop <= -flush_pct) & (rsi <= rsi_max)
    pf = vbt.Portfolio.from_signals(
        price, entries, exits=False,
        sl_stop=stop_atr_frac, tp_stop=target_atr_frac, fees=0.0004, slippage=0.0004,
    )
    returns_by_config = {key: pf[key].trades.returns.values for key in pf.wrapper.columns}
    # then: research.pipeline.evaluate_sweep(returns_by_config, holdout_returns)
"""
from __future__ import annotations

try:                                # guarded: optional, machine-side dependency
    import vectorbt as vbt          # noqa: F401
    HAVE_VECTORBT = True
except Exception:                   # pragma: no cover
    vbt = None
    HAVE_VECTORBT = False


def require_vectorbt() -> None:
    if not HAVE_VECTORBT:
        raise ImportError(
            "vectorbt is not installed. `pip install vectorbt` on your research "
            "machine, then use this template; the gating still runs via "
            "research.pipeline.evaluate_sweep on the returns it produces."
        )
