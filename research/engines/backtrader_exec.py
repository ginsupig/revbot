"""Backtrader realistic-fill TEMPLATE — run with `pip install backtrader`.

VectorBT finds candidates fast but its fills are idealized. Once a config clears
the gate, re-run it here for execution realism (commission, slippage, order
types, intrabar fills, sizing) before any deploy — the last check between a
paper edge and a live one. NOT imported by the test suite; guarded import.

Wiring sketch:

    import backtrader as bt
    class FlushBounce(bt.Strategy):
        params = dict(flush_pct=0.05, drop_window=3, rsi_max=35,
                      stop_atr=1.2, target_atr=3.5, trail_atr=1.5)
        def __init__(self):
            self.rsi = bt.ind.RSI(period=14); self.atr = bt.ind.ATR(period=14)
        def next(self):
            ...  # entry on flush+oversold; bracket+trailing exit matching reference.EXIT
    cerebro = bt.Cerebro()
    cerebro.broker.setcommission(commission=0.0004)
    cerebro.broker.set_slippage_perc(0.0004)
    # add data, strategy, analyzers (TradeAnalyzer, DrawDown), run.
"""
from __future__ import annotations

try:                                # guarded: optional, machine-side dependency
    import backtrader as bt         # noqa: F401
    HAVE_BACKTRADER = True
except Exception:                   # pragma: no cover
    bt = None
    HAVE_BACKTRADER = False


def require_backtrader() -> None:
    if not HAVE_BACKTRADER:
        raise ImportError(
            "backtrader is not installed. `pip install backtrader` on your "
            "research machine to run the realistic-fill validation."
        )
