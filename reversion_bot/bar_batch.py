"""Split a multi-symbol Alpaca bars frame into per-symbol frames.

alpaca-py's ``get_stock_bars(...).df`` returns a MultiIndex (symbol, timestamp)
DataFrame when asked for several symbols in one request. The batched live fetch
(``run_real_backtest.fetch_alpaca_bars_batch``) needs to hand each symbol a frame
shaped exactly like the single-symbol path — a flat frame with a ``date`` column.

Kept here, pandas-only and free of any alpaca import, so the reshape is
unit-testable without SDK/network (the fetch itself needs creds; this doesn't).
"""
from __future__ import annotations


def split_bars_by_symbol(df, single_symbol=None) -> dict:
    """Reshape a (multi-symbol) bars frame into ``{symbol: DataFrame}``.

    Accepts the raw ``.df`` (MultiIndex symbol/timestamp). Resets the index,
    renames ``timestamp`` -> ``date`` (matching ``fetch_alpaca_bars``), and groups
    by the ``symbol`` column into one flat frame per name. If the frame has no
    ``symbol`` column (the single-symbol case) and ``single_symbol`` is given, the
    whole frame is returned under that key. Empty / None -> ``{}``.
    """
    if df is None or getattr(df, "empty", True):
        return {}
    df = df.reset_index()
    if "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "date"})
    out: dict = {}
    if "symbol" in df.columns:
        for sym, group in df.groupby("symbol"):
            out[str(sym)] = group.reset_index(drop=True)
    elif single_symbol is not None:
        out[single_symbol] = df
    return out
