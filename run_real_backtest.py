import os
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from dotenv import load_dotenv
from reversion_bot.walkforward import run_walkforward_backtest


def _parse_timeframe(timeframe) -> TimeFrame:
    """Normalize a timeframe string to an alpaca-py TimeFrame.

    Accepts both the bare forms ('Day', 'Minute', 'Hour', '5Min') and Alpaca's
    own '<n><unit>' strings ('1Day', '1Min', '1Hour'). The latter was the live
    trap: MARKET_REGIME_TIMEFRAME defaulted to '1Day', which the old exact-match
    chain rejected — so the regime fetch raised every cycle and the filter
    silently failed open (never suppressed a long).
    """
    key = str(timeframe).strip().lower()
    mapping = {
        "minute": TimeFrame.Minute, "1min": TimeFrame.Minute, "1minute": TimeFrame.Minute,
        "5min": TimeFrame(5, TimeFrameUnit.Minute), "15min": TimeFrame(15, TimeFrameUnit.Minute),
        "hour": TimeFrame.Hour, "1hour": TimeFrame.Hour,
        "day": TimeFrame.Day, "1day": TimeFrame.Day, "daily": TimeFrame.Day,
    }
    tf = mapping.get(key)
    if tf is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return tf


def _apply_http_timeout(client, timeout):
    """Inject a default per-request timeout into an alpaca-py client's session.

    alpaca-py issues HTTP requests with no timeout, so a hung connection blocks
    the worker thread forever (seen live: 'Read timed out (read timeout=None)',
    which froze a poll cycle). Wrap the session's request() to supply a timeout
    whenever one isn't already set, so a stalled call aborts and raises — which
    every caller already handles (skip the symbol / fail-open). Best-effort.
    """
    if timeout is None or timeout <= 0:
        return client
    session = getattr(client, "_session", None)
    if session is None:
        return client
    original = session.request

    def _request_with_timeout(method, url, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = timeout
        return original(method, url, **kwargs)

    session.request = _request_with_timeout
    return client


def fetch_alpaca_bars(symbol, start, end, timeframe='1Day'):
    load_dotenv()
    api_key = os.getenv('APCA_API_KEY_ID')
    api_secret = os.getenv('APCA_API_SECRET_KEY')
    base_url = os.getenv('APCA_API_BASE_URL')

    if not api_key or not api_secret:
        raise ValueError("Missing Alpaca API credentials. Check your .env file.")

    if base_url and base_url.startswith("https://data"):
        client = StockHistoricalDataClient(api_key, api_secret, url_override=base_url)
    else:
        client = StockHistoricalDataClient(api_key, api_secret)

    # Bound every HTTP call so a hung data connection can't freeze a poll cycle.
    _apply_http_timeout(client, float(os.getenv("ALPACA_HTTP_TIMEOUT", "15")))

    tf = _parse_timeframe(timeframe)

    req_kwargs = {
        'symbol_or_symbols': symbol,
        'timeframe': tf,
    }
    if start is not None:
        req_kwargs['start'] = start
    if end is not None:
        req_kwargs['end'] = end

    req = StockBarsRequest(
        **req_kwargs,
        feed='iex',
    )

    bars = client.get_stock_bars(req).df
    bars = bars.reset_index()
    bars = bars.rename(columns={'timestamp': 'date'})
    return bars


def main():
    symbol = 'AAPL'
    start = '2023-01-01'
    end = '2024-04-01'
    timeframe = '5Min'
    bars = fetch_alpaca_bars(symbol, start, end, timeframe)
    print(f"Fetched {len(bars)} bars for {symbol} ({timeframe} bars)")
    bars = bars.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'})
    param_grid = {
        'band_length': [15, 20, 30],
        'band_std_1': [0.75, 1.0, 1.25],
        'band_std_2': [1.75, 2.0, 2.5],
        'rsi_max':    [40.0, 45.0, 50.0],
    }
    results, oos_metrics, _ = run_walkforward_backtest(bars, param_grid=param_grid, n_splits=5)
    print("Backtest complete.")
    for i, res in enumerate(results):
        print(f"Fold {i+1} last rows:")
        print(res.tail(10))


if __name__ == "__main__":
    main()