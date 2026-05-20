import os
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv
from reversion_bot.walkforward import run_walkforward_backtest


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

    from alpaca.data.timeframe import TimeFrameUnit

    if timeframe == 'Minute':
        tf = TimeFrame.Minute
    elif timeframe == '5Min':
        tf = TimeFrame(5, TimeFrameUnit.Minute)
    elif timeframe == 'Hour':
        tf = TimeFrame.Hour
    elif timeframe == 'Day':
        tf = TimeFrame.Day
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

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