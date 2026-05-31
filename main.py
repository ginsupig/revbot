import os
import asyncio
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from reversion_bot.config import (
    ReversionConfig,
    RiskConfig,
    ExecutionConfig,
    PerformanceConfig,
    PortfolioConfig,
)
from reversion_bot.service import ReversionService
from reversion_bot.execution import AlpacaExecutor
from reversion_bot.models import PositionPlan
from reversion_bot.governor import ExecutionGovernor
from reversion_bot.portfolio import PortfolioState
from reversion_bot.allowlist import parse_allowlist, filter_symbols
from run_real_backtest import fetch_alpaca_bars

# --- Helper Functions ---

def is_market_open(executor) -> bool:
    try:
        clock = executor.client.get_clock()
        return bool(clock.is_open)
    except Exception:
        return False

def is_morning_blackout() -> bool:
    """Block new entries during the first 30 minutes after open (8:30–9:00 AM CT)."""
    now_ct = datetime.now(ZoneInfo("America/Chicago"))
    return now_ct.hour < 9

def get_account_equity(executor):
    account = executor.client.get_account()
    return float(account.equity)

def parse_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

def get_fetch_days(timeframe: str, lookback: int) -> int:
    tf = timeframe.strip().lower()
    # Estimate days needed to cover the lookback window
    bars_per_day = 78 if "5" in tf else 390
    return max(3, int((lookback / bars_per_day) * 3) + 2)

def build_fetch_window(timeframe: str, lookback: int) -> tuple[str, str]:
    now_utc = datetime.now(timezone.utc)
    fetch_days = get_fetch_days(timeframe, lookback)
    start_utc = now_utc - timedelta(days=fetch_days)
    return (
        start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

async def fetch_bars_for_symbol(symbol: str, timeframe: str, lookback: int):
    start, end = build_fetch_window(timeframe, lookback)
    return await asyncio.to_thread(fetch_alpaca_bars, symbol, start, end, timeframe)

async def evaluate_symbol_only(symbol, lookback, timeframe, service, executor):
    try:
        if await asyncio.to_thread(executor.has_open_long_position, symbol):
            return None

        bars = await fetch_bars_for_symbol(symbol, timeframe, lookback)
        if bars is None or len(bars) < lookback:
            return None

        bars = bars.tail(lookback)
        account_equity = await asyncio.to_thread(get_account_equity, executor)
        result = await asyncio.to_thread(service.evaluate_symbol, symbol, bars, account_equity)
        result["_account_equity"] = account_equity
        return result
    except Exception as e:
        print(f"[ERROR] {symbol}: {e}")
        return None

async def execute_candidates(governor, executor, portfolio_state, candidates):
    try:
        if not candidates:
            print("[INFO] No candidates to execute.")
            return

        trades_this_cycle = 0
        for candidate in candidates:
            if governor.approve(candidate, executor=executor, trades_executed_this_cycle=trades_this_cycle):
                print(f"[INFO] Executing trade for {candidate['symbol']}.")
                await asyncio.to_thread(executor.submit_order, candidate)
                portfolio_state.update(candidate)
                trades_this_cycle += 1
            else:
                print(f"[INFO] Candidate {candidate['symbol']} not approved by governor.")
    except Exception as e:
        print(f"[ERROR] Failed to execute candidates: {e}")

# --- EOD Liquidation ---

def is_eod_liquidation_window() -> bool:
    """True during the final stretch before the regular close (3:00 PM CT / 4:00 PM ET).

    Window length is configurable via EOD_LIQUIDATION_MINUTES (minutes before close
    to start flattening; default 10). Note: assumes the regular 3:00 PM CT close and
    does not adjust for half-day early closes.
    """
    lead_minutes = int(os.getenv("EOD_LIQUIDATION_MINUTES", 10))
    now_ct = datetime.now(ZoneInfo("America/Chicago"))
    close_ct = now_ct.replace(hour=15, minute=0, second=0, microsecond=0)
    start_ct = close_ct - timedelta(minutes=lead_minutes)
    return start_ct <= now_ct < close_ct


async def liquidate_all_positions(executor):
    """Cancel all open orders then market-sell every open position."""
    print("[EOD] Starting end-of-day liquidation...")

    # Cancel all open orders first so stops/limits don't interfere
    try:
        executor.client.cancel_all_orders()
        print("[EOD] All open orders cancelled.")
    except Exception as e:
        print(f"[EOD] Failed to cancel orders: {e}")

    # Market-sell every open position
    try:
        positions = executor.client.list_positions()
    except Exception as e:
        print(f"[EOD] Failed to fetch positions: {e}")
        return

    if not positions:
        print("[EOD] No open positions to liquidate.")
        return

    for pos in positions:
        symbol = str(pos.symbol).upper()
        qty = int(float(pos.qty))
        try:
            executor.client.submit_order(
                symbol=symbol,
                qty=qty,
                side="sell",
                type="market",
                time_in_force="day",
            )
            print(f"[EOD] Market sell submitted: {symbol} x{qty}")
        except Exception as e:
            print(f"[EOD] Failed to submit sell for {symbol}: {e}")


# --- Main Entry Point ---

async def main():
    load_dotenv()

    # 1. Environment and Credentials
    api_key = os.getenv("APCA_API_KEY_ID")
    api_secret = os.getenv("APCA_API_SECRET_KEY")
    base_url = os.getenv("APCA_API_BASE_URL")
    
    if not api_key or not api_secret:
        raise ValueError("Missing Alpaca API credentials in .env file.")

    timeframe = os.getenv("TRADE_TIMEFRAME", "5Min")
    lookback = int(os.getenv("TRADE_LOOKBACK", 160))
    poll_interval = int(os.getenv("TRADE_POLL_INTERVAL", 30))

    # 2. INITIALIZE CONFIGURATIONS FIRST
    strategy_config = ReversionConfig(
        min_history=lookback,
        min_dollar_volume=float(os.getenv("MIN_DOLLAR_VOLUME", 750000.0)),
        min_price=float(os.getenv("MIN_PRICE", 5.0)),
        require_reclaim_lb1=parse_bool(os.getenv("REQUIRE_RECLAIM_LB1", "False")),
        use_vwap_filter=parse_bool(os.getenv("USE_VWAP_FILTER", "False")),
        use_trend_filter=parse_bool(os.getenv("USE_TREND_FILTER", "False")),
    )

    risk_config = RiskConfig(
        risk_per_trade_pct=float(os.getenv("RISK_PER_TRADE_PCT", 0.005)),
        max_position_value_pct=float(os.getenv("MAX_POSITION_VALUE_PCT", 0.15)),
        min_rr=float(os.getenv("MIN_RR", 1.5)),
    )

    if not base_url:
        raise ValueError("Missing APCA_API_BASE_URL in .env file.")
    exec_config = ExecutionConfig(
        paper=True if "paper" in base_url.lower() else False,
        base_url=base_url,
    )

    # 3. Initialize Executor and Symbol List
    executor = AlpacaExecutor(api_key, api_secret, exec_config)
    
    # Attempt dynamic scan
    symbols = executor.scan_symbols(
        min_price=strategy_config.min_price,
        min_dollar_volume=strategy_config.min_dollar_volume,
        max_count=20
    )

    # Patched: Clean fallback watchlist with high-probability mean-reversion tickers
    if not symbols:
        print("[WARN] Scanner returned no symbols. Using fallback watchlist.")
        symbols = ["MU", "WDC", "ASTS", "NVDA", "AMD", "SMCI", "CRWD", "AAPL", "TSLA", "APP", "META", "INOD"]

    # 3b. Per-symbol allowlist gate (autotune_run.py writes TRADE_ALLOWLIST with
    # only the names whose OOS profit factor cleared the threshold). Unset means
    # gating is off; explicitly empty means nothing qualified -> trade nothing.
    allowlist = parse_allowlist(os.getenv("TRADE_ALLOWLIST"))
    if allowlist is not None:
        symbols, dropped = filter_symbols(symbols, allowlist)
        if dropped:
            print(f"[ALLOWLIST] Skipping {len(dropped)} name(s) not in TRADE_ALLOWLIST: {', '.join(dropped)}")
        if not symbols:
            print("[ALLOWLIST] No symbols passed the allowlist gate — nothing to trade this session.")

    # 4. Initialize Remaining Services
    perf_config = PerformanceConfig(state_dir="state/performance")
    portfolio_config = PortfolioConfig(
        max_open_positions=int(os.getenv("MAX_OPEN_POSITIONS", 4)),
        max_trades_per_cycle=int(os.getenv("MAX_TRADES_PER_CYCLE", 2)),
        max_total_exposure_pct=float(os.getenv("MAX_TOTAL_EXPOSURE_PCT", 0.95)),
        max_positions_per_bucket=int(os.getenv("MAX_POSITIONS_PER_BUCKET", 2)),
        max_portfolio_heat_pct=float(os.getenv("MAX_PORTFOLIO_HEAT_PCT", 0.02)),
        max_daily_new_positions=int(os.getenv("MAX_DAILY_NEW_POSITIONS", 12)),
        max_reversion_positions=int(os.getenv("MAX_REVERSION_POSITIONS", 3)),
        max_trend_positions=int(os.getenv("MAX_TREND_POSITIONS", 2)),
        max_trendfail_positions=int(os.getenv("MAX_TRENDFAIL_POSITIONS", 1)),
        max_positions_per_regime=int(os.getenv("MAX_POSITIONS_PER_REGIME", 3)),
        drawdown_pause_pct=float(os.getenv("DRAWDOWN_PAUSE_PCT", 0.025)),
        reduce_size_after_drawdown_pct=float(os.getenv("REDUCE_SIZE_AFTER_DRAWDOWN_PCT", 0.01)),
        reduced_risk_multiplier=float(os.getenv("REDUCED_RISK_MULTIPLIER", 0.60)),
        symbol_cooldown_minutes=int(os.getenv("SYMBOL_COOLDOWN_MINUTES", 30)),
    )
    
    service = ReversionService(strategy_config, risk_config, perf_config)
    portfolio_state = PortfolioState()
    governor = ExecutionGovernor(config=portfolio_config, portfolio_state=portfolio_state)

    print(f"--- REVERSION BOT STARTING ---")
    print(f"Timeframe: {timeframe} | Lookback: {lookback} | Mode: {'PAPER' if exec_config.paper else 'LIVE'}")
    print(f"Monitoring: {', '.join(symbols)}")

    try:
        while True:
            market_open = await asyncio.to_thread(is_market_open, executor)
            
            if not market_open:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Market Closed. Sleeping...")
                await asyncio.sleep(poll_interval)
                continue

            if is_eod_liquidation_window():
                await liquidate_all_positions(executor)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] EOD liquidation done. Sleeping...")
                await asyncio.sleep(poll_interval)
                continue

            if is_morning_blackout():
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Morning blackout (9:30-10:00 ET). Sleeping...")
                await asyncio.sleep(poll_interval)
                continue

            eval_tasks = [
                evaluate_symbol_only(s, lookback, timeframe, service, executor) 
                for s in symbols
            ]
            raw_results = await asyncio.gather(*eval_tasks)
            all_results = [r for r in raw_results if r is not None]
            candidates = [r for r in all_results if r.get("go_long")]

            # Keep equity tracking current for drawdown checks
            if all_results:
                try:
                    eq = await asyncio.to_thread(get_account_equity, executor)
                    portfolio_state.update_equity(eq)
                except Exception:
                    pass

            await execute_candidates(governor, executor, portfolio_state, candidates)
            
            await asyncio.sleep(poll_interval)
            
    except KeyboardInterrupt:
        print("\n[STOP] Shutting down...")

if __name__ == "__main__":
    asyncio.run(main())