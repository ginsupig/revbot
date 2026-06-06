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
from run_real_backtest import fetch_alpaca_bars

# --- Trading Universe ---
# Leveraged tech universe (replaces the dynamic Alpaca market scan).
#   3x sector/index ETFs: TQQQ (Nasdaq-100), SOXL (Semiconductors), TECL (Tech Select)
#   2x single-stock ETFs: NVDL (NVDA), TSLL (TSLA), AAPU (AAPL), METU (META),
#                         GGLL (GOOGL), MSFU (MSFT), AMZU (AMZN)
# Note: US single-stock leveraged ETFs top out at 2x; only index/sector ETFs offer 3x.
LEVERAGED_TECH_UNIVERSE = [
    "TQQQ", "SOXL", "TECL",
    "NVDL", "TSLL", "AAPU", "METU", "GGLL", "MSFU", "AMZU",
]

# --- Helper Functions ---

def is_market_open(executor) -> bool:
    try:
        clock = executor.client.get_clock()
        return bool(clock.is_open)
    except Exception:
        return False

def parse_session_time(env_name: str, default: str) -> tuple[int, int]:
    """Parse an HH:MM (24h, America/Chicago) schedule value from the environment.

    Falls back to ``default`` if the value is missing or malformed.
    """
    raw = os.getenv(env_name, default)
    try:
        hh, mm = str(raw).strip().split(":")
        hour, minute = int(hh), int(mm)
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return hour, minute
    except (ValueError, AttributeError):
        pass
    dh, dm = default.split(":")
    print(f"[WARN] Invalid {env_name}={raw!r}; falling back to {default}.")
    return int(dh), int(dm)


def get_session_start_ct() -> datetime:
    """Daily launch time (America/Chicago). Configurable via SESSION_START (HH:MM)."""
    hour, minute = parse_session_time("SESSION_START", "09:00")
    now_ct = datetime.now(ZoneInfo("America/Chicago"))
    return now_ct.replace(hour=hour, minute=minute, second=0, microsecond=0)


def get_session_close_ct() -> datetime:
    """Daily close time (America/Chicago). Configurable via SESSION_END (HH:MM)."""
    hour, minute = parse_session_time("SESSION_END", "15:00")
    now_ct = datetime.now(ZoneInfo("America/Chicago"))
    return now_ct.replace(hour=hour, minute=minute, second=0, microsecond=0)


def is_before_session_start() -> bool:
    """Block new entries before the scheduled launch time (default 9:00 AM CT)."""
    return datetime.now(ZoneInfo("America/Chicago")) < get_session_start_ct()

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

def is_session_closing() -> bool:
    """True once the scheduled close routine begins, and for the rest of the day.

    Flattening starts EOD_LIQUIDATION_MINUTES (default 10) before the scheduled
    close time (SESSION_END, default 3:00 PM CT / 4:00 PM ET) and stays active
    through to the regular market close so nothing is re-opened afterward.
    Note: assumes the configured close and does not adjust for half-day early closes.
    """
    lead_minutes = int(os.getenv("EOD_LIQUIDATION_MINUTES", 10))
    now_ct = datetime.now(ZoneInfo("America/Chicago"))
    close_ct = get_session_close_ct()
    start_ct = close_ct - timedelta(minutes=lead_minutes)
    return now_ct >= start_ct


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
        # Entry-signal thresholds. These share the same env var names as
        # autotune_run.py / walk-forward so a tuned .env applies live too.
        # Defaults match the ReversionConfig dataclass (unset == unchanged behavior).
        band_length=int(os.getenv("BAND_LENGTH", 20)),
        band_std_1=float(os.getenv("BAND_STD_1", 1.0)),
        band_std_2=float(os.getenv("BAND_STD_2", 2.0)),
        ri_threshold=float(os.getenv("RI_THRESHOLD", -0.5)),
        rsi_max=float(os.getenv("RSI_MAX", 48.0)),
        adx_max=float(os.getenv("ADX_MAX", 40.0)),
        max_vwap_extension_pct=float(os.getenv("MAX_VWAP_EXTENSION_PCT", 0.012)),
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
        # Leveraged ETFs trade with wide spreads -> default to marketable limit entries.
        use_limit_entry=parse_bool(os.getenv("USE_LIMIT_ENTRY", "True")),
        limit_entry_offset_bps=float(os.getenv("LIMIT_ENTRY_OFFSET_BPS", 8.0)),
        tif=os.getenv("TRADE_TIF", "day"),
    )

    # 3. Initialize Executor and Symbol List
    executor = AlpacaExecutor(api_key, api_secret, exec_config)

    # Fixed leveraged tech universe (dynamic scan replaced). Override via TRADE_SYMBOL.
    env_symbols = os.getenv("TRADE_SYMBOL", "")
    if env_symbols:
        symbols = [s.strip().upper() for s in env_symbols.split(",") if s.strip()]
    else:
        symbols = list(LEVERAGED_TECH_UNIVERSE)

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

    start_ct = get_session_start_ct()
    close_ct = get_session_close_ct()
    lead_minutes = int(os.getenv("EOD_LIQUIDATION_MINUTES", 10))

    print(f"--- REVERSION BOT STARTING ---")
    print(f"Timeframe: {timeframe} | Lookback: {lookback} | Mode: {'PAPER' if exec_config.paper else 'LIVE'}")
    print(f"Monitoring: {', '.join(symbols)}")
    print(
        f"Schedule (CT): launch {start_ct.strftime('%H:%M')} | "
        f"flatten {(close_ct - timedelta(minutes=lead_minutes)).strftime('%H:%M')} | "
        f"close {close_ct.strftime('%H:%M')}"
    )

    try:
        while True:
            market_open = await asyncio.to_thread(is_market_open, executor)
            
            if not market_open:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Market Closed. Sleeping...")
                await asyncio.sleep(poll_interval)
                continue

            if is_session_closing():
                await liquidate_all_positions(executor)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Session closed (>= {os.getenv('SESSION_END', '15:00')} CT). Flattened. Sleeping...")
                await asyncio.sleep(poll_interval)
                continue

            if is_before_session_start():
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Pre-launch (< {os.getenv('SESSION_START', '09:00')} CT). Sleeping...")
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