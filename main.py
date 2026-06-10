import os
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
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
from reversion_bot.allowlist import (
    parse_allowlist,
    filter_symbols,
    parse_symbol_csv,
    apply_watchlist,
)
from reversion_bot.symbol_params import load_symbol_params, build_symbol_configs
from reversion_bot.single_instance import acquire_lock, release_lock
from reversion_bot.schedule import session_done
from reversion_bot.heartbeat import write_heartbeat
from reversion_bot.market_regime import (
    is_risk_off,
    suppress_longs_if_risk_off,
    suppress_longs_by_sector,
)
from reversion_bot.sectors import sector_for, etf_for_sector
from reversion_bot.channel import select_breakout_exits
from run_real_backtest import fetch_alpaca_bars

# Lock scoped to this checkout (not cwd), so a separate deployment can still
# run its own bot — only a duplicate of *this* one is refused.
LOCK_PATH = Path(__file__).resolve().parent / "state" / "revbot.lock"
HEARTBEAT_PATH = Path(__file__).resolve().parent / "state" / "heartbeat.json"

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

def is_session_over(executor) -> bool:
    """True once today's session has ended (drives the clean end-of-day exit).

    Uses the broker clock so it's correct on half-days/holidays. On any clock
    error it returns False — we'd rather keep sleeping than shut down blind.
    """
    try:
        clock = executor.client.get_clock()
        if bool(clock.is_open):
            return False
        now_ct = datetime.now(ZoneInfo("America/Chicago"))
        next_open_ct = clock.next_open.astimezone(ZoneInfo("America/Chicago"))
        return session_done(False, next_open_ct, now_ct)
    except Exception:
        return False

def get_account_equity(executor):
    account = executor.client.get_account()
    return float(account.equity)


def preflight_check(executor, paper: bool) -> bool:
    """Validate credentials + connectivity before the trading loop starts.

    One get_account() call confirms the API keys are authorized for THIS
    endpoint. On an auth failure we fail fast with a clear paper/live hint
    instead of letting every scan/order call fail cryptically all session.
    Returns True to proceed, False to abort.
    """
    mode = "PAPER" if paper else "LIVE"
    bar = "=" * 68
    try:
        account = executor.client.get_account()
    except Exception as e:
        msg = str(e).lower()
        print(f"\n{bar}")
        if any(t in msg for t in ("not authorized", "unauthorized", "forbidden", "401", "403")):
            print(f"[PREFLIGHT] Alpaca REJECTED your credentials for the {mode} endpoint.")
            print("[PREFLIGHT] Most likely the API key/secret don't match the base URL —")
            print("[PREFLIGHT] paper and live use SEPARATE keys. If APCA_API_BASE_URL points")
            print("[PREFLIGHT] at live, set your LIVE key/secret (and vice versa).")
        else:
            print(f"[PREFLIGHT] Could not reach Alpaca ({mode} endpoint): {e}")
        print(f"{bar}\n")
        return False

    status = getattr(account, "status", "?")
    equity = getattr(account, "equity", "?")
    bp = getattr(account, "buying_power", "?")
    blocked = bool(getattr(account, "trading_blocked", False)) or bool(
        getattr(account, "account_blocked", False)
    )
    shorting_enabled = getattr(account, "shorting_enabled", None)

    if not paper:
        print("*" * 68)
        print("*** LIVE TRADING — REAL MONEY. Orders will hit your funded account. ***")
        print("*" * 68)
    print(f"[PREFLIGHT] Connected to Alpaca {mode}: status={status} "
          f"equity={equity} buying_power={bp}")
    if blocked:
        print("[PREFLIGHT] ABORT: broker reports this account trading_blocked/account_blocked.")
        return False
    if shorting_enabled is False:
        print("[PREFLIGHT] WARNING: shorting is NOT enabled on this account — "
              "short signals will be rejected by the broker.")
    return True

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


async def evaluate_market_regime(symbol: str, timeframe: str, ema_length: int) -> bool:
    """Fetch the benchmark and decide if the market is risk-off (below trend).

    Fail-open: any fetch error or thin data returns False (risk-on) so a
    benchmark hiccup never blocks trading.
    """
    try:
        now_utc = datetime.now(timezone.utc)
        # Enough calendar days to cover ema_length daily bars (with slack for
        # weekends/holidays); harmless over-fetch for intraday timeframes.
        days = max(ema_length * 3, 90)
        start = (now_utc - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        end = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        bars = await asyncio.to_thread(fetch_alpaca_bars, symbol, start, end, timeframe)
        return is_risk_off(bars, ema_length)
    except Exception as e:
        print(f"[REGIME] benchmark fetch failed ({e}); treating as risk-on.")
        return False


# Sector-regime cache: the gate is a DAILY EMA, so a sector ETF's risk-off state
# can't change within the session faster than its daily bar. Refresh each proxy
# ETF at most hourly instead of every poll cycle (keeps the data load to a few
# ETF fetches/day rather than per-cycle). etf -> (epoch_fetched, risk_off).
_SECTOR_REGIME_CACHE: dict = {}
_SECTOR_REGIME_TTL_S = 3600


async def evaluate_sector_regime(active_sectors, timeframe: str, ema_length: int) -> dict:
    """Resolve {sector: risk_off} for the active sectors via their proxy ETFs.

    Each ETF is fetched at most once per _SECTOR_REGIME_TTL_S (reusing the
    fail-open evaluate_market_regime path); stale ETFs are refreshed concurrently.
    """
    import time as _time
    now = _time.time()
    etfs = {etf_for_sector(s) for s in active_sectors}
    etfs.discard(None)

    fresh: dict = {}
    stale = []
    for etf in etfs:
        cached = _SECTOR_REGIME_CACHE.get(etf)
        if cached and now - cached[0] < _SECTOR_REGIME_TTL_S:
            fresh[etf] = cached[1]
        else:
            stale.append(etf)

    if stale:
        results = await asyncio.gather(*[
            evaluate_market_regime(etf, timeframe, ema_length) for etf in stale
        ])
        for etf, ro in zip(stale, results):
            _SECTOR_REGIME_CACHE[etf] = (now, ro)
            fresh[etf] = ro

    regime = {}
    for sector in active_sectors:
        etf = etf_for_sector(sector)
        if etf in fresh:
            regime[sector] = fresh[etf]
    return regime

async def evaluate_symbol_only(symbol, lookback, timeframe, service, executor, short_bias=False):
    try:
        if await asyncio.to_thread(executor.has_open_position, symbol):
            return None

        bars = await fetch_bars_for_symbol(symbol, timeframe, lookback)
        if bars is None or len(bars) < lookback:
            return None

        bars = bars.tail(lookback)
        account_equity = await asyncio.to_thread(get_account_equity, executor)
        result = await asyncio.to_thread(service.evaluate_symbol, symbol, bars, account_equity, short_bias)
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
        signed_qty = int(float(pos.qty))
        qty = abs(signed_qty)
        if qty == 0:
            continue
        # Flatten in the closing direction: sell to exit a long, buy to cover a short.
        side = "sell" if signed_qty > 0 else "buy"
        try:
            executor.client.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type="market",
                time_in_force="day",
            )
            print(f"[EOD] Market {side} submitted: {symbol} x{qty}")
        except Exception as e:
            print(f"[EOD] Failed to submit {side} for {symbol}: {e}")


async def manage_channel_exits(executor, exec_config, lookback, timeframe):
    """Breakout exit: market-close any LONG that has reached the upper regression
    channel, capturing the run before the (wide) bracket target. Longs only; the
    broker-side stop + bracket target remain the safety net. Best-effort — any
    fetch/close error is logged and skipped so a hiccup never disrupts the loop.
    """
    try:
        positions = await asyncio.to_thread(executor.get_positions)
    except Exception as e:
        print(f"[CHANNEL-EXIT] position fetch failed ({e}); skipping.")
        return

    longs = [p for p in positions if _safe_qty(p) > 0]
    if not longs:
        return

    look = int(getattr(exec_config, "channel_lookback", 80))
    k = float(getattr(exec_config, "channel_k", 2.0))
    threshold = float(getattr(exec_config, "channel_exit_threshold", 0.80))

    # Fetch each held long's recent closes (held names are skipped by the entry
    # scan, so they aren't fetched there).
    closes_by_symbol = {}
    for p in longs:
        symbol = str(p.symbol).upper()
        try:
            bars = await fetch_bars_for_symbol(symbol, timeframe, max(lookback, look))
            if bars is not None and len(bars) >= look:
                closes_by_symbol[symbol] = bars["close"].to_numpy()
        except Exception as e:
            print(f"[CHANNEL-EXIT] {symbol} bar fetch failed ({e}); skipping.")

    for symbol in select_breakout_exits(longs, closes_by_symbol, threshold, look, k):
        try:
            await asyncio.to_thread(executor.close_long, symbol)
            print(f"[CHANNEL-EXIT] {symbol} reached the upper channel — taking the breakout, exiting.")
        except Exception as e:
            print(f"[CHANNEL-EXIT] {symbol} close failed: {e}")


def _safe_qty(position) -> float:
    try:
        return float(getattr(position, "qty", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


# --- Main Entry Point ---

async def main():
    load_dotenv()

    # 0. Single-instance guard: refuse to start if another bot from this
    # checkout is already running (prevents duplicate orders on the account).
    acquired, holder_pid = acquire_lock(LOCK_PATH)
    if not acquired:
        print(f"[LOCK] Another revbot instance is already running (PID {holder_pid}). Exiting.")
        return 0

    # 1. Environment and Credentials
    api_key = os.getenv("APCA_API_KEY_ID")
    api_secret = os.getenv("APCA_API_SECRET_KEY")
    base_url = os.getenv("APCA_API_BASE_URL")
    
    if not api_key or not api_secret:
        raise ValueError("Missing Alpaca API credentials in .env file.")

    timeframe = os.getenv("TRADE_TIMEFRAME", "5Min")
    lookback = int(os.getenv("TRADE_LOOKBACK", 160))
    poll_interval = int(os.getenv("TRADE_POLL_INTERVAL", 30))

    # Market-regime filter: when the benchmark is below its trend EMA (risk-off),
    # suppress NEW long entries. Dip-buying every oversold name in a market-wide
    # selloff is the losing trade; shorts and existing positions are unaffected.
    use_market_filter = parse_bool(os.getenv("USE_MARKET_REGIME_FILTER", "True"), default=True)
    regime_symbol = os.getenv("MARKET_REGIME_SYMBOL", "SPY")
    regime_timeframe = os.getenv("MARKET_REGIME_TIMEFRAME", "1Day")
    regime_ema = int(os.getenv("MARKET_REGIME_EMA", 50))
    # Favor-shorts mode (opt-in): when risk-off, also relax the short trigger so
    # the bot leans short into the downtrend instead of only sitting in cash.
    favor_shorts_risk_off = parse_bool(os.getenv("FAVOR_SHORTS_IN_RISK_OFF", "False"))
    # Sector-aware regime gate (opt-in): judge each long against ITS sector ETF's
    # trend (SMH/XLK/XLE/...), not just SPY — so a semis selloff suppresses semis
    # longs even while SPY holds. Default off = today's SPY-only behavior.
    # SECTOR_REGIME_COMBINE: "or" (sector OR SPY) | "sector" (sector signal only).
    use_sector_filter = parse_bool(os.getenv("USE_SECTOR_REGIME_FILTER", "False"))
    sector_combine = (os.getenv("SECTOR_REGIME_COMBINE", "or") or "or").strip().lower()

    # 2. INITIALIZE CONFIGURATIONS FIRST
    strategy_config = ReversionConfig(
        min_history=lookback,
        min_dollar_volume=float(os.getenv("MIN_DOLLAR_VOLUME", 750000.0)),
        min_price=float(os.getenv("MIN_PRICE", 5.0)),
        max_spread_bps=float(os.getenv("MAX_SPREAD_BPS", 40.0)),
        # Bollinger band entry zone (lb1/lb2 = sma -/+ band_std * std).
        band_length=int(os.getenv("BAND_LENGTH", 20)),
        band_std_1=float(os.getenv("BAND_STD_1", 1.0)),
        band_std_2=float(os.getenv("BAND_STD_2", 2.0)),
        # Indicator lookbacks.
        ri_length=int(os.getenv("RI_LENGTH", 20)),
        rsi_length=int(os.getenv("RSI_LENGTH", 14)),
        adx_length=int(os.getenv("ADX_LENGTH", 14)),
        trend_ema_length=int(os.getenv("TREND_EMA_LENGTH", 50)),
        # Long oversold gates.
        ri_threshold=float(os.getenv("RI_THRESHOLD", -0.5)),
        rsi_max=float(os.getenv("RSI_MAX", 48.0)),
        adx_max=float(os.getenv("ADX_MAX", 40.0)),
        adx_hard_max=float(os.getenv("ADX_HARD_MAX", 50.0)),
        rsi_hard_max=float(os.getenv("RSI_HARD_MAX", 70.0)),
        max_vwap_extension_pct=float(os.getenv("MAX_VWAP_EXTENSION_PCT", 0.012)),
        require_reclaim_lb1=parse_bool(os.getenv("REQUIRE_RECLAIM_LB1", "False")),
        require_bullish_close=parse_bool(os.getenv("REQUIRE_BULLISH_CLOSE", "False")),
        require_volume_expansion=parse_bool(os.getenv("REQUIRE_VOLUME_EXPANSION", "False")),
        use_vwap_filter=parse_bool(os.getenv("USE_VWAP_FILTER", "False")),
        use_trend_filter=parse_bool(os.getenv("USE_TREND_FILTER", "False")),
        # Volume confirmation.
        volume_lookback=int(os.getenv("VOLUME_LOOKBACK", 20)),
        volume_multiplier_min=float(os.getenv("VOLUME_MULTIPLIER_MIN", 1.0)),
        # Trend-fail strategy.
        trendfail_window=int(os.getenv("TRENDFAIL_WINDOW", 20)),
        trendfail_threshold=float(os.getenv("TRENDFAIL_THRESHOLD", 0.005)),
        # Short side (mirror) gates.
        enable_shorts=parse_bool(os.getenv("ENABLE_SHORTS", "True"), default=True),
        ri_short_threshold=float(os.getenv("RI_SHORT_THRESHOLD", 0.5)),
        rsi_min=float(os.getenv("RSI_MIN", 52.0)),
        rsi_hard_min=float(os.getenv("RSI_HARD_MIN", 30.0)),
        risk_off_rsi_min=float(os.getenv("RISK_OFF_RSI_MIN", 45.0)),
        risk_off_ri_short_threshold=float(os.getenv("RISK_OFF_RI_SHORT_THRESHOLD", 0.25)),
    )

    risk_config = RiskConfig(
        risk_per_trade_pct=float(os.getenv("RISK_PER_TRADE_PCT", 0.005)),
        max_position_value_pct=float(os.getenv("MAX_POSITION_VALUE_PCT", 0.15)),
        min_rr=float(os.getenv("MIN_RR", 1.5)),
        # Per-style ATR stop/target multiples.
        stop_atr_multiple=float(os.getenv("STOP_ATR_MULTIPLE", 1.20)),
        target_atr_multiple=float(os.getenv("TARGET_ATR_MULTIPLE", 2.00)),
        trend_stop_atr_multiple=float(os.getenv("TREND_STOP_ATR_MULTIPLE", 1.20)),
        trend_target_atr_multiple=float(os.getenv("TREND_TARGET_ATR_MULTIPLE", 3.00)),
        trendfail_stop_atr_multiple=float(os.getenv("TRENDFAIL_STOP_ATR_MULTIPLE", 1.10)),
        trendfail_target_atr_multiple=float(os.getenv("TRENDFAIL_TARGET_ATR_MULTIPLE", 2.20)),
        atr_length=int(os.getenv("ATR_LENGTH", 14)),
        atr_floor_pct=float(os.getenv("ATR_FLOOR_PCT", 0.0035)),
    )

    if not base_url:
        raise ValueError("Missing APCA_API_BASE_URL in .env file.")
    exec_config = ExecutionConfig(
        paper=True if "paper" in base_url.lower() else False,
        base_url=base_url,
        conn_pool_maxsize=int(os.getenv("CONN_POOL_MAXSIZE", 32)),
        use_limit_entry=parse_bool(os.getenv("USE_LIMIT_ENTRY", "True"), default=True),
        limit_entry_offset_bps=float(os.getenv("LIMIT_ENTRY_OFFSET_BPS", 8.0)),
        tif=os.getenv("TRADE_TIF", "day"),
        # Opt-in: run the order path on the maintained alpaca-py SDK. Default off
        # = legacy alpaca_trade_api (unchanged). Paper-test before flipping on.
        use_alpaca_py=parse_bool(os.getenv("USE_ALPACA_PY", "False")),
        # Opt-in breakout exit (default off). Pair with a WIDE TARGET_ATR_MULTIPLE
        # so the bracket target is just a backstop the active channel exit beats.
        use_channel_exit=parse_bool(os.getenv("USE_CHANNEL_EXIT", "False")),
        channel_exit_threshold=float(os.getenv("CHANNEL_EXIT_THRESHOLD", 0.80)),
        channel_lookback=int(os.getenv("CHANNEL_LOOKBACK", 80)),
        channel_k=float(os.getenv("CHANNEL_K", 2.0)),
    )

    # 3. Initialize Executor and Symbol List
    executor = AlpacaExecutor(api_key, api_secret, exec_config)

    # Preflight: confirm the keys are authorized for this endpoint before doing
    # anything else. Bad creds (e.g. paper keys against the live URL) used to
    # surface as cryptic, repeated scan failures; now we abort cleanly with a
    # clear hint and release the lock so the supervisor doesn't hot-loop.
    if not preflight_check(executor, exec_config.paper):
        print("[PREFLIGHT] Aborting before any orders — fix the above and restart.")
        release_lock(LOCK_PATH)
        return 0

    # Attempt dynamic scan
    symbols = executor.scan_symbols(
        min_price=strategy_config.min_price,
        min_dollar_volume=strategy_config.min_dollar_volume,
        max_count=20
    )

    # Fallback watchlist = the names that cleared the walk-forward gate in the
    # last 90-day autotune (PF >= 1.10, Sharpe >= 0 out of sample). Keeping the
    # fallback in sync with the allowlist means a scanner whiff still trades only
    # vetted symbols. Re-run autotune_run.py to refresh both this list and
    # TRADE_ALLOWLIST.
    if not symbols:
        print("[WARN] Scanner returned no symbols. Using fallback watchlist.")
        symbols = [
            "AMD", "NVDA", "SMCI", "META", "ARM",
            "CRWD", "FCX", "OXY", "CLF", "HIMS",
        ]

    # 3a-bis. Force-include watchlist / force-exclude blocklist. These wrap the
    # scanner (and run before the allowlist gate) so curated names trade even if
    # the liquidity scan misses them, and chronic losers never trade even if the
    # scan keeps surfacing them. Both are optional comma-separated .env values.
    include = parse_symbol_csv(os.getenv("TRADE_WATCHLIST"))
    exclude = parse_symbol_csv(os.getenv("TRADE_BLOCKLIST"))
    if include or exclude:
        before = {str(s).upper() for s in symbols}
        symbols = apply_watchlist(symbols, include, exclude)
        added = [s for s in symbols if str(s).upper() not in before]
        if added:
            print(f"[WATCHLIST] Force-including {len(added)} name(s): {', '.join(added)}")
        if exclude:
            print(f"[BLOCKLIST] Excluding: {', '.join(sorted(exclude))}")

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
    perf_config = PerformanceConfig(
        state_dir=os.getenv("PERF_STATE_DIR", "state/performance"),
        enable_adaptive_threshold=parse_bool(os.getenv("PERF_ENABLE_ADAPTIVE_THRESHOLD", "True"), default=True),
        min_samples=int(os.getenv("PERF_MIN_SAMPLES", 20)),
        max_threshold_adj=float(os.getenv("PERF_MAX_THRESHOLD_ADJ", 0.05)),
    )
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
        direction_flip_cooldown_minutes=int(os.getenv("DIRECTION_FLIP_COOLDOWN_MINUTES", 60)),
    )
    
    # Per-symbol params: each allowlisted name trades with its OWN tuned config
    # (written by autotune_run.py), falling back to the global strategy_config
    # for anything without a stored entry. Missing file -> empty -> prior behavior.
    params_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.getenv("SYMBOL_PARAMS_FILE", "symbol_params.json"),
    )
    symbol_configs = build_symbol_configs(load_symbol_params(params_path), strategy_config)
    if symbol_configs:
        covered = [s for s in symbols if str(s).upper() in symbol_configs]
        print(f"[PARAMS] Per-symbol configs loaded: {len(symbol_configs)} "
              f"({len(covered)} of {len(symbols)} in today's universe use their own params).")

    service = ReversionService(
        strategy_config,
        risk_config,
        perf_config,
        symbol_configs=symbol_configs,
        min_trade_score=float(os.getenv("MIN_TRADE_SCORE", 0.36)),
    )
    portfolio_state = PortfolioState()
    governor = ExecutionGovernor(config=portfolio_config, portfolio_state=portfolio_state)

    print(f"--- REVERSION BOT STARTING ---")
    print(f"Timeframe: {timeframe} | Lookback: {lookback} | Mode: {'PAPER' if exec_config.paper else 'LIVE'}")
    print(f"Shorts: {'ENABLED' if strategy_config.enable_shorts else 'disabled'}")
    if use_market_filter:
        extra = " + favor shorts" if favor_shorts_risk_off else ""
        print(f"Market regime filter: ON ({regime_symbol} {regime_timeframe} EMA{regime_ema} — block longs when risk-off{extra})")
    elif favor_shorts_risk_off:
        print(f"Market regime: favor-shorts only ({regime_symbol} {regime_timeframe} EMA{regime_ema})")
    else:
        print("Market regime filter: off")
    if use_sector_filter:
        covered = sorted({sector_for(s) for s in symbols} - {None})
        mode = "sector OR SPY" if sector_combine != "sector" else "sector-only"
        print(f"Sector regime gate: ON ({mode}, {regime_timeframe} EMA{regime_ema}) — "
              f"sectors in play: {', '.join(covered) or '(none mapped)'}")
    if exec_config.use_channel_exit:
        print(f"Breakout exit: ON (exit longs at channel pos >= "
              f"{exec_config.channel_exit_threshold:g}, lookback {exec_config.channel_lookback} bars) "
              f"— set a WIDE TARGET_ATR_MULTIPLE as the backstop.")
    print(f"Monitoring: {', '.join(symbols)}")

    cycle = 0
    try:
        while True:
            market_open = await asyncio.to_thread(is_market_open, executor)
            cycle += 1

            # Liveness heartbeat: written every cycle (in all branches) so a
            # headless/logged-off bot can be health-checked without log access.
            try:
                write_heartbeat(str(HEARTBEAT_PATH), {
                    "cycle": cycle,
                    "market_open": bool(market_open),
                    "monitoring": len(symbols),
                    "mode": "PAPER" if exec_config.paper else "LIVE",
                })
            except Exception:
                pass

            if not market_open:
                # End-of-day: once the session is over, flatten anything still
                # open (covers half-days where the 14:50 CT EOD window never
                # fired) and exit cleanly (code 0) so the supervisor stops until
                # tomorrow's open trigger. Pre-open, just sleep until the bell.
                if await asyncio.to_thread(is_session_over, executor):
                    await liquidate_all_positions(executor)
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] [SESSION] Market closed for the day. Exiting cleanly.")
                    break
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Market closed (pre-open). Sleeping...")
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

            # Resolve the market regime once per cycle, up front, so the
            # short-bias relaxation can be applied during evaluation (not just
            # the long suppression afterwards).
            risk_off = False
            if use_market_filter or favor_shorts_risk_off:
                risk_off = await evaluate_market_regime(regime_symbol, regime_timeframe, regime_ema)
            short_bias = risk_off and favor_shorts_risk_off
            if risk_off:
                tag = "risk-off (favoring shorts)" if short_bias else "risk-off"
                print(f"[REGIME] {regime_symbol} {regime_timeframe} below EMA{regime_ema} — {tag}.")

            # Sector regime (cached, hourly): which of today's sectors are below
            # their proxy ETF's trend. Resolved up front so the long-suppression
            # step can judge each name against its own sector.
            sector_regime = {}
            if use_sector_filter:
                active_sectors = {sector_for(s) for s in symbols}
                active_sectors.discard(None)
                sector_regime = await evaluate_sector_regime(
                    active_sectors, regime_timeframe, regime_ema
                )
                off = sorted(s for s, ro in sector_regime.items() if ro)
                if off:
                    print(f"[REGIME] sector risk-off (EMA{regime_ema}): {', '.join(off)}")

            eval_tasks = [
                evaluate_symbol_only(s, lookback, timeframe, service, executor, short_bias)
                for s in symbols
            ]
            raw_results = await asyncio.gather(*eval_tasks)
            all_results = [r for r in raw_results if r is not None]
            candidates = [r for r in all_results if r.get("go_long") or r.get("go_short")]

            # Regime gate: drop new longs in a risk-off tape (keep shorts). The
            # sector-aware gate supersedes the SPY-only one and, under "or",
            # still folds SPY in as the broad overlay (and the fallback for any
            # unclassified name).
            dropped = []
            if use_sector_filter:
                candidates, dropped = suppress_longs_by_sector(
                    candidates, sector_regime,
                    fallback_risk_off=(risk_off if use_market_filter else False),
                    combine=sector_combine,
                )
            elif use_market_filter and risk_off:
                candidates, dropped = suppress_longs_if_risk_off(candidates, True)
            if dropped:
                names = ', '.join(c['symbol'] for c in dropped)
                print(f"[REGIME] suppressed {len(dropped)} long(s): {names}")

            # Keep equity tracking current for drawdown checks
            if all_results:
                try:
                    eq = await asyncio.to_thread(get_account_equity, executor)
                    portfolio_state.update_equity(eq)
                except Exception:
                    pass

            await execute_candidates(governor, executor, portfolio_state, candidates)

            # Breakout exit: take profit on any long that has run to the upper
            # channel, before its (wide backstop) bracket target. Opt-in.
            if exec_config.use_channel_exit:
                await manage_channel_exits(executor, exec_config, lookback, timeframe)

            await asyncio.sleep(poll_interval)

        # Clean end-of-day exit (loop broke after the session closed).
        return 0
    except KeyboardInterrupt:
        print("\n[STOP] Shutting down...")
        return 0
    finally:
        # On close, free the single-instance lock so the next launch can start.
        release_lock(LOCK_PATH)
        print("[LOCK] Released single-instance lock.")

if __name__ == "__main__":
    import sys
    # Exit code 0 = clean (session over / duplicate) -> supervisor stops.
    # A crash raises -> non-zero exit -> supervisor restarts with backoff.
    sys.exit(asyncio.run(main()) or 0)