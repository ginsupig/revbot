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
from reversion_bot.schedule import session_done, within_minutes_before_close
from reversion_bot.heartbeat import write_heartbeat
from reversion_bot.market_regime import (
    is_risk_off,
    suppress_longs_if_risk_off,
    suppress_longs_by_sector,
)
from reversion_bot.sectors import sector_for, etf_for_sector
from reversion_bot.universe import select_base_universe, broad_reversion_universe
from reversion_bot.swing import is_moc_entry_window, should_block_entry_now, holds_overnight
from reversion_bot.channel import select_breakout_exits, select_breakdown_exits
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
        return is_risk_off(
            bars, ema_length,
            buffer_pct=float(os.getenv("REGIME_BUFFER_PCT", 0.005)),
            confirm_bars=int(os.getenv("REGIME_CONFIRM_BARS", 3)),
        )
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

async def evaluate_symbol_only(symbol, lookback, timeframe, service, executor, short_bias=False, account_equity=None):
    try:
        if await asyncio.to_thread(executor.has_open_position, symbol):
            return None

        bars = await fetch_bars_for_symbol(symbol, timeframe, lookback)
        if bars is None or len(bars) < lookback:
            return None

        bars = bars.tail(lookback)
        if account_equity is None:
            account_equity = await asyncio.to_thread(get_account_equity, executor)
        result = await asyncio.to_thread(service.evaluate_symbol, symbol, bars, account_equity, short_bias)
        result["_account_equity"] = account_equity
        return result
    except Exception as e:
        import traceback
        print(f"[ERROR] {symbol}: {e}\n{traceback.format_exc()}")
        return None

async def execute_candidates(governor, executor, portfolio_state, candidates):
    if not candidates:
        print("[INFO] No candidates to execute.")
        return

    trades_this_cycle = 0
    for candidate in candidates:
        # Isolate each candidate: a broker rejection on one (e.g. a short blocked
        # by shorting_enabled, a wash-trade error, or a transient 4xx) must NOT
        # abort the remaining approved candidates or skip their portfolio_state
        # update. Previously one exception unwound the whole loop.
        symbol = candidate.get("symbol", "?")
        try:
            if governor.approve(candidate, executor=executor, trades_executed_this_cycle=trades_this_cycle):
                print(f"[INFO] Executing trade for {symbol}.")
                await asyncio.to_thread(executor.submit_order, candidate)
                portfolio_state.update(candidate)
                trades_this_cycle += 1
            else:
                print(f"[INFO] Candidate {symbol} not approved by governor.")
        except Exception as e:
            print(f"[ERROR] Failed to execute candidate {symbol}: {e}")

# --- EOD Liquidation ---

def _get_close_ct(executor) -> datetime:
    """Get today's close time from the broker clock (handles half-days).

    Falls back to the standard 15:00 CT if the API call fails.
    """
    ct = ZoneInfo("America/Chicago")
    now_ct = datetime.now(ct)
    try:
        clock = executor.client.get_clock()
        return clock.next_close.astimezone(ct)
    except Exception:
        return now_ct.replace(hour=15, minute=0, second=0, microsecond=0)


def is_eod_liquidation_window(executor) -> bool:
    """True during the final stretch before the close.

    Uses the broker clock so it fires correctly on half-day early closes
    (e.g. day before July 4th, day after Thanksgiving).
    """
    lead_minutes = int(os.getenv("EOD_LIQUIDATION_MINUTES", 10))
    now_ct = datetime.now(ZoneInfo("America/Chicago"))
    close_ct = _get_close_ct(executor)
    start_ct = close_ct - timedelta(minutes=lead_minutes)
    return start_ct <= now_ct < close_ct


def is_eod_entry_cutoff(executor) -> bool:
    """True in the final stretch before the close where NEW entries are blocked.

    Uses the broker clock so it fires correctly on half-day early closes.
    Set EOD_ENTRY_CUTOFF_MINUTES=0 to disable.
    """
    minutes = int(os.getenv("EOD_ENTRY_CUTOFF_MINUTES", 20))
    if minutes <= 0:
        return False
    now_ct = datetime.now(ZoneInfo("America/Chicago"))
    close_ct = _get_close_ct(executor)
    start_ct = close_ct - timedelta(minutes=minutes)
    return start_ct <= now_ct < close_ct


# Once-per-day stamp so the carryover flatten runs on the day's FIRST open cycle
# but a same-day crash-restart does NOT flatten positions opened earlier today.
_RECONCILE_STAMP = Path(__file__).resolve().parent / "state" / "last_reconcile.date"


def _today_ct() -> str:
    return datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d")


def _already_reconciled_today() -> bool:
    try:
        return _RECONCILE_STAMP.read_text().strip() == _today_ct()
    except Exception:
        return False


def _mark_reconciled_today() -> None:
    try:
        _RECONCILE_STAMP.parent.mkdir(parents=True, exist_ok=True)
        _RECONCILE_STAMP.write_text(_today_ct())
    except Exception:
        pass


async def reconcile_carryover(executor) -> None:
    """Flatten orphaned OVERNIGHT carryover on the day's first market-open cycle.

    The bot is flat-overnight by design, so a position open at the first run of a
    new day is an orphan from a prior session whose EOD flatten didn't complete
    (seen live: NVDA/SMCI). Date-stamped so a same-day crash-restart does NOT
    flatten positions the bot legitimately opened today. Opt out with
    FLATTEN_CARRYOVER_ON_START=false (warns and leaves them instead).
    """
    if _already_reconciled_today():
        return
    try:
        positions = await asyncio.to_thread(executor.get_positions)
    except Exception as e:
        print(f"[STARTUP] carryover check skipped (position fetch failed: {e}).")
        return  # don't stamp; retry next cycle
    if not positions:
        _mark_reconciled_today()
        return
    names = ", ".join(f"{getattr(p, 'symbol', '?')}x{getattr(p, 'qty', '?')}" for p in positions)
    if not parse_bool(os.getenv("FLATTEN_CARRYOVER_ON_START", "True"), default=True):
        print(f"[STARTUP] WARNING: {len(positions)} overnight carryover position(s) open "
              f"and FLATTEN_CARRYOVER_ON_START=false — leaving them: {names}")
        _mark_reconciled_today()
        return
    print(f"[STARTUP] Flattening {len(positions)} overnight carryover position(s) "
          f"before trading: {names}")
    await liquidate_all_positions(executor)
    _mark_reconciled_today()


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
        if signed_qty == 0:
            continue
        # The global cancel_all_orders above frees each position's bracket legs,
        # but that qty releases server-side ASYNCHRONOUSLY: an immediate market
        # order hits "insufficient qty available (available: 0)" and — with no
        # retry — the position is left open OVERNIGHT (the live 7/3 flatten
        # failure). Also cancel the symbol's own working orders and retry the
        # broker-atomic close a few times, mirroring execution.close_long, so the
        # flatten actually completes before the bell.
        try:
            executor._cancel_orders_for_symbol(symbol)
        except Exception as e:
            print(f"[EOD] Per-symbol cancel failed for {symbol}: {e}")
        last_exc = None
        for _ in range(4):
            try:
                await asyncio.to_thread(executor.client.close_position, symbol)
                print(f"[EOD] Close submitted: {symbol} (was {signed_qty:+d})")
                last_exc = None
                break
            except Exception as e:
                last_exc = e
                await asyncio.sleep(0.5)
        if last_exc is not None:
            print(f"[EOD] Failed to close {symbol} after retries: {last_exc}")


async def manage_channel_exits(executor, exec_config, lookback, timeframe):
    """Channel exit: market-close positions that have reached the regression
    channel extremes — longs at the upper band, shorts at the lower band.
    The broker-side stop + bracket target remain the safety net. Best-effort.
    """
    try:
        positions = await asyncio.to_thread(executor.get_positions)
    except Exception as e:
        print(f"[CHANNEL-EXIT] position fetch failed ({e}); skipping.")
        return

    longs = [p for p in positions if _safe_qty(p) > 0]
    shorts = [p for p in positions if _safe_qty(p) < 0]
    if not longs and not shorts:
        return

    look = int(getattr(exec_config, "channel_lookback", 80))
    k = float(getattr(exec_config, "channel_k", 2.0))
    threshold = float(getattr(exec_config, "channel_exit_threshold", 0.80))
    # Short exit threshold: mirror at the lower band (1.0 - threshold).
    short_threshold = 1.0 - threshold

    # Channel was validated on daily bars — use daily regardless of the bot's
    # intraday timeframe, so an 80-bar lookback covers 80 trading days (not
    # ~6.5 hours of 5-min bars which would be far too noisy).
    channel_tf = os.getenv("CHANNEL_EXIT_TIMEFRAME", "1Day")

    # Fetch recent closes for all held positions.
    all_held = longs + shorts
    closes_by_symbol = {}
    for p in all_held:
        symbol = str(p.symbol).upper()
        if symbol in closes_by_symbol:
            continue
        try:
            bars = await fetch_bars_for_symbol(symbol, channel_tf, max(lookback, look))
            if bars is not None and len(bars) >= look:
                closes_by_symbol[symbol] = bars["close"].to_numpy()
        except Exception as e:
            print(f"[CHANNEL-EXIT] {symbol} bar fetch failed ({e}); skipping.")

    # Long breakout exits (upper channel)
    for symbol in select_breakout_exits(longs, closes_by_symbol, threshold, look, k):
        try:
            await asyncio.to_thread(executor.close_long, symbol)
            print(f"[CHANNEL-EXIT] {symbol} reached the upper channel — exiting long.")
        except Exception as e:
            print(f"[CHANNEL-EXIT] {symbol} long close failed: {e}")

    # Short breakdown exits (lower channel)
    for symbol in select_breakdown_exits(shorts, closes_by_symbol, short_threshold, look, k):
        try:
            await asyncio.to_thread(executor.close_short, symbol)
            print(f"[CHANNEL-EXIT] {symbol} reached the lower channel — covering short.")
        except Exception as e:
            print(f"[CHANNEL-EXIT] {symbol} short cover failed: {e}")


async def manage_trailing_stops(executor, portfolio_state, risk_config):
    """Ratchet each open position's stop leg toward the trailing stop.

    Longs: high-water ratchets UP, stop raised toward high_water - trail*ATR.
    Shorts: low-water ratchets DOWN, stop lowered toward low_water + trail*ATR.
    Best-effort and per-symbol — a hiccup on one name can't block the loop, and
    the fixed bracket stop remains the backstop.
    """
    try:
        positions = await asyncio.to_thread(executor.get_positions)
    except Exception as e:
        print(f"[TRAIL] position fetch failed ({e}); skipping.")
        return
    for p in positions:
        qty = _safe_qty(p)
        if qty == 0:
            continue
        symbol = str(getattr(p, "symbol", "")).upper()
        try:
            last_price = float(getattr(p, "current_price", 0) or 0)
        except (TypeError, ValueError):
            continue
        if last_price <= 0:
            continue

        if qty > 0:
            # Long trailing stop: ratchet UP
            state = portfolio_state.get_trail_state(symbol)
            if state is None:
                continue
            entry_price, stop_distance, _ = state
            hw = portfolio_state.update_high_water(symbol, last_price)
            if hw is None:
                continue
            new_stop = await asyncio.to_thread(
                executor.update_trailing_stop, symbol, entry_price, stop_distance, hw,
                last_price, risk_config.stop_atr_multiple, risk_config.trail_atr_multiple,
            )
            if new_stop is not None:
                print(f"[TRAIL] {symbol} long stop -> {new_stop:.2f} (high-water {hw:.2f})")
        else:
            # Short trailing stop: ratchet DOWN
            state = portfolio_state.get_trail_state_short(symbol)
            if state is None:
                continue
            entry_price, stop_distance, _ = state
            lw = portfolio_state.update_low_water(symbol, last_price)
            if lw is None:
                continue
            new_stop = await asyncio.to_thread(
                executor.update_trailing_stop_short, symbol, entry_price, stop_distance,
                lw, last_price, risk_config.stop_atr_multiple, risk_config.trail_atr_multiple,
            )
            if new_stop is not None:
                print(f"[TRAIL] {symbol} short stop -> {new_stop:.2f} (low-water {lw:.2f})")


def _safe_qty(position) -> float:
    try:
        return float(getattr(position, "qty", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


# Activity fetches use the legacy REST client (get_activities), independent of
# the order path: the opt-in alpaca-py order client doesn't expose activities.
# Built once and cached — reconcile runs at most twice a session.
_activity_client = None


def _get_activity_client(api_key, api_secret, base_url):
    global _activity_client
    if _activity_client is None:
        from alpaca_trade_api.rest import REST
        _activity_client = REST(api_key, api_secret, base_url)
    return _activity_client


def _attr(obj, name, default=None):
    """Read a field whether the activity is an SDK object or a raw dict."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _fetch_recent_fills(client, days: int):
    """Normalized FILL activities over the last ``days`` (paginated, ascending).

    Same normalized shape pnl_report uses (symbol/side/qty/price/time) so the
    pure FIFO helpers consume it directly.
    """
    after = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    fills = []
    page_token = None
    page_size = 100
    while True:
        batch = client.get_activities(
            activity_types="FILL", after=after, direction="asc",
            page_size=page_size, page_token=page_token,
        )
        if not batch:
            break
        for a in batch:
            qty, price = _attr(a, "qty"), _attr(a, "price")
            symbol, side = _attr(a, "symbol"), _attr(a, "side")
            if qty is None or price is None or not symbol or not side:
                continue
            fills.append({
                "symbol": str(symbol), "side": str(side).lower(),
                "qty": abs(float(qty)), "price": float(price),
                "time": str(_attr(a, "transaction_time", "")),
            })
        if len(batch) < page_size:
            break
        page_token = _attr(batch[-1], "id")
        if not page_token:
            break
    return fills


async def reconcile_trade_outcomes(service, api_key, api_secret, base_url, days) -> None:
    """Book realized PnL of CLOSED round-trips into outcomes so the adaptive
    threshold can learn from results (not from its own entry-time opinion).

    Best-effort: any failure (missing creds, SDK, network) is logged and skipped
    so it never disrupts the trading loop. Idempotent via the tracker's cursor,
    so calling it at startup and at EOD never double-counts a trade.
    """
    try:
        client = await asyncio.to_thread(_get_activity_client, api_key, api_secret, base_url)
        fills = await asyncio.to_thread(_fetch_recent_fills, client, days)
        n = await asyncio.to_thread(service.reconcile_outcomes, fills)
        if n:
            print(f"[OUTCOMES] Booked {n} realized trade outcome(s) for threshold adaptation.")
    except Exception as e:
        print(f"[OUTCOMES] Reconcile skipped ({e}).")


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
    # Trailing stop on open longs (default ON): each cycle, ratchet the bracket's
    # stop leg up toward high_water - TRAIL_ATR_MULTIPLE*ATR. The validated edge
    # (execution_tuning_backtest.py). Set USE_TRAILING_STOP=False to disable.
    use_trailing_stop = parse_bool(os.getenv("USE_TRAILING_STOP", "True"), default=True)
    # Swing / overnight-hold mode (opt-in, default OFF = today's intraday flat-by-bell
    # behavior). The validated daily broad-reversion edge enters at the CLOSE
    # (market-on-close) and holds overnight/multi-day, the ATR trail riding winners.
    # When ON: entries only in the MOC submission window, and EOD/session/carryover
    # flattens are skipped so positions hold. Pair with TIF=cls, TRADE_TIMEFRAME=1Day,
    # USE_BROAD_REVERSION_UNIVERSE=true. See reversion_bot/swing.py.
    swing_mode = parse_bool(os.getenv("SWING_OVERNIGHT_MODE", "False"), default=False)
    moc_window_min = int(os.getenv("MOC_ENTRY_WINDOW_MINUTES", 20))
    # Persistent daemon mode (opt-in, default OFF = today's exit-at-close behavior).
    # When ON, the bot does NOT exit at the close — it sleeps until the next session
    # and keeps running, so you launch it ONCE and it's always alive for the MOC
    # window (no scheduler/relaunch needed). Recommended with SWING_OVERNIGHT_MODE.
    run_persistent = parse_bool(os.getenv("RUN_PERSISTENT", "False"), default=False)
    persist_sleep_s = int(os.getenv("PERSIST_CLOSED_SLEEP_SECONDS", 600))
    regime_symbol = os.getenv("MARKET_REGIME_SYMBOL", "SPY")
    regime_timeframe = os.getenv("MARKET_REGIME_TIMEFRAME", "1Day")
    regime_ema = int(os.getenv("MARKET_REGIME_EMA", 50))
    # Favor-shorts mode (opt-in): when risk-off, also relax the short trigger so
    # the bot leans short into the downtrend instead of only sitting in cash.
    # Lean short in risk-off tapes when shorts are enabled. Off by default
    # (shorts are off); turn on with ENABLE_SHORTS + this flag.
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
        min_price=float(os.getenv("MIN_PRICE", 2.0)),
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
        rsi_max=float(os.getenv("RSI_MAX", 40.0)),
        oversold_gate=os.getenv("OVERSOLD_GATE", "and"),
        adx_max=float(os.getenv("ADX_MAX", 40.0)),
        adx_hard_max=float(os.getenv("ADX_HARD_MAX", 50.0)),
        rsi_hard_max=float(os.getenv("RSI_HARD_MAX", 70.0)),
        max_vwap_extension_pct=float(os.getenv("MAX_VWAP_EXTENSION_PCT", 0.012)),
        require_reclaim_lb1=parse_bool(os.getenv("REQUIRE_RECLAIM_LB1", "False")),
        require_bullish_close=parse_bool(os.getenv("REQUIRE_BULLISH_CLOSE", "False")),
        require_volume_expansion=parse_bool(os.getenv("REQUIRE_VOLUME_EXPANSION", "False")),
        use_vwap_filter=parse_bool(os.getenv("USE_VWAP_FILTER", "False")),
        use_trend_filter=parse_bool(os.getenv("USE_TREND_FILTER", "True")),
        trend_filter_band_pct=float(os.getenv("TREND_FILTER_BAND_PCT", 0.02)),
        loss_reentry_cooldown_minutes=int(os.getenv("LOSS_REENTRY_COOLDOWN_MINUTES", 90)),
        # Volume confirmation.
        volume_lookback=int(os.getenv("VOLUME_LOOKBACK", 20)),
        volume_multiplier_min=float(os.getenv("VOLUME_MULTIPLIER_MIN", 1.0)),
        # Trend-fail strategy.
        trendfail_window=int(os.getenv("TRENDFAIL_WINDOW", 20)),
        trendfail_threshold=float(os.getenv("TRENDFAIL_THRESHOLD", 0.005)),
        # Short side (mirror) gates. Off by default: OOS backtest showed shorts
        # bleeding on daily large-caps (8% WR). Enable with ENABLE_SHORTS=True
        # paired with USE_MARKET_REGIME_FILTER + FAVOR_SHORTS_IN_RISK_OFF to
        # only short in confirmed downtrends.
        enable_shorts=parse_bool(os.getenv("ENABLE_SHORTS", "False"), default=False),
        ri_short_threshold=float(os.getenv("RI_SHORT_THRESHOLD", 0.5)),
        rsi_min=float(os.getenv("RSI_MIN", 65.0)),
        rsi_hard_min=float(os.getenv("RSI_HARD_MIN", 30.0)),
        risk_off_rsi_min=float(os.getenv("RISK_OFF_RSI_MIN", 55.0)),
        risk_off_ri_short_threshold=float(os.getenv("RISK_OFF_RI_SHORT_THRESHOLD", 0.30)),
    )

    risk_config = RiskConfig(
        risk_per_trade_pct=float(os.getenv("RISK_PER_TRADE_PCT", 0.005)),
        max_position_value_pct=float(os.getenv("MAX_POSITION_VALUE_PCT", 0.15)),
        min_rr=float(os.getenv("MIN_RR", 1.5)),
        # Per-style ATR stop/target multiples.
        stop_atr_multiple=float(os.getenv("STOP_ATR_MULTIPLE", 1.20)),
        target_atr_multiple=float(os.getenv("TARGET_ATR_MULTIPLE", 3.50)),
        trail_atr_multiple=float(os.getenv("TRAIL_ATR_MULTIPLE", 1.50)),
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
    scanned_symbols = list(symbols)
    fallback_watchlist = [
        "AMD", "NVDA", "SMCI", "META", "ARM",
        "CRWD", "FCX", "OXY", "CLF", "HIMS",
    ]
    # USE_BROAD_REVERSION_UNIVERSE: trade the validated neutral large-cap basket
    # (research/portfolio_sim.py — two-window OOS under live ATR-risk sizing,
    # ~+20%/window at <7% MTM drawdown at the 4-cap) instead of the scanner/curated
    # watchlist. Default OFF = today's behavior. Occupancy cap, ATR sizing and the
    # trailing stop stay as-is; this only swaps the universe the engine scans.
    use_broad_universe = parse_bool(os.getenv("USE_BROAD_REVERSION_UNIVERSE", "False"), default=False)
    symbols = select_base_universe(scanned_symbols, fallback_watchlist, use_broad_universe)
    if use_broad_universe:
        print(f"[BROAD] USE_BROAD_REVERSION_UNIVERSE on — trading the neutral large-cap "
              f"reversion universe ({len(symbols)} names); scanner/fallback overridden.")
    elif not scanned_symbols:
        print("[WARN] Scanner returned no symbols. Using fallback watchlist.")

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
    # The per-symbol allowlist is the curated path's vetting gate; in broad-reversion
    # mode the neutral universe IS the filter, so applying it would collapse the
    # basket to a handful of names. Skip it (with a note) when broad mode is on.
    allowlist = parse_allowlist(os.getenv("TRADE_ALLOWLIST"))
    if allowlist is not None and use_broad_universe:
        print("[BROAD] TRADE_ALLOWLIST ignored in broad-reversion mode (the universe is the filter).")
    elif allowlist is not None:
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
        # Persist the fitted ML model across restarts (no daily cold-start).
        persist_ml_model=parse_bool(os.getenv("PERSIST_ML_MODEL", "True"), default=True),
        ml_model_max_age_hours=float(os.getenv("ML_MODEL_MAX_AGE_HOURS", 168.0)),
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
        direction_flip_cooldown_minutes=int(os.getenv("DIRECTION_FLIP_COOLDOWN_MINUTES", 60)),  # whipsaw guard
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
        min_trade_score=float(os.getenv("MIN_TRADE_SCORE", 0.45)),
    )
    portfolio_state = PortfolioState()
    governor = ExecutionGovernor(config=portfolio_config, portfolio_state=portfolio_state)

    print(f"--- REVERSION BOT STARTING ---")
    print(f"Timeframe: {timeframe} | Lookback: {lookback} | Mode: {'PAPER' if exec_config.paper else 'LIVE'}")
    print(f"Shorts: {'ENABLED' if strategy_config.enable_shorts else 'disabled'}")
    if swing_mode:
        print(f"SWING/OVERNIGHT mode: ON — enter in the MOC window (last {moc_window_min}m before "
              f"2:50pm CT), hold overnight/multi-day, no EOD/carryover flatten (tif={exec_config.tif}).")
    if use_trailing_stop:
        print(f"Trailing stop: ON (stop ratchets to high_water - {risk_config.trail_atr_multiple:g}xATR; "
              f"target {risk_config.target_atr_multiple:g}xATR backstop)")
    else:
        print("Trailing stop: off")
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

    # Book outcomes from any closes since the last run (a prior session, or a
    # crash mid-day) before trading, so today's threshold adaptation starts from
    # the freshest realized evidence.
    outcome_reconcile_days = int(os.getenv("OUTCOME_RECONCILE_DAYS", "7"))
    await reconcile_trade_outcomes(service, api_key, api_secret, base_url, outcome_reconcile_days)

    cycle = 0
    last_booked_date = None      # persistent mode: book outcomes once per closed session
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
                # End-of-day: once the session is over, flatten anything still open
                # (intraday mode) and book the session's realized outcomes. In
                # non-persistent mode the bot then exits cleanly (a scheduler/
                # supervisor relaunches it next session); in RUN_PERSISTENT mode it
                # sleeps and stays alive so it's always up for the next MOC window.
                # Pre-open, just sleep until the bell.
                session_over = await asyncio.to_thread(is_session_over, executor)
                if session_over:
                    # Swing mode holds overnight (the ATR trail manages positions);
                    # only flatten at session end in intraday mode.
                    if not holds_overnight(swing_mode):
                        await liquidate_all_positions(executor)
                    today_ct = _today_ct()
                    if today_ct != last_booked_date:        # book once per closed session
                        await reconcile_trade_outcomes(service, api_key, api_secret, base_url, outcome_reconcile_days)
                        last_booked_date = today_ct
                    held = " (holding overnight)" if holds_overnight(swing_mode) else ""
                    if not run_persistent:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] [SESSION] Market closed for the day{held}. Exiting cleanly.")
                        break
                # Persistent (or pre-open): stay alive. Sleep longer while closed.
                phase = "post-close" if session_over else "pre-open"
                nap = persist_sleep_s if (session_over and run_persistent) else poll_interval
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Market closed ({phase}). Sleeping {nap}s...")
                await asyncio.sleep(nap)
                continue

            if is_eod_liquidation_window(executor) and not holds_overnight(swing_mode):
                await liquidate_all_positions(executor)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] EOD liquidation done. Sleeping...")
                await asyncio.sleep(poll_interval)
                continue

            # Day's first open cycle: flatten any orphaned overnight carryover
            # before trading (no-op after the first run each day). Skipped in swing
            # mode — there, overnight positions are intended holds, not orphans.
            if not holds_overnight(swing_mode):
                await reconcile_carryover(executor)

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

            # Fetch equity once per cycle — not per symbol (saves ~20 API calls).
            try:
                cycle_equity = await asyncio.to_thread(get_account_equity, executor)
            except Exception:
                cycle_equity = None

            eval_tasks = [
                evaluate_symbol_only(s, lookback, timeframe, service, executor, short_bias, account_equity=cycle_equity)
                for s in symbols
            ]
            raw_results = await asyncio.gather(*eval_tasks)
            all_results = [r for r in raw_results if r is not None]
            candidates = [r for r in all_results if r.get("go_long") or r.get("go_short")]

            # Entry timing. Swing mode INVERTS the intraday cutoff: it trades once a
            # day at the close, so entries are taken ONLY inside the market-on-close
            # window and blocked the rest of the day. Intraday mode keeps the EOD
            # cutoff (no entries that can't be flattened before the bell).
            if candidates:
                if swing_mode:
                    in_moc = is_moc_entry_window(
                        datetime.now(ZoneInfo("America/Chicago")), moc_window_min)
                    if should_block_entry_now(swing_mode, in_moc):
                        print(f"[SWING] Outside the MOC entry window — holding {len(candidates)} "
                              f"candidate(s) for the close.")
                        candidates = []
                elif is_eod_entry_cutoff(executor):
                    print(f"[EOD] Entry cutoff — blocking {len(candidates)} new entr(ies) near the close.")
                    candidates = []

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

            # Prune stale position_meta for symbols no longer held
            try:
                positions = await asyncio.to_thread(executor.client.list_positions)
                open_syms = [str(p.symbol).upper() for p in positions]
                portfolio_state.prune_closed_positions(open_syms)
            except Exception:
                pass

            await execute_candidates(governor, executor, portfolio_state, candidates)

            # Trailing stop: ratchet each open long's stop up toward the trailing
            # level (the validated edge). On by default.
            if use_trailing_stop:
                await manage_trailing_stops(executor, portfolio_state, risk_config)

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