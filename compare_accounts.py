"""Side-by-side scoreboard for two Alpaca accounts — the A/B verdict tool.

We run two bots on two separate paper accounts (original intraday vs the swing
overnight book) over the same calendar window. This pulls each account's fills
and diffs the numbers that actually settle the bet: realized PnL, win-rate,
profit factor, turnover, and the drawdown of the realized-PnL curve — then calls
a winner on PnL, risk-adjusted PnL (PnL / max realized drawdown), and efficiency
(return on turnover).

Each account is a *different* keypair, so we can't lean on the single-account
env the report scripts assume. Instead you point at each account's ``.env`` file:

    python compare_accounts.py \
        --account "Old=C:/revbot/.env" \
        --account "Swing=C:/revbot_swing/.env" \
        --days 7

Falls back to the two default folders if no --account is given:
    Old   -> ./.env
    Swing -> ./.env.swing   (override with --account as above)

Read-only: it only calls get_account()/get_activities(). It never places or
cancels an order, and it does not touch either live bot's state. Reuses the same
FIFO realized-PnL engine as pnl_report so the numbers reconcile exactly.

Discipline caveat this prints too: one week is not out-of-sample and the trade
count is tiny — treat the verdict as a leaderboard, not a validated edge.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional, Sequence, Tuple

from reversion_bot.report_client import (
    PAPER_URL,
    _auth_failure_message,
    endpoint_mode,
)
from reversion_bot.trade_report import (
    build_daily_timeline,
    build_trades,
    traded_notional,
)


# -----------------------------------------------------------------------------
# env parsing (do NOT exec the file — just read KEY=VALUE)
# -----------------------------------------------------------------------------
def parse_env_file(path: str) -> dict:
    """Read a dotenv-style file into a dict without executing it.

    Handles ``KEY=VALUE``, ``export KEY=VALUE``, surrounding quotes, inline
    ``#`` comments (only when unquoted), and skips blanks/comment lines. Missing
    file -> empty dict (the caller reports the clearer error).
    """
    out: dict = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError:
        return out
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if val and val[0] in "\"'":
            quote = val[0]
            end = val.find(quote, 1)
            if end != -1:
                val = val[1:end]                  # content between the quotes only
            else:
                val = val[1:]                     # unterminated quote — take the rest
        else:
            val = val.split(" #", 1)[0].strip()   # drop an unquoted inline comment
        if key:
            out[key] = val
    return out


def parse_account_arg(spec: str) -> Tuple[str, str]:
    """Split a ``LABEL=path`` (or ``LABEL:path``) --account spec into its parts."""
    for sep in ("=", ":"):
        # ':' only counts as the label separator if it isn't the drive colon of
        # a Windows path like C:/... — require the label to have no path chars.
        if sep in spec:
            label, rest = spec.split(sep, 1)
            if label and "/" not in label and "\\" not in label and "." not in label:
                return label.strip(), rest.strip()
    raise argparse.ArgumentTypeError(
        f"--account must look like LABEL=path/to/.env (got {spec!r})"
    )


# -----------------------------------------------------------------------------
# client build (parameterized — the report scripts only do one account)
# -----------------------------------------------------------------------------
def build_client_for(creds: dict, rest_cls=None):
    """Build + preflight an Alpaca REST client from an explicit creds dict.

    Mirrors report_client.make_client but takes creds directly (so two accounts
    can be opened in one process). Returns ``(client, account)``. ``rest_cls`` is
    injectable for tests. Raises SystemExit on missing/bad creds.
    """
    key = creds.get("APCA_API_KEY_ID")
    secret = creds.get("APCA_API_SECRET_KEY")
    base_url = creds.get("APCA_API_BASE_URL", PAPER_URL) or PAPER_URL
    if not key or not secret:
        raise SystemExit(
            "Missing Alpaca creds in the account's .env — need APCA_API_KEY_ID "
            "and APCA_API_SECRET_KEY."
        )
    mode = endpoint_mode(base_url)
    if rest_cls is None:
        from alpaca_trade_api.rest import REST as rest_cls
    client = rest_cls(key, secret, base_url)
    try:
        account = client.get_account()
    except Exception as e:
        print(_auth_failure_message(mode, base_url, e))
        raise SystemExit(1)
    return client, account


def _fetch_fills(client, days: int) -> List[dict]:
    # Import lazily so this module is usable (and testable) without pnl_report's
    # own import side effects; it's the same paginating FILL fetch.
    from pnl_report import fetch_fills
    return fetch_fills(client, days)


# -----------------------------------------------------------------------------
# metrics (pure — network-free, unit-testable)
# -----------------------------------------------------------------------------
def max_realized_drawdown(daily: Sequence[Tuple[str, float]]) -> float:
    """Deepest peak-to-trough dip (in $) of the CUMULATIVE realized-PnL curve.

    ``daily`` is ascending ``(date, realized_pnl)``. The curve starts at 0, so a
    book that only ever gains has drawdown 0. This is the risk axis the raw PnL
    total hides: two books can end at the same PnL, but the one that first sank
    -$4k to get there is the riskier bet.
    """
    cum = 0.0
    peak = 0.0
    maxdd = 0.0
    for _, pnl in daily:
        cum += pnl
        peak = max(peak, cum)
        maxdd = max(maxdd, peak - cum)
    return maxdd


def account_metrics(label: str, fills: List[dict], equity: Optional[float] = None) -> dict:
    """Compute the comparison metrics for one account from its fills.

    Reuses build_trades (FIFO realized PnL, win-rate, profit factor) and
    build_daily_timeline (per-day realized curve) so everything reconciles with
    pnl_report to the cent.
    """
    trips = build_trades(fills)                       # ALL symbols, FIFO
    turnover = sum(traded_notional(fills).values())
    timeline = build_daily_timeline(fills, newest_first=False)
    daily = [(r["date"], r["realized_pnl"]) for r in timeline["rows"]]   # ascending
    maxdd = max_realized_drawdown(daily)
    realized = trips["total_pnl"]
    pf = trips["profit_factor"]
    return {
        "label": label,
        "equity": equity,
        "fills": len(fills),
        "trading_days": timeline["days_in_window"],
        "round_trips": trips["round_trips"],
        "wins": trips["wins"],
        "losses": trips["losses"],
        "win_rate": trips["win_rate"],
        "profit_factor": pf,
        "avg_win": trips["avg_win"],
        "avg_loss": trips["avg_loss"],
        "realized_pnl": realized,
        "turnover": turnover,
        "return_on_turnover": (realized / turnover) if turnover else 0.0,
        "max_drawdown": maxdd,
        # risk-adjusted: PnL earned per $ of worst realized dip. inf when the book
        # never drew down (pure-gain curve) and made money; 0 if flat/never up.
        "return_on_drawdown": (
            (realized / maxdd) if maxdd > 0 else (float("inf") if realized > 0 else 0.0)
        ),
    }


def _pf_str(pf: float) -> str:
    return "inf" if pf == float("inf") else f"{pf:.2f}"


def _rod_str(rod: float) -> str:
    return "inf" if rod == float("inf") else f"{rod:.2f}"


def pick_winner(a: dict, b: dict, key: str, higher_is_better: bool = True) -> Optional[str]:
    """Label of the account that wins on ``key`` (None on a tie)."""
    va, vb = a[key], b[key]
    if va == vb:
        return None
    better = va > vb if higher_is_better else va < vb
    return a["label"] if better else b["label"]


def build_verdict(a: dict, b: dict) -> dict:
    """Winner on each axis + the headline call (realized PnL)."""
    return {
        "realized_pnl": pick_winner(a, b, "realized_pnl"),
        "return_on_drawdown": pick_winner(a, b, "return_on_drawdown"),
        "return_on_turnover": pick_winner(a, b, "return_on_turnover"),
        "win_rate": pick_winner(a, b, "win_rate"),
        "max_drawdown": pick_winner(a, b, "max_drawdown", higher_is_better=False),
        "headline": pick_winner(a, b, "realized_pnl"),
    }


# -----------------------------------------------------------------------------
# formatting
# -----------------------------------------------------------------------------
def _row(label: str, a_val: str, b_val: str, win: Optional[str],
         a_label: str, b_label: str) -> str:
    mark_a = " <" if win == a_label else ""
    mark_b = " <" if win == b_label else ""
    return f"  {label:<22}{a_val + mark_a:>20}{b_val + mark_b:>22}"


def format_comparison(a: dict, b: dict, days: int, same_account: bool = False) -> str:
    v = build_verdict(a, b)
    al, bl = a["label"], b["label"]

    def money(x):
        return f"${x:,.2f}"

    lines = [
        f"Account A/B — last {days} days",
        "=" * 64,
        f"  {'METRIC':<22}{al:>20}{bl:>22}",
        f"  {'-'*20:<22}{'-'*18:>20}{'-'*20:>22}",
        _row("equity now",
             money(a["equity"]) if a["equity"] is not None else "-",
             money(b["equity"]) if b["equity"] is not None else "-",
             None, al, bl),
        _row("realized PnL", money(a["realized_pnl"]), money(b["realized_pnl"]),
             v["realized_pnl"], al, bl),
        _row("round-trips",
             f"{a['round_trips']} ({a['wins']}-{a['losses']})",
             f"{b['round_trips']} ({b['wins']}-{b['losses']})",
             None, al, bl),
        _row("win-rate", f"{a['win_rate']*100:.0f}%", f"{b['win_rate']*100:.0f}%",
             v["win_rate"], al, bl),
        _row("profit factor", _pf_str(a["profit_factor"]), _pf_str(b["profit_factor"]),
             None, al, bl),
        _row("avg win / avg loss",
             f"{money(a['avg_win'])}/{money(a['avg_loss'])}",
             f"{money(b['avg_win'])}/{money(b['avg_loss'])}",
             None, al, bl),
        _row("turnover", money(a["turnover"]), money(b["turnover"]), None, al, bl),
        _row("return on turnover",
             f"{a['return_on_turnover']*100:.2f}%",
             f"{b['return_on_turnover']*100:.2f}%",
             v["return_on_turnover"], al, bl),
        _row("max realized DD", money(a["max_drawdown"]), money(b["max_drawdown"]),
             v["max_drawdown"], al, bl),
        _row("PnL / max DD", _rod_str(a["return_on_drawdown"]),
             _rod_str(b["return_on_drawdown"]), v["return_on_drawdown"], al, bl),
        _row("trading days", str(a["trading_days"]), str(b["trading_days"]),
             None, al, bl),
        "=" * 64,
    ]

    head = v["headline"]
    if head is None:
        lines.append("  VERDICT: dead heat on realized PnL.")
    else:
        margin = abs(a["realized_pnl"] - b["realized_pnl"])
        lines.append(f"  VERDICT: {head} leads on realized PnL by {money(margin)}.")
        # Flag a hollow win: ahead on PnL but behind on risk-adjusted return.
        if v["return_on_drawdown"] not in (None, head):
            lines.append(
                f"           ...but {v['return_on_drawdown']} has the better "
                "PnL-per-drawdown (took less heat to get there)."
            )

    if same_account:
        lines.append("")
        lines.append("  WARNING: both .env files resolve to the SAME account "
                     "number — the A/B is void. Give each bot a different keypair.")
    lines.append("")
    lines.append("  NOTE: one short window is NOT out-of-sample and the trade "
                 "count is tiny.")
    lines.append("        Treat this as a leaderboard, not a validated edge "
                 "(see CLAUDE.md).")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _load_account(label: str, env_path: str, days: int, rest_cls=None) -> Tuple[dict, str]:
    """Open one account, fetch its fills, compute metrics. Returns (metrics, acct_no)."""
    creds = parse_env_file(env_path)
    if not creds:
        raise SystemExit(f"[{label}] could not read any keys from {env_path!r}.")
    client, account = build_client_for(creds, rest_cls=rest_cls)
    acct_no = str(getattr(account, "account_number", "") or "")
    mode = endpoint_mode(creds.get("APCA_API_BASE_URL", PAPER_URL) or PAPER_URL)
    equity = None
    try:
        equity = float(getattr(account, "equity", None))
    except (TypeError, ValueError):
        equity = None
    print(f"[{label}] Alpaca {mode} — account {acct_no or '?'}, "
          f"equity {equity if equity is not None else '?'}")
    fills = _fetch_fills(client, days)
    return account_metrics(label, fills, equity=equity), acct_no


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Side-by-side realized-PnL comparison of two Alpaca accounts."
    )
    parser.add_argument(
        "--account", action="append", type=parse_account_arg, metavar="LABEL=ENVPATH",
        help="an account to compare, as LABEL=path/to/.env. Pass exactly twice.",
    )
    parser.add_argument("--days", type=int, default=7, help="lookback window (default 7)")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = parser.parse_args(argv)

    accounts = args.account or [("Old", ".env"), ("Swing", ".env.swing")]
    if len(accounts) != 2:
        parser.error("need exactly two --account entries (got "
                     f"{len(accounts)}).")

    (a_metrics, a_no) = _load_account(accounts[0][0], accounts[0][1], args.days)
    (b_metrics, b_no) = _load_account(accounts[1][0], accounts[1][1], args.days)
    same_account = bool(a_no) and a_no == b_no

    if args.json:
        print(json.dumps({
            "days": args.days,
            "accounts": [a_metrics, b_metrics],
            "verdict": build_verdict(a_metrics, b_metrics),
            "same_account": same_account,
        }, indent=2, default=lambda o: None if o == float("inf") else o))
        return

    print()
    print(format_comparison(a_metrics, b_metrics, args.days, same_account=same_account))


if __name__ == "__main__":
    main()
