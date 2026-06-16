"""Shared Alpaca client builder for the read-only report scripts.

Gives pnl_report / fees_report (and churn_report / pnl_7_14, which reuse them)
a startup preflight that mirrors the bot's: it validates the keys with one
get_account() call, prints which endpoint it's actually using, and on an auth
failure prints a clear PAPER-vs-LIVE hint and exits — instead of dumping a raw
APIError traceback.

This is the recurring trap: load_dotenv() (override=False, the house default)
lets a stale shell APCA_API_BASE_URL win over .env, and the scripts default to
the paper endpoint when it's unset — so live keys hit paper and Alpaca returns
"request is not authorized".
"""
from __future__ import annotations

import os

PAPER_URL = "https://paper-api.alpaca.markets"
_AUTH_TOKENS = ("not authorized", "unauthorized", "forbidden", "401", "403")


def endpoint_mode(base_url: str) -> str:
    return "PAPER" if "paper" in (base_url or "").lower() else "LIVE"


def is_auth_error(exc: Exception) -> bool:
    return any(t in str(exc).lower() for t in _AUTH_TOKENS)


def _auth_failure_message(mode: str, base_url: str, exc: Exception) -> str:
    bar = "=" * 70
    if is_auth_error(exc):
        return (
            f"\n{bar}\n"
            f"[report] Alpaca REJECTED your credentials for the {mode} endpoint\n"
            f"[report]   ({base_url}).\n"
            f"[report] Almost always a paper/live mismatch: the key/secret don't\n"
            f"[report] match the URL. load_dotenv() does NOT override a shell var,\n"
            f"[report] so a stale $env:APCA_API_BASE_URL beats your .env. Check:\n"
            f"[report]     echo $env:APCA_API_BASE_URL\n"
            f"[report] and set it to the endpoint your keys belong to, e.g.\n"
            f"[report]     $env:APCA_API_BASE_URL = \"https://api.alpaca.markets\"\n"
            f"{bar}\n"
        )
    return f"\n[report] Could not reach Alpaca ({mode}, {base_url}): {exc}\n"


def make_client(rest_cls=None):
    """Build a validated Alpaca REST client for a report script.

    Prints the endpoint/mode on success; on bad/missing creds prints a clear
    hint and raises SystemExit (no traceback). ``rest_cls`` is injectable for
    tests so no real SDK/network is needed.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    key = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")
    base_url = os.getenv("APCA_API_BASE_URL", PAPER_URL)
    if not key or not secret:
        raise SystemExit(
            "Missing Alpaca creds. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY "
            "(see .env.example)."
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

    print(f"[report] Alpaca {mode} ({base_url}) — account "
          f"{getattr(account, 'account_number', '?')} "
          f"{getattr(account, 'status', '?')}, equity {getattr(account, 'equity', '?')}")
    return client
