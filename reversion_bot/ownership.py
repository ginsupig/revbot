"""Order ownership: scope destructive actions to orders revbot actually placed.

WHY
---
``main.liquidate_all_positions`` used to call ``cancel_all_orders()`` and then
``close_position`` on every row of ``list_positions()``. No universe filter, no
owner filter. It fires three times a session (session end, the EOD window, and
``reconcile_carryover`` on the day's first cycle), and SWING_OVERNIGHT_MODE=false
leaves all three live -- so it flattened the whole account, several times a day.

It caused no damage only because revbot is the sole occupant of its paper
account. revbot's .env previously held ACCOUNT 1 keys; there, the morning
carryover sweep would have cancelled gapbot's brackets and liquidated its
positions before the bell.

This is the pattern rotation ('rot-') and gapbot ('gapbot-') already use: tag
our own orders, and never touch anything else.

FAIL-CLOSED
-----------
``owned_symbols`` returns ``None`` -- not an empty set -- when the broker lookup
fails. None means "ownership unknown" and callers must touch NOTHING. An empty
set would be indistinguishable from a genuinely empty account, which is exactly
how a transient API error would silently restore flatten-everything behaviour.
"""
from __future__ import annotations

import logging
import uuid

COID_PREFIX = "revbot-"


def tag_client_order_id(existing: str | None = None) -> str:
    """Return a revbot-owned client_order_id, preserving one already supplied."""
    if existing:
        return existing
    return f"{COID_PREFIX}{uuid.uuid4().hex}"


def is_ours(order) -> bool:
    """True only when the order carries our client_order_id prefix.

    Anything unlabelled is somebody else's -- another bot's, or a hand trade.
    Absence of a tag is never treated as ownership.
    """
    coid = getattr(order, "client_order_id", None)
    return bool(coid) and str(coid).startswith(COID_PREFIX)


def owned_symbols(client, status: str = "all", limit: int = 500):
    """Set of symbols revbot opened, or None when that cannot be determined.

    None is the "unknown" signal; see the fail-closed note in the module
    docstring. Callers MUST NOT coerce it to an empty set.
    """
    try:
        orders = client.list_orders(status=status, limit=limit)
    except Exception as exc:  # noqa: BLE001 - unknown, never assume
        logging.warning("ownership: order lookup failed (%s); treating ownership "
                        "as UNKNOWN and touching nothing", exc)
        return None
    out = set()
    for o in orders or []:
        if is_ours(o):
            sym = str(getattr(o, "symbol", "") or "").upper()
            if sym:
                out.add(sym)
    return out


def cancel_our_open_orders(client) -> int:
    """Cancel only revbot's own working orders. Returns how many were cancelled.

    Deliberately does NOT call cancel_all_orders(): that strips the protective
    stops off every other position in the account.
    """
    try:
        orders = client.list_orders(status="open", limit=500)
    except Exception as exc:  # noqa: BLE001
        logging.warning("ownership: open-order lookup failed (%s); cancelling "
                        "nothing", exc)
        return 0
    n = 0
    for o in orders or []:
        if not is_ours(o):
            continue
        oid = getattr(o, "id", None)
        if oid is None:
            continue
        try:
            client.cancel_order(oid)
            n += 1
        except Exception as exc:  # noqa: BLE001 - one leg must not block the rest
            logging.warning("ownership: cancel failed for %s: %s", oid, exc)
    return n
