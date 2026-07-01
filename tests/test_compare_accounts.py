import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import compare_accounts as ca


# ---- env parsing ------------------------------------------------------------
def test_parse_env_file_handles_quotes_comments_export(tmp_path):
    p = tmp_path / ".env"
    p.write_text(
        "# a comment\n"
        "\n"
        "APCA_API_KEY_ID=ABC123\n"
        "export APCA_API_SECRET_KEY='sec ret'\n"
        'APCA_API_BASE_URL="https://paper-api.alpaca.markets"  # inline\n'
        "PLAIN=value # trailing comment\n"
    )
    env = ca.parse_env_file(str(p))
    assert env["APCA_API_KEY_ID"] == "ABC123"
    assert env["APCA_API_SECRET_KEY"] == "sec ret"          # quotes stripped, space kept
    assert env["APCA_API_BASE_URL"] == "https://paper-api.alpaca.markets"
    assert env["PLAIN"] == "value"                           # unquoted inline comment dropped


def test_parse_env_missing_file_is_empty():
    assert ca.parse_env_file("/no/such/file.env") == {}


def test_parse_account_arg_windows_path():
    label, path = ca.parse_account_arg("Swing=C:/revbot_swing/.env")
    assert label == "Swing" and path == "C:/revbot_swing/.env"


# ---- drawdown ---------------------------------------------------------------
def test_max_realized_drawdown_pure_gain_is_zero():
    assert ca.max_realized_drawdown([("d1", 10.0), ("d2", 5.0)]) == 0.0


def test_max_realized_drawdown_peak_to_trough():
    # cumulative: +100, +50 (dip 50), +150 -> deepest dip from a peak is 50
    daily = [("d1", 100.0), ("d2", -50.0), ("d3", 100.0)]
    assert ca.max_realized_drawdown(daily) == 50.0


# ---- metrics from fills -----------------------------------------------------
def _round_trip(sym, buy_px, sell_px, qty=10, day="2026-06-25"):
    return [
        {"symbol": sym, "side": "buy", "qty": qty, "price": buy_px,
         "time": f"{day}T14:30:00Z"},
        {"symbol": sym, "side": "sell", "qty": qty, "price": sell_px,
         "time": f"{day}T15:30:00Z"},
    ]


def test_account_metrics_basic():
    fills = _round_trip("AAA", 100, 110) + _round_trip("BBB", 50, 45)
    m = ca.account_metrics("Swing", fills, equity=98000.0)
    assert m["realized_pnl"] == 100.0 * 0 + (110 - 100) * 10 + (45 - 50) * 10  # +100 -50 = 50
    assert m["round_trips"] == 2
    assert m["wins"] == 1 and m["losses"] == 1
    assert m["win_rate"] == 0.5
    assert m["turnover"] > 0
    assert m["equity"] == 98000.0


# ---- verdict ----------------------------------------------------------------
def test_pick_winner_and_verdict():
    a = ca.account_metrics("Old", _round_trip("X", 100, 90), equity=115000.0)      # -100
    b = ca.account_metrics("Swing", _round_trip("Y", 100, 120), equity=98000.0)    # +200
    v = ca.build_verdict(a, b)
    assert v["headline"] == "Swing"
    assert v["realized_pnl"] == "Swing"
    # Old lost money and drew down; lower max_drawdown is better -> Swing (0 DD)
    assert v["max_drawdown"] == "Swing"


def test_format_comparison_flags_same_account_and_prints_verdict():
    a = ca.account_metrics("Old", _round_trip("X", 100, 120))     # +200
    b = ca.account_metrics("Swing", _round_trip("Y", 100, 90))    # -100
    out = ca.format_comparison(a, b, days=7, same_account=True)
    assert "VERDICT: Old leads" in out
    assert "WARNING" in out and "SAME account" in out
    assert "out-of-sample" in out          # discipline caveat always present


# ---- client build (injected fake REST, no network) --------------------------
class _FakeAccount:
    account_number = "PA_TEST_1"
    status = "ACTIVE"
    equity = "98390.06"


class _FakeREST:
    def __init__(self, key, secret, base_url):
        self.key, self.secret, self.base_url = key, secret, base_url

    def get_account(self):
        return _FakeAccount()


def test_build_client_for_injected():
    creds = {"APCA_API_KEY_ID": "k", "APCA_API_SECRET_KEY": "s",
             "APCA_API_BASE_URL": "https://paper-api.alpaca.markets"}
    client, account = ca.build_client_for(creds, rest_cls=_FakeREST)
    assert account.account_number == "PA_TEST_1"
    assert client.base_url.endswith("paper-api.alpaca.markets")


def test_build_client_for_missing_creds_exits():
    import pytest
    with pytest.raises(SystemExit):
        ca.build_client_for({}, rest_cls=_FakeREST)
