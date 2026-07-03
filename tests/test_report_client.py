import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from reversion_bot.report_client import endpoint_mode, is_auth_error, make_client


def test_endpoint_mode():
    assert endpoint_mode("https://paper-api.alpaca.markets") == "PAPER"
    assert endpoint_mode("https://api.alpaca.markets") == "LIVE"


def test_is_auth_error():
    assert is_auth_error(Exception("request is not authorized")) is True
    assert is_auth_error(Exception("403 Forbidden")) is True
    assert is_auth_error(Exception("connection timed out")) is False


class _Acct:
    status = "ACTIVE"
    equity = "1000"


class _OkREST:
    def __init__(self, *a):
        pass

    def get_account(self):
        return _Acct()


class _AuthREST:
    def __init__(self, *a):
        pass

    def get_account(self):
        raise Exception("request is not authorized")


def _set_creds(monkeypatch, url="https://api.alpaca.markets"):
    monkeypatch.setenv("APCA_API_KEY_ID", "LIVEKEY123")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "secret")
    monkeypatch.setenv("APCA_API_BASE_URL", url)


def test_missing_creds_exits(monkeypatch):
    # make_client() calls load_dotenv(), which would repopulate the creds from the
    # on-disk .env and defeat the delenv below. Stub it so this test hermetically
    # exercises the genuine "no credentials present" safety exit.
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    with pytest.raises(SystemExit):
        make_client(rest_cls=_OkREST)


def test_auth_error_exits_with_clear_hint(monkeypatch, capsys):
    _set_creds(monkeypatch)
    with pytest.raises(SystemExit):
        make_client(rest_cls=_AuthREST)
    out = capsys.readouterr().out.lower()
    assert "rejected your credentials" in out
    assert "paper/live" in out
    assert "apca_api_base_url" in out


def test_success_returns_client_and_announces_endpoint(monkeypatch, capsys):
    _set_creds(monkeypatch, "https://api.alpaca.markets")
    client = make_client(rest_cls=_OkREST)
    assert isinstance(client, _OkREST)
    out = capsys.readouterr().out
    assert "LIVE" in out and "ACTIVE" in out
