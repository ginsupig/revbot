import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_real_backtest import _apply_http_timeout


class _Session:
    def __init__(self):
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append(kwargs)
        return "RESP"


class _Client:
    def __init__(self, session=None):
        if session is not None:
            self._session = session


def test_injects_timeout_when_absent():
    sess = _Session()
    c = _apply_http_timeout(_Client(sess), 15.0)
    c._session.request("GET", "http://x")
    assert sess.calls[-1]["timeout"] == 15.0


def test_injects_when_timeout_is_none():
    sess = _Session()
    c = _apply_http_timeout(_Client(sess), 15.0)
    c._session.request("GET", "http://x", timeout=None)
    assert sess.calls[-1]["timeout"] == 15.0


def test_preserves_explicit_timeout():
    sess = _Session()
    c = _apply_http_timeout(_Client(sess), 15.0)
    c._session.request("GET", "http://x", timeout=3.0)
    assert sess.calls[-1]["timeout"] == 3.0


def test_noop_when_disabled_or_no_session():
    sess = _Session()
    # timeout <= 0 -> don't wrap
    _apply_http_timeout(_Client(sess), 0)
    sess.request("GET", "http://x")
    assert "timeout" not in sess.calls[-1]
    # client without a _session -> no crash
    _apply_http_timeout(_Client(), 15.0)
