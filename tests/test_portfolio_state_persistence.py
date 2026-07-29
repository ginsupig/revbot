"""PortfolioState persistence hardening (audit M6): atomic writes, loud
corruption fallback, checkout-anchored default state dir."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reversion_bot.portfolio import PortfolioState


def test_save_is_atomic_no_tmp_left_behind(tmp_path):
    st = PortfolioState(state_dir=str(tmp_path))
    st.update_equity(100_000.0)
    files = {p.name for p in tmp_path.iterdir()}
    assert "portfolio_state.json" in files
    assert not any(n.endswith(".tmp") for n in files)
    data = json.loads((tmp_path / "portfolio_state.json").read_text())
    assert data["last_equity"] == 100_000.0


def test_corrupt_state_falls_back_with_error_log(tmp_path, caplog):
    (tmp_path / "portfolio_state.json").write_text('{"peak_equity": 123, TRUNC')
    st = PortfolioState(state_dir=str(tmp_path))
    import logging
    with caplog.at_level(logging.ERROR):
        data = st._load()
    assert data["peak_equity"] is None            # defaulted
    assert any("unreadable" in r.message for r in caplog.records)


def test_default_state_dir_is_checkout_anchored(monkeypatch, tmp_path):
    # Launching from a random cwd must NOT relocate the state.
    monkeypatch.chdir(tmp_path)
    expected = Path(PortfolioState.DEFAULT_STATE_DIR)
    st = PortfolioState()
    assert st.state_dir == expected
    assert st.state_dir.is_absolute()
    repo_root = Path(__file__).resolve().parent.parent
    assert str(st.state_dir).startswith(str(repo_root))
