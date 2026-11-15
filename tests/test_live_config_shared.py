"""main.py and the live-judging backtests must build config from the same code.

Fleet rule: a backtest that judges the live bot shares live's gates/settings
through shared functions, with a test that fails if they drift.
"""
import ast
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reversion_bot.config import ReversionConfig, RiskConfig
from reversion_bot.live_config import risk_config_from_env, strategy_config_from_env

ROOT = Path(__file__).resolve().parent.parent


def _calls(path, name):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and getattr(n.func, "id", getattr(n.func, "attr", None)) == name]


def test_main_builds_config_only_through_live_config():
    assert not _calls(ROOT / "main.py", "ReversionConfig")
    assert not _calls(ROOT / "main.py", "RiskConfig")
    assert _calls(ROOT / "main.py", "strategy_config_from_env")
    assert _calls(ROOT / "main.py", "risk_config_from_env")


def test_regime_ab_uses_live_builders_and_live_gate():
    path = ROOT / "research" / "regime_gate_ab.py"
    assert _calls(path, "strategy_config_from_env")
    assert _calls(path, "risk_config_from_env")
    assert _calls(path, "is_risk_off")


def test_env_reaches_both_configs(monkeypatch):
    monkeypatch.setenv("RSI_MAX", "33")
    monkeypatch.setenv("TARGET_ATR_MULTIPLE", "2.75")
    monkeypatch.delenv("MAX_POSITION_VALUE_PCT", raising=False)
    monkeypatch.setenv("MAX_POSITION_PCT", "0.045")
    s, r = strategy_config_from_env(160), risk_config_from_env()
    assert isinstance(s, ReversionConfig) and isinstance(r, RiskConfig)
    assert s.min_history == 160 and s.rsi_max == 33.0
    assert r.target_atr_multiple == 2.75 and r.max_position_value_pct == 0.045


def test_ml_probability_never_changes_go_long(tmp_path):
    """regime_gate_ab stubs the ML probability; that is only valid while ML
    cannot change an entry decision."""
    import numpy as np
    import pandas as pd
    from reversion_bot.config import PerformanceConfig
    from reversion_bot.service import ReversionService

    rng = np.random.default_rng(7)
    n = 400
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    close[-25:] *= np.linspace(1.0, 0.95, 25)          # push into an oversold dip
    df = pd.DataFrame({"date": pd.date_range("2026-03-02 15:00", periods=n, freq="5min", tz="UTC"),
                       "open": close, "high": close * 1.002, "low": close * 0.998,
                       "close": close, "volume": np.full(n, 5e5)})
    cfg = ReversionConfig(min_history=160, min_dollar_volume=0.0, require_spread_data=False)
    seen = []
    for p in (0.0, 1.0):
        svc = ReversionService(cfg, RiskConfig(), PerformanceConfig(state_dir=str(tmp_path / str(p))),
                               log_file=str(tmp_path / f"{p}.log"))
        svc._get_ml_probability = lambda _df, p=p: p
        seen.append([(r["go_long"], r["router_reason"], r["entry_style"])
                     for r in (svc.evaluate_symbol("T", df.iloc[i - 160:i].reset_index(drop=True), 1e5)
                               for i in range(200, n + 1, 5))])
    assert seen[0] == seen[1]
    assert any(g for g, _, _ in seen[0]) or any(e == "mean_reversion" for _, _, e in seen[0])
