from reversion_bot.analytics import run_monte_carlo


def test_monte_carlo_runs():
    out = run_monte_carlo(10000, 0.0005, 0.01, days=50, sims=500, seed=1)
    assert out['sims'] == 500
    assert out['days'] == 50
    assert 'var_95' in out
