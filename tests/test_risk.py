from reversion_bot.models import ReversionDecision
from reversion_bot.risk import RiskManager


def test_build_long_plan():
    decision = ReversionDecision(
        signal='LONG_REVERSION',
        reason='Validated_Long_Reversion',
        symbol='SPY',
        close=100.0,
        sma=102.0,
        atr=0.8,
    )
    plan = RiskManager().build_long_plan(account_equity=100000, decision=decision)
    assert plan.qty > 0
    assert plan.stop_price < plan.entry_price < plan.target_price
    assert plan.rr_ratio >= 1.2
