import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    assert plan.side == 'long'
    assert plan.stop_price < plan.entry_price < plan.target_price
    assert plan.rr_ratio >= 1.2


def test_build_short_plan_inverts_stop_and_target():
    decision = ReversionDecision(
        signal='SHORT_REVERSION',
        reason='Validated_Short_Reversion',
        symbol='SPY',
        close=100.0,
        sma=98.0,
        atr=0.8,
    )
    plan = RiskManager().build_short_plan(account_equity=100000, decision=decision)
    assert plan.qty > 0
    assert plan.side == 'short'
    # Mirror of the long bracket: stop above entry, target below.
    assert plan.target_price < plan.entry_price < plan.stop_price
    assert plan.rr_ratio >= 1.0
    assert plan.risk_per_share > 0 and plan.reward_per_share > 0


def test_short_and_long_plans_size_identically():
    # Same entry/atr/equity -> identical risk budget and share count either way.
    long_dec = ReversionDecision(signal='LONG_REVERSION', reason='x', close=100.0, atr=0.8)
    short_dec = ReversionDecision(signal='SHORT_REVERSION', reason='x', close=100.0, atr=0.8)
    rm = RiskManager()
    long_plan = rm.build_long_plan(account_equity=100000, decision=long_dec)
    short_plan = rm.build_short_plan(account_equity=100000, decision=short_dec)
    assert long_plan.qty == short_plan.qty
    assert long_plan.risk_per_share == short_plan.risk_per_share
