Reversion Bot Rewrite Pack

Included files:
- service.py
- risk.py
- execution.py

What changed:
- service.py now uses weighted scoring instead of 2-of-4 binary voting
- ML contribution uses probability instead of class label
- risk.py can scale size modestly by conviction score without adding a new gate
- execution.py stays live-only and straightforward, with no preview path added

Drop-in order:
1. Replace your current reversion_bot/service.py
2. Replace your current reversion_bot/risk.py
3. Replace your current reversion_bot/execution.py

No extra gating layers were introduced.
