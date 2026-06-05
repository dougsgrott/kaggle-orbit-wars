"""``flow_value_dr`` brain (AG15) — `flow_value` + friendly-flip defense + regroup.

`flow_value_def` (proactive defense) plus the Producer's **pressure-gradient
regroup**: after attack/defense waves, leftover ships (safe_drain not yet spent)
move toward a materially more enemy-stressed owned planet that is still mine at the
fleet's arrival turn. Regroup is ship-neutral so the flow-diff value can't reward
it — it's a separate positional heuristic, hence A/B'd on top of defense.

Public API:
    plan_turn(obs, config=None) -> list[list]
"""
from __future__ import annotations

from typing import List

from ..utils import aim_with_prediction
from .flow_value import _plan


def plan_turn(obs, config=None) -> List[list]:
    return _plan(obs, config, aim_with_prediction, enable_defense=True, enable_regroup=True)
