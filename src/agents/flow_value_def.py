"""``flow_value_def`` brain (AG15) — `flow_value` + friendly-flip proactive defense.

Identical to `flow_value` except owned planets the do-nothing projection shows
*flipping* within the horizon are added as **defensive targets**. The flow-diff
value already prices defense correctly (reinforcing a planet about to flip keeps its
production *and* denies it to the enemy → high `Δnet_me − Σ_opp Δnet_opp`), so the
only change is making those planets candidates; a role mutex keeps a reinforced
planet from also being a source. Isolates the defense lever (no regroup).

Public API:
    plan_turn(obs, config=None) -> list[list]
"""
from __future__ import annotations

from typing import List

from ..utils import aim_with_prediction
from .flow_value import _plan


def plan_turn(obs, config=None) -> List[list]:
    return _plan(obs, config, aim_with_prediction, enable_defense=True)
