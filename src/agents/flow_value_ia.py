"""``flow_value_ia`` brain (AG14) — `flow_value` with continuous-intercept aim.

Identical to `flow_value` (the competitive flow-diff champion) except the aimer:
it uses `utils.intercept_aim` (a sub-turn continuous-fixed-point lead + a byte-exact
**swept-pair** first-contact verify) instead of `aim_with_prediction`. Measured
motivation: ~4.8% of `flow_value`'s fleets sail out of bounds / into the sun
because the default verify checks the target's *endpoint* each turn rather than its
*moving* segment, approving near-misses on fast-orbiting targets. The swept-pair
verify rejects those (we keep the ships home) and the continuous lead finds better
angles. The A/B isolates the aimer — every other part of the planner is shared.

Public API:
    plan_turn(obs, config=None) -> list[list]
"""
from __future__ import annotations

from typing import List

from ..utils import intercept_aim
from .flow_value import _plan


def plan_turn(obs, config=None) -> List[list]:
    """`flow_value`'s planner with the continuous-intercept aimer."""
    return _plan(obs, config, intercept_aim)
