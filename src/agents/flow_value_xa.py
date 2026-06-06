"""``flow_value_xa`` brain (AG17) — AG16 per-format config + byte-exact aim.

The first structural lever the gap attribution ([wiki/producer_diff.md] #1) said the
tuned config *can't* buy. `flow_value_cfg` (AG16) still aims with the AG14
continuous-intercept solver over `predict_target_position`, an orbit approximation
that does not byte-match the engine over a multi-turn flight — the residual-miss
source AG14 diagnosed and deferred. This brain swaps that for `intercept_aim_exact`,
which leads + swept-pair verifies against the **interpreter's own per-turn positions**
(`traj_xy` from the projection), so there is no prediction error.

Because our value scores a candidate by *replaying the launch through the
interpreter with the aimed angle*, a cleaner aim should help twice: fewer whiffs and
a less aim-coupled value signal (good captures no longer suppressed by an aim wobble).

Identical to `flow_value_cfg` otherwise (same per-format config); the only change is
`exact_aim=True`. Requires `kaggle_environments`.

Public API:
    plan_turn(obs, config=None) -> list[list]
"""
from __future__ import annotations

from typing import List

from ..utils import aim_with_prediction  # placeholder; unused on the exact-aim path
from .flow_value import _plan
from .flow_value_cfg import _num_players, _CFG_2P, _CFG_4P, _THRESHOLD, _MIN_SPEND


def plan_turn(obs, config=None) -> List[list]:
    """AG16 per-format config, aimed byte-exact against the projection (AG17)."""
    fmt = _CFG_4P if _num_players(obs) >= 4 else _CFG_2P
    return _plan(
        obs, config, aim_with_prediction, exact_aim=True,
        enable_defense=True, enable_regroup=True,
        threshold=_THRESHOLD, min_spend=_MIN_SPEND, **fmt,
    )
