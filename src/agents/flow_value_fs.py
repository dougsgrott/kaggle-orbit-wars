"""``flow_value_fs`` brain (AG18, cheap test) — AG16 config, scored without truncation.

The gap-attribution structural lever #2 ([wiki/producer_diff.md]) is "the value
time-truncates candidate scoring." Measured: in single-process play `flow_value_cfg`
truncates ~7-8% of turns (the heavy boards, up to ~120 candidates), dropping 19-63
candidates and falling back on the crude `prio` pre-sort — yet the slowest turn is
only ~0.5 s, far under the 1 s soft `actTimeout`. So the truncation is caused by the
conservative 0.40 s soft cap, not a compute wall.

This brain is `flow_value_cfg` with the per-turn caps **raised** (soft 0.90 s / hard
0.95 s) so heavy turns score every candidate, while staying bank-safe (the worst
observed full-score turn is well under 1 s; the 60 s overage bank absorbs rare
spikes). It is the cheap, low-risk test of whether *scoring every candidate* lifts
placement — before committing to the batched-numpy analytic projector (AG18 proper),
which would buy the same "no truncation" more cheaply and enable wider shortlists.

Public API:
    plan_turn(obs, config=None) -> list[list]
"""
from __future__ import annotations

from typing import List

from ..utils import intercept_aim
from .flow_value import _plan
from .flow_value_cfg import _num_players, _CFG_2P, _CFG_4P, _THRESHOLD, _MIN_SPEND

# Raised per-turn scoring caps (vs flow_value's 0.40 s / 0.85 s). Heavy boards
# full-score in ~0.5-0.9 s; bank-safe (1 s soft timeout + 60 s overage bank).
_SOFT_BUDGET_S = 0.90
_HARD_BUDGET_S = 0.95


def plan_turn(obs, config=None) -> List[list]:
    """AG16 per-format config, scored without truncation (raised time caps)."""
    fmt = _CFG_4P if _num_players(obs) >= 4 else _CFG_2P
    return _plan(
        obs, config, intercept_aim,
        enable_defense=True, enable_regroup=True,
        threshold=_THRESHOLD, min_spend=_MIN_SPEND,
        soft_budget=_SOFT_BUDGET_S, hard_budget=_HARD_BUDGET_S, **fmt,
    )
