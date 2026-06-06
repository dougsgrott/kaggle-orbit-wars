"""``flow_value_cfg`` brain (AG16) — the Producer's tuned per-format config.

The gap attribution ([wiki/producer_diff.md]) showed our value engine was sound but
**mistuned**: `flow_value_def` runs one config for both formats at our own knob
values, while the Producer switches a 2P preset (`ProducerLiteConfig`) and a 4P
preset (`CONFIG_4P`). Moved one at a time the knobs are flat/small, but moved
together they are **super-additive** — our engine goes from 0% to ~19–31% 1v1 vs the
`sota` Producer and 4P mean-place 2.75 → 2.00 (`fv_abl_all`).

This brain promotes that measured config to a first-class champion candidate: it
selects the Producer's preset by player count, on top of the AG13 value + AG15
defense, with the AG14 continuous-intercept aim (flat *alone* on the old base, but
part of the winning whole) and regroup on (an AG15 negative vs a mixed field, but
neutral/positive vs the Producer — kept a toggle, re-measured here).

    | knob              | 2P            | 4P            |
    |-------------------|---------------|---------------|
    | horizon H         | 18            | 13            |
    | sources/targets/def | 12 / 12 / 4 | 6 / 12 / 2    |
    | fire threshold    | 1.5           | 1.5           |
    | min ships/launch  | 4             | 4             |
    | defense / regroup | on / on       | on / on       |
    | aim               | intercept     | intercept     |

All planner logic is reused — this is pure config selection over
[flow_value.py]'s parameterised `_plan`. Requires `kaggle_environments`.

Public API:
    plan_turn(obs, config=None) -> list[list]
"""
from __future__ import annotations

from typing import List

from ..utils import intercept_aim
from .roi_greedy import _field
from .flow_value import _plan

# Per-format presets (the Producer's ProducerLiteConfig / CONFIG_4P, mapped to our
# _plan knobs). 2P is the default; 4P narrows sources + horizon + defensive quota.
_CFG_2P = dict(H=18, max_sources=12, max_targets=12, max_def=4)
_CFG_4P = dict(H=13, max_sources=6, max_targets=12, max_def=2)
# Format-independent knobs (shared by both presets).
_THRESHOLD = 1.5
_MIN_SPEND = 4


def _num_players(obs) -> int:
    planets = _field(obs, "planets")
    return max(2, max((int(p[1]) for p in planets), default=0) + 1)


def plan_turn(obs, config=None) -> List[list]:
    """Return this turn's Shots, scored by the competitive flow-diff value at the
    Producer's per-format config (AG16). See the module docstring."""
    fmt = _CFG_4P if _num_players(obs) >= 4 else _CFG_2P
    return _plan(
        obs, config, intercept_aim,
        enable_defense=True, enable_regroup=True,
        threshold=_THRESHOLD, min_spend=_MIN_SPEND, **fmt,
    )
