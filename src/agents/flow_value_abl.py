"""Ablation brains — single steps from ``flow_value_def`` toward the Producer.

These are **experiment-only** (not promoted, never `DEFAULT`, not submitted). Each
is `flow_value_def` (our shipped brain: AG13 value + AG15 defense, `aim_with_prediction`)
with exactly one knob moved to the Producer's setting, so an A/B vs the `sota`
Producer attributes how much of the 1-5 gap that one difference accounts for. The
cumulative `fv_abl_all` runs our value engine at the Producer's full per-format
config — the pivotal "config vs structural" read. See wiki/producer_diff.md.

`_plan`'s knobs default to the module constants, so importing/using these does not
change `flow_value` / `flow_value_def` behaviour.
"""
from __future__ import annotations

from typing import List

from ..utils import aim_with_prediction, intercept_aim
from .roi_greedy import _field
from .flow_value import _plan


def _num_players(obs) -> int:
    planets = _field(obs, "planets")
    return max(2, max((int(p[1]) for p in planets), default=0) + 1)


def _fmt_knobs(obs) -> dict:
    """The Producer's per-format projection/shortlist preset (2P vs 4P)."""
    if _num_players(obs) >= 4:                      # CONFIG_4P
        return dict(H=13, max_sources=6, max_targets=12, max_def=2)
    return dict(H=18, max_sources=12, max_targets=12, max_def=4)  # ProducerLiteConfig


# --- single-delta ablations (base = flow_value_def) -------------------------
def fv_abl_h(obs, config=None) -> List[list]:
    """#4 horizon: 14 -> 18 (2P) / 13 (4P)."""
    n = _num_players(obs)
    return _plan(obs, config, aim_with_prediction, enable_defense=True,
                 H=(13 if n >= 4 else 18))


def fv_abl_wide(obs, config=None) -> List[list]:
    """#5 shortlist widths: Producer per-format source/target caps."""
    n = _num_players(obs)
    src = 6 if n >= 4 else 12
    return _plan(obs, config, aim_with_prediction, enable_defense=True,
                 max_sources=src, max_targets=12)


def fv_abl_thresh(obs, config=None) -> List[list]:
    """#6 fire threshold: 2.0 -> 1.5."""
    return _plan(obs, config, aim_with_prediction, enable_defense=True, threshold=1.5)


def fv_abl_minship(obs, config=None) -> List[list]:
    """#7 min ships per launch: 1 -> 4."""
    return _plan(obs, config, aim_with_prediction, enable_defense=True, min_spend=4)


def fv_abl_aim(obs, config=None) -> List[list]:
    """#1 aim: aim_with_prediction -> intercept_aim (cheap proxy for byte-exact)."""
    return _plan(obs, config, intercept_aim, enable_defense=True)


def fv_abl_regroup(obs, config=None) -> List[list]:
    """#8 regroup: OFF -> ON."""
    return _plan(obs, config, aim_with_prediction, enable_defense=True, enable_regroup=True)


# --- cumulative ------------------------------------------------------------
def fv_abl_format(obs, config=None) -> List[list]:
    """#3 per-format preset: H + shortlist widths set by player count."""
    return _plan(obs, config, aim_with_prediction, enable_defense=True, **_fmt_knobs(obs))


def fv_abl_all(obs, config=None) -> List[list]:
    """Our value engine at the Producer's FULL per-format config (the pivotal read):
    per-format H+widths + threshold 1.5 + min_spend 4 + regroup ON + intercept aim."""
    return _plan(obs, config, intercept_aim, enable_defense=True, enable_regroup=True,
                 threshold=1.5, min_spend=4, **_fmt_knobs(obs))
