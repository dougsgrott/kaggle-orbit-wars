"""Our candidate agent brains — distinct strategies to experiment with and compare.

This project is experimental: a single strategy may turn out weak, so brains are
kept side by side rather than mutated in place. Each module here exposes the same
pure contract — `plan_turn(obs, config=None) -> list[Shot]` (a Shot is
`[from_planet_id, angle, num_ships]`) — built on `src.utils`. That uniform
interface lets the eval harness sweep `REGISTRY` to rank every brain against each
other and the ladder, and lets the single Kaggle entry point (`src/agent.py`)
wrap whichever brain is current.

This is the counterpart to `src/opponents/`: opponents are *fixed yardsticks*
that never import the solution package; the brains here are *ours* and reuse it.

To add a strategy: drop a module exposing `plan_turn`, then register it below.
"""
from __future__ import annotations

from typing import Callable, Dict

from .roi_greedy import plan_turn as roi_greedy

# name -> plan_turn callable. The Kaggle entry point and the eval sweep both
# select brains by these names.
REGISTRY: Dict[str, Callable] = {
    "roi_greedy": roi_greedy,
}

# The brain `src/agent.py` submits unless told otherwise (our current best).
# Point this at a new entry once a better brain wins the comparison.
DEFAULT = "roi_greedy"
