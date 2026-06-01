"""
Our single Kaggle entry point — a thin wrapper over the current **DEFAULT** brain.

The Kaggle environment expects a top-level `def agent(obs, config=None)` that
returns a list of moves `[[from_planet_id, angle, num_ships], ...]`. All actual
decision logic lives in the coexisting brains under `src/agents/`
(see docs/adr/0002); this module just selects `agents.DEFAULT` from the registry
and forwards the call. Swapping what we submit is changing the `DEFAULT` pointer,
not this file.

Import note: the Official env loads this file by reading its *source* and
exec'ing it (it is not imported as the `src.agent` module), so this uses an
**absolute** import (`from src.agents ...`), not a relative one. Locally — under
`arena.run_episode` / `eval.run_match` — the repo root is on `sys.path`, so this
resolves. Packaging a single self-contained submission file for Kaggle (inlining
the brain + utils) is AG3.
"""

from __future__ import annotations

from src.agents import REGISTRY, DEFAULT

_PLAN_TURN = REGISTRY[DEFAULT]


def agent(obs, config=None):
    return _PLAN_TURN(obs, config)
