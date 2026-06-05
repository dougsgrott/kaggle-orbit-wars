"""
M4b "mcts_om" brain — MCTS with a real opponent model + 4P-aware value (AG10).

AG9 showed plain `mcts` is *flat* vs greedy `lookahead`: with a do-nothing opponent
model, deeper search just finds steamroll lines a real opponent would punish (search
*depth* wasn't the bottleneck — the opponent model was). This brain fills the two
hooks AG9 left open, the indicated unlock for the measured weakness (**4P, where we
consistently lose** — see wiki/measured_log.md):

  * **Opponent model:** rivals play **v1** (`roi_greedy_predict`, motion-aware greedy)
    at every **tree edge** and the first few **rollout** turns, instead of sitting
    still, so the search anticipates the response to its moves ("if I empty my home to
    attack, the rival counter-captures it") rather than assuming a passive board. (A
    *full* per-turn policy rollout overshot the 1 s budget and faulted, so opponents
    act only for the first `OPP_ROLLOUT_TURNS` — enough to launch the punishing fleet.)
  * **4P-aware value:** leaves are scored by **placement** (`value_placement`) — the
    Elo basis Kaggle ranks on — not raw `my − max(opponent)`. This stops the search
    from over-committing to pad a lead (which paints a target in FFA) and optimises
    for *finishing well* among all players.

Everything else is the AG9 search core unchanged (`mcts_plan`): same UCT, candidate
portfolio, anytime budget. This brain is just `mcts_plan` with both hooks supplied,
so the two levers are isolable for A/B (pass one hook at a time).

Requires `kaggle_environments` (via the WorldModel). Pure `plan_turn(obs, config)`.

Public API:
    plan_turn(obs, config=None) -> list[list]
"""
from __future__ import annotations

from typing import List

from .roi_greedy import _field
from .roi_greedy_predict import plan_turn as _v1
from .mcts import mcts_plan, value_placement


def plan_turn(obs, config=None) -> List[list]:
    """MCTS with v1 opponents + placement-aware leaves. See module docstring."""
    planets = _field(obs, "planets") or []
    num_players = max(2, max((int(p[1]) for p in planets), default=0) + 1)
    return mcts_plan(
        obs,
        num_players=num_players,
        opponent_policy=_v1,
        value_fn=value_placement,
        config=config if isinstance(config, dict) else None,
    )
