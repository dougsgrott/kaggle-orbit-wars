"""
M3 "lookahead" brain — greedy N-turn lookahead over the WorldModel (AG8).

The first **search** lever, after four heuristic levers all plateaued vs the Boss
(see wiki/measured_log.md). Instead of scoring shots with a one-turn heuristic,
this brain *simulates* a small set of candidate whole-turn moves forward through
the WorldModel (the in-process interpreter, ADR-0003) and keeps the move whose
**K-turn future board** scores best.

Candidate set (portfolio — keeps branching tiny and reuses everything we built):
the proposed move from each existing brain — v1 `roi_greedy_predict`, `missions`,
`roi_ledger`, `roi_defense` — plus **hold** (`[]`). Each is a legal, motion-aimed
move list already; we just decide *which* by simulation rather than by heuristic
score. Duplicate candidate moves are de-duplicated so we don't pay to simulate the
same future twice.

Evaluation: for each candidate, `WorldModel.step` it this turn (opponent modeled
as do-nothing — a deliberate first approximation), then `rollout` K−1 more
do-nothing turns, and score the leaf as `my_ships − max(opponent_ships)` via
`worldmodel.score`. Argmax wins; ties fall back to the earliest candidate (v1
first), so on a board where lookahead is indifferent we behave like v1.

Budget: candidates (≤5) × K steps × ~0.5 ms/step (incl. deepcopy) ≈ tens of ms,
well under the 1 s turn limit (K capped at LOOKAHEAD_TURNS; measured).

Requires `kaggle_environments` (via the WorldModel) — a search brain, not a pure
one. Pure `plan_turn(obs, config)` contract and legal Shot output regardless.

Public API:
    plan_turn(obs, config=None)   -> list[list]
"""
from __future__ import annotations

from typing import List

from .. import worldmodel as wm
from .roi_greedy import _field
from .roi_greedy_predict import plan_turn as _v1
from .missions import plan_turn as _missions
from .roi_ledger import plan_turn as _ledger
from .roi_defense import plan_turn as _defense

# How many turns to simulate each candidate forward. Small: the value comes from
# seeing captures/recaptures resolve, and the budget scales with this.
LOOKAHEAD_TURNS = 10

# Candidate move generators, in priority order (ties keep the earliest = v1).
_CANDIDATE_BRAINS = (_v1, _missions, _ledger, _defense)


def _candidate_moves(obs) -> List[list]:
    """The de-duplicated set of candidate whole-turn moves to evaluate: each
    brain's proposal plus hold ([])."""
    seen = set()
    cands: List[list] = []
    for brain in _CANDIDATE_BRAINS:
        mv = brain(obs) or []
        key = tuple(tuple(m) for m in mv)
        if key not in seen:
            seen.add(key)
            cands.append(mv)
    # Always consider holding fire too (sometimes the best move is to bank).
    if () not in seen:
        cands.append([])
    return cands


def _leaf_value(fstate, me: int) -> int:
    """Score a simulated future: my ships minus the strongest opponent's."""
    scores = wm.score(fstate)
    mine = scores[me]
    others = [s for i, s in enumerate(scores) if i != me]
    return mine - (max(others) if others else 0)


def plan_turn(obs, config=None) -> List[list]:
    """Pick the candidate move with the best K-turn simulated future. See module docstring."""
    me = int(_field(obs, "player"))
    planets = _field(obs, "planets")
    num_players = max((int(p[1]) for p in planets), default=0) + 1
    num_players = max(2, num_players)

    candidates = _candidate_moves(obs)
    if len(candidates) == 1:
        return candidates[0]  # nothing to choose between (e.g. all empty)

    base = wm.from_obs(obs, num_players=num_players, config=config if isinstance(config, dict) else None)

    best_move = candidates[0]
    best_val = None
    for mv in candidates:
        # This turn: I play `mv`, opponents do nothing (first-approx opponent model).
        actions = [[] for _ in range(num_players)]
        actions[me] = mv
        fs = wm.step(base, actions)
        # Then roll out the rest of the horizon with everyone holding.
        fs = wm.rollout(fs, policies=[None] * num_players, turns=LOOKAHEAD_TURNS - 1)
        val = _leaf_value(fs, me)
        if best_val is None or val > best_val:
            best_val = val
            best_move = mv

    return best_move
