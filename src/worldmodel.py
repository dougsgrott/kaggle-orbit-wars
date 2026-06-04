"""
WorldModel — the agent's forward model for lookahead / search (M3, ADR-0003).

A thin, deterministic wrapper over the official `interpreter()` called in-process.
A read-only spike showed the interpreter is **directly callable** (no `env.run`
harness), runs **~0.4 ms/step**, deep-copies in **~64 µs**, and **byte-matches
`env.run`** for a given seed — so it is both a fast forward model and its own
oracle (zero physics-divergence risk). We wrap it rather than transcribing a
standalone Engine; the custom Engine stays deferred until bulk rollout volume
bites (ADR-0001 / ADR-0003).

Like `src/arena.py`, this module imports `kaggle_environments`; the pure
heuristic brains stay env-free, but search brains that use this only run where
the env is installed.

State shape mirrors the interpreter's: a list of per-agent SimpleNamespaces
(`observation` / `action` / `status` / `reward`) plus an `env` SimpleNamespace
(`configuration` / `done` / `info`). `step()` returns a fresh, deep-copied state
so callers can fork the present and explore branches without mutating each other.

Public API:
    from_obs(obs, num_players=2, config=None) -> ForwardState
    step(fstate, actions_per_player)          -> ForwardState   (advanced 1 turn)
    rollout(fstate, policies, turns)          -> ForwardState   (advance N turns)
    score(fstate)                             -> list[int]      (ships per player)
    planets_of(fstate) / fleets_of(fstate) / step_of(fstate)   (accessors)
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, List, Optional, Sequence

from .arena import score_board

# Defaults matching the official env configuration (COMPETITION_DATA.md).
_DEFAULT_CONFIG = {
    "shipSpeed": 6.0,
    "episodeSteps": 500,
    "cometSpeed": 4.0,
}

# Obs fields the interpreter reads off state[0].observation each step.
_OBS_FIELDS = (
    "planets", "fleets", "initial_planets", "angular_velocity",
    "comets", "comet_planet_ids", "next_fleet_id", "step",
)


@dataclass
class ForwardState:
    """A simulatable game state: the interpreter's `state` list + its `env`."""

    state: list
    env: SimpleNamespace
    num_players: int


def _field(obs, key, default=None):
    """Read `key` from a dict-or-attr observation (mirrors the brains' accessor)."""
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def from_obs(obs, num_players: int = 2, config: Optional[dict] = None) -> ForwardState:
    """Build a simulatable `ForwardState` from a plain observation.

    `obs` is whatever a brain receives in `plan_turn` (dict or attr bag): it must
    carry the full board (`planets`, `fleets`, `initial_planets`,
    `angular_velocity`, `comets`, `comet_planet_ids`) — i.e. a live-game obs.
    Everything is deep-copied so the source obs is never mutated by stepping.
    """
    cfg = dict(_DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    planets = [list(p) for p in (_field(obs, "planets", []) or [])]
    fleets = [list(f) for f in (_field(obs, "fleets", []) or [])]
    initial = [list(p) for p in (_field(obs, "initial_planets", planets) or [])]
    # next_fleet_id must exceed every existing fleet id so new launches get
    # unique ids (a synthetic obs may omit it).
    nfi = _field(obs, "next_fleet_id", None)
    if nfi is None:
        nfi = (max((int(f[0]) for f in fleets), default=-1) + 1)

    board = SimpleNamespace(
        step=int(_field(obs, "step", 0) or 0),
        planets=planets,
        fleets=fleets,
        initial_planets=initial,
        angular_velocity=float(_field(obs, "angular_velocity", 0.0) or 0.0),
        comets=copy.deepcopy(_field(obs, "comets", []) or []),
        comet_planet_ids=list(_field(obs, "comet_planet_ids", []) or []),
        next_fleet_id=int(nfi),
        player=0,
    )

    state = [SimpleNamespace(observation=board, action=[], status="ACTIVE", reward=0)]
    for i in range(1, num_players):
        state.append(
            SimpleNamespace(
                observation=SimpleNamespace(player=i),
                action=[],
                status="ACTIVE",
                reward=0,
            )
        )
    env = SimpleNamespace(
        configuration=SimpleNamespace(**cfg),
        done=False,
        info={"seed": 0},
    )
    return ForwardState(state=state, env=env, num_players=num_players)


def step(fstate: ForwardState, actions_per_player: Sequence) -> ForwardState:
    """Advance one turn under `actions_per_player` (a list of move-lists, one per
    player). Returns a **fresh deep-copied** ForwardState; the input is unchanged.

    Mirrors the harness: bump `observation.step` on every agent, assign each
    agent's action, then call the interpreter once.
    """
    from kaggle_environments.envs.orbit_wars.orbit_wars import interpreter

    nxt = copy.deepcopy(fstate)
    next_step = int(getattr(nxt.state[0].observation, "step", 0)) + 1
    for i, agent_state in enumerate(nxt.state):
        agent_state.observation.step = next_step
        act = actions_per_player[i] if i < len(actions_per_player) else []
        agent_state.action = list(act) if act else []
    interpreter(nxt.state, nxt.env)
    return nxt


def rollout(
    fstate: ForwardState,
    policies: Sequence[Optional[Callable]],
    turns: int,
) -> ForwardState:
    """Advance `turns` turns, querying each player's `policies[i]` for moves each
    turn (a `plan_turn`-style callable, or None = do nothing). Returns the final
    deep-copied state. Stops early if the env marks the episode done.
    """
    cur = fstate
    for _ in range(turns):
        if getattr(cur.env, "done", False):
            break
        actions = []
        for i in range(cur.num_players):
            pol = policies[i] if i < len(policies) else None
            if pol is None:
                actions.append([])
            else:
                obs_i = obs_for_player(cur, i)
                actions.append(pol(obs_i) or [])
        cur = step(cur, actions)
    return cur


def obs_for_player(fstate: ForwardState, i: int):
    """A plain dict observation for player `i` — the shared board + that player's
    id, as a brain's `plan_turn` expects.

    Used when rolling out with policies, and by search brains (MCTS, AG9) that
    need to generate candidate moves / query an opponent model from a *simulated*
    state mid-tree, not just the live obs.
    """
    board = fstate.state[0].observation
    return {
        "player": i,
        "planets": board.planets,
        "fleets": board.fleets,
        "initial_planets": board.initial_planets,
        "angular_velocity": board.angular_velocity,
        "comets": board.comets,
        "comet_planet_ids": board.comet_planet_ids,
        "step": getattr(board, "step", 0),
    }


# Back-compat private alias (older callers referenced the underscored name).
_obs_for_player = obs_for_player


def planets_of(fstate: ForwardState) -> list:
    return fstate.state[0].observation.planets


def fleets_of(fstate: ForwardState) -> list:
    return fstate.state[0].observation.fleets


def step_of(fstate: ForwardState) -> int:
    return int(getattr(fstate.state[0].observation, "step", 0))


def score(fstate: ForwardState) -> List[int]:
    """Ships per player (on owned planets + in owned fleets) — reuses
    `arena.score_board`, the interpreter's own end-of-game scoring."""
    board = fstate.state[0].observation
    return score_board(board.planets, board.fleets, fstate.num_players)
