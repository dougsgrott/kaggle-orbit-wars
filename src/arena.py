"""
Arena — the stable match-running seam (see docs/adr/0001).

`run_episode(agents, config)` plays one Orbit Wars episode and returns a
structured `EpisodeResult` giving each agent its **Placement** and final
ship-count score. This module is the *only* place that imports
`kaggle_environments`; every caller depends on this interface instead of the
Official env. When the custom Engine arrives (ADR-0001) it slots in behind
`run_episode` with no caller changes.

Design notes
------------
- **Placement is derived from the final board**, not the env reward. The
  orbit_wars interpreter only stores reward = +1 for every score-leader and
  -1 for everyone else (and gives +1 to *all* tied leaders), so it cannot rank
  a 4-player game or break ties. We instead sum ships on owned planets plus
  ships in owned fleets at game end — exactly the interpreter's own scoring —
  and rank deterministically (higher score first; ties broken by lower player
  index). The raw env reward is still carried on each outcome for reference.
- **Player-count-agnostic from the start.** `num_players` flows through to the
  env; 2-player is exercised in slice A1, 4-player FFA lands in A2 through this
  same interface.

Public API
----------
    run_episode(agents, config=None)            -> EpisodeResult
    EpisodeConfig(num_players=2, seed=None, ...)
    EpisodeResult                               (.outcomes/.ranking/.winner/...)
    AgentOutcome                                (.index/.placement/.score/...)
    score_board(planets, fleets, num_players)   -> list[int]      (pure)
    compute_placements(scores)                  -> list[int]      (pure)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

# Quiet kaggle_environments' noisy OpenSpiel import banner (dozens of INFO lines
# emitted the first time the package is imported). We target only that logger so
# we don't touch global logging config. Setting the level before the module is
# imported still takes effect: the logger object is created/looked-up by name
# here, and its level is consulted at emit time inside run_episode().
logging.getLogger("kaggle_environments.envs.open_spiel_env.open_spiel_env").setLevel(
    logging.WARNING
)

# An agent is anything the Official env's `env.run` accepts: a callable
# `agent(obs, config) -> moves`, a path to a Kaggle-compatible .py file, or a
# builtin name registered by the env ("random" / "starter").
Agent = Union[str, Callable]


@dataclass(frozen=True)
class EpisodeConfig:
    """Inputs for one episode. Env-agnostic on purpose: the Engine swap (per
    ADR-0001) must read the same config."""

    num_players: int = 2
    seed: Optional[int] = None
    # Optional knobs; left None means "use the env's own default".
    episode_steps: Optional[int] = None
    act_timeout: Optional[float] = None


@dataclass(frozen=True)
class AgentOutcome:
    """One agent's result in an episode."""

    index: int  # player slot, 0..num_players-1
    placement: int  # 1 = best (most ships); distinct per agent, ties broken below
    score: int  # final ship count: ships on owned planets + ships in owned fleets
    reward: Optional[float]  # raw env reward (+1 leader / -1 other / None on error)
    status: str  # final agent status reported by the env ("DONE", "ERROR", ...)


@dataclass(frozen=True)
class EpisodeResult:
    """Structured outcome of one episode. Downstream aggregation reads these
    fields and never re-parses environment internals."""

    outcomes: Sequence[AgentOutcome]  # indexed by player slot
    num_players: int
    seed: Optional[int]  # the seed actually used (resolved by the env)
    num_steps: int  # number of recorded steps (turns played + 1)

    @property
    def ranking(self) -> List[int]:
        """Agent indices ordered best -> worst (i.e. by Placement)."""
        return [o.index for o in sorted(self.outcomes, key=lambda o: o.placement)]

    @property
    def winner(self) -> int:
        """Index of the 1st-place agent."""
        return self.ranking[0]

    def placement_of(self, index: int) -> int:
        return self.outcomes[index].placement

    def score_of(self, index: int) -> int:
        return self.outcomes[index].score


# ---------------------------------------------------------------------------
# Pure helpers (no env dependency) — kept module-level so they can be unit
# tested without kaggle_environments.
# ---------------------------------------------------------------------------


def score_board(planets, fleets, num_players: int) -> List[int]:
    """Final score per player = ships on owned planets + ships in owned fleets.

    Mirrors the orbit_wars interpreter's own end-of-game scoring. Neutral
    owners (-1) and any out-of-range owner are ignored.
    """
    scores = [0] * num_players
    for p in planets:
        owner = p[1]
        if 0 <= owner < num_players:
            scores[owner] += p[5]
    for f in fleets:
        owner = f[1]
        if 0 <= owner < num_players:
            scores[owner] += f[6]
    return scores


def compute_placements(scores: Sequence[int]) -> List[int]:
    """Deterministic 1-based Placement per player index from final scores.

    Higher score ranks first. Ties are broken by lower player index, so every
    agent gets a distinct Placement (1st, 2nd, ...) — this keeps downstream
    placement aggregation unambiguous and reproducible.
    """
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    placements = [0] * len(scores)
    for rank, i in enumerate(order):
        placements[i] = rank + 1
    return placements


# ---------------------------------------------------------------------------
# The Arena entry point.
# ---------------------------------------------------------------------------


def run_episode(
    agents: Sequence[Agent], config: Optional[EpisodeConfig] = None
) -> EpisodeResult:
    """Play one Orbit Wars episode and return a structured `EpisodeResult`.

    `agents` is a list of agents (callables, .py paths, or builtin names), one
    per player slot. `config` carries player count and seed. This is the only
    function that touches the Official env.
    """
    if config is None:
        config = EpisodeConfig()
    if len(agents) != config.num_players:
        raise ValueError(
            f"expected {config.num_players} agents, got {len(agents)}"
        )

    # Local import: kaggle_environments is heavy and is the seam we encapsulate.
    from kaggle_environments import make

    env_config: dict = {}
    if config.seed is not None:
        env_config["seed"] = config.seed
    if config.episode_steps is not None:
        env_config["episodeSteps"] = config.episode_steps
    if config.act_timeout is not None:
        env_config["actTimeout"] = config.act_timeout

    env = make("orbit_wars", configuration=env_config, debug=False)
    env.run(list(agents))

    return _extract_result(env, config.num_players)


def _extract_result(env, num_players: int) -> EpisodeResult:
    """Turn a finished env into an EpisodeResult — the result-extraction half of
    the encapsulated seam."""
    last = env.steps[-1]
    obs = last[0].observation  # full board is shared across all agent slots
    scores = score_board(obs["planets"], obs["fleets"], num_players)
    placements = compute_placements(scores)

    outcomes = tuple(
        AgentOutcome(
            index=i,
            placement=placements[i],
            score=int(scores[i]),
            reward=last[i].reward,
            status=last[i].status,
        )
        for i in range(num_players)
    )
    seed = (env.info or {}).get("seed") if getattr(env, "info", None) else None
    return EpisodeResult(
        outcomes=outcomes,
        num_players=num_players,
        seed=seed,
        num_steps=len(env.steps),
    )


# ---------------------------------------------------------------------------
# Demo: print the placements for one 2-player game.
#   python -m src.arena
# ---------------------------------------------------------------------------


def _demo() -> None:
    opp = Path(__file__).parent / "opponents"
    agents = [str(opp / "weakest_first.py"), str(opp / "production_first.py")]
    names = ["weakest_first", "production_first"]

    result = run_episode(agents, EpisodeConfig(num_players=2, seed=2026))

    print(f"Orbit Wars episode — seed {result.seed}, {result.num_steps} steps")
    for o in sorted(result.outcomes, key=lambda o: o.placement):
        medal = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}[o.placement]
        print(f"  {medal}  {names[o.index]:<18} score={o.score:>6}  reward={o.reward}")
    print(f"Winner: {names[result.winner]}")


if __name__ == "__main__":
    _demo()
