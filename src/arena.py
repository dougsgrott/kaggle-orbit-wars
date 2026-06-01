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
- **Player-count-agnostic.** `num_players` (2 or 4) flows through to the env;
  the same interface plays 1v1 and 4P FFA — only the number changes.
- **Fault-tolerant.** A callable agent that raises or blows its per-turn budget
  is contained by a guard (the turn yields no Shots; the agent is skipped for
  the rest of the game) rather than aborting the episode. File/builtin agents
  can't be guarded in-process, so their faults are detected from the env's
  per-step status instead. Either way a faulted agent is forced to **last
  Placement**, and the episode still returns a full result.

Public API
----------
    run_episode(agents, config=None)            -> EpisodeResult
    record_episode(agents, config=None)         -> EpisodeTrace   (per-turn record)
    EpisodeConfig(num_players=2, seed=None, ...)
    EpisodeResult                               (.outcomes/.ranking/.winner/...)
    EpisodeTrace                                (.frames/.result/...)
    AgentOutcome                          (.index/.placement/.score/.faulted/...)
    score_board(planets, fleets, num_players)   -> list[int]      (pure)
    compute_placements(scores, faulted=None)    -> list[int]      (pure)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Set, Union

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
    faulted: bool = False  # raised or timed out -> forced to last Placement


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


@dataclass(frozen=True)
class Frame:
    """One recorded turn of an episode — the board after step `step`."""

    step: int
    planets: list  # [[id, owner, x, y, radius, ships, production], ...]
    fleets: list  # [[id, owner, x, y, angle, from_planet_id, ships], ...]
    actions: list  # per-player action submitted that turn (moves, or None)


@dataclass(frozen=True)
class EpisodeTrace:
    """The full per-turn record of an episode: every `Frame` plus the final
    `EpisodeResult`. Produced by `record_episode`, consumed by the game-trace
    debugger; carries no `kaggle_environments` types so callers stay decoupled
    from the env."""

    frames: Sequence[Frame]
    result: EpisodeResult
    num_players: int
    seed: Optional[int]

    def __len__(self) -> int:
        return len(self.frames)


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


def compute_placements(
    scores: Sequence[int], faulted: Optional[Iterable[int]] = None
) -> List[int]:
    """Deterministic 1-based Placement per player index from final scores.

    Higher score ranks first. Ties are broken by lower player index, so every
    agent gets a distinct Placement (1st, 2nd, ...) — this keeps downstream
    placement aggregation unambiguous and reproducible.

    `faulted` is the set of player indices that raised or timed out; they are
    ranked below every non-faulted agent regardless of score (so a crashed
    agent that happened to keep ships still places last), ordered among
    themselves by the same score-then-index rule.
    """
    faulted_set = set(faulted or ())
    order = sorted(
        range(len(scores)), key=lambda i: (i in faulted_set, -scores[i], i)
    )
    placements = [0] * len(scores)
    for rank, i in enumerate(order):
        placements[i] = rank + 1
    return placements


# ---------------------------------------------------------------------------
# Fault isolation for callable agents.
# ---------------------------------------------------------------------------


class _GuardedAgent:
    """Wrap a callable agent so a raised exception or an over-budget turn is
    contained instead of aborting the episode.

    On a fault the turn yields no Shots (`[]`), the agent is flagged
    `faulted`, and every later turn short-circuits to `[]` — mirroring the
    env's own "once errored, stop calling" behaviour. The Arena reads
    `faulted` afterwards to force the agent to last Placement.

    Timeout uses a worker thread (a plain Python callable can't be preempted):
    if it doesn't finish within `timeout` seconds we abandon it (it's a daemon
    thread, so it can't keep the process alive) and move on. Because we stop
    calling a faulted agent, at most one such thread is ever spawned per agent.
    `timeout=None` disables the time budget but still catches exceptions.
    """

    def __init__(self, fn: Callable, timeout: Optional[float] = None) -> None:
        self._fn = fn
        self._timeout = timeout
        self.faulted = False

    def __call__(self, observation, configuration=None):
        if self.faulted:
            return []
        if self._timeout is None:
            try:
                return self._invoke(observation, configuration)
            except BaseException:
                self.faulted = True
                return []
        box: dict = {}

        def target():
            try:
                box["ok"] = self._invoke(observation, configuration)
            except BaseException as exc:  # noqa: BLE001 - contained on purpose
                box["err"] = exc

        worker = threading.Thread(target=target, daemon=True)
        worker.start()
        worker.join(self._timeout)
        if worker.is_alive() or "err" in box:
            self.faulted = True
            return []
        return box.get("ok", [])

    def _invoke(self, observation, configuration):
        # Match the env's arity handling: pass (obs, config) truncated to the
        # wrapped function's positional-arg count (some agents take obs only).
        fn = self._fn
        args = [observation, configuration]
        if hasattr(fn, "__code__") and hasattr(fn.__code__, "co_argcount"):
            args = args[: fn.__code__.co_argcount]
        return fn(*args)


def _guard(agent: Agent, timeout: Optional[float]) -> Optional[_GuardedAgent]:
    """Wrap a callable agent in a guard; return None for file-path / builtin
    agents (which run out-of-our-reach inside the env and are fault-detected
    from the env's per-step status instead)."""
    return _GuardedAgent(agent, timeout) if callable(agent) else None


# ---------------------------------------------------------------------------
# The Arena entry point.
# ---------------------------------------------------------------------------


def _run_env(agents: Sequence[Agent], config: Optional[EpisodeConfig]):
    """Validate, fault-guard the agents, and play one episode. Returns
    `(env, guards, num_players)` — the shared core of `run_episode` (result
    only) and `record_episode` (full trace). The only code that touches the
    Official env."""
    if config is None:
        config = EpisodeConfig()
    if config.num_players not in (2, 4):
        raise ValueError(
            f"num_players must be 2 or 4 (orbit_wars supports 1v1 and 4P FFA), "
            f"got {config.num_players}"
        )
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

    # Wrap callable agents in a fault guard; pass file/builtin agents through
    # unchanged (the env runs them and we read faults from its per-step status).
    # `act_timeout` is the Arena's own per-turn budget, NOT the env's actTimeout:
    # the env can't preempt an in-process callable (it only burns a 60s overage),
    # so we enforce the budget ourselves in the guard.
    guards = [_guard(a, config.act_timeout) for a in agents]
    runnable = [g if g is not None else a for g, a in zip(guards, agents)]

    env = make("orbit_wars", configuration=env_config, debug=False)
    env.run(runnable)
    return env, guards, config.num_players


def run_episode(
    agents: Sequence[Agent], config: Optional[EpisodeConfig] = None
) -> EpisodeResult:
    """Play one Orbit Wars episode and return a structured `EpisodeResult`.

    `agents` is a list of agents (callables, .py paths, or builtin names), one
    per player slot. `config` carries player count and seed.
    """
    env, guards, num_players = _run_env(agents, config)
    return _extract_result(env, num_players, guards)


def record_episode(
    agents: Sequence[Agent], config: Optional[EpisodeConfig] = None
) -> "EpisodeTrace":
    """Play one episode and return a full per-turn `EpisodeTrace` (plus the
    `EpisodeResult`). The trace is the Arena's episode *record*: the game-trace
    debugger (`src/replay.py`) renders it turn by turn without re-running any
    agent. Like `run_episode`, this is the only path that touches the env."""
    env, guards, num_players = _run_env(agents, config)
    result = _extract_result(env, num_players, guards)
    frames = tuple(
        Frame(
            step=t,
            planets=[list(p) for p in step_states[0].observation["planets"]],
            fleets=[list(f) for f in step_states[0].observation["fleets"]],
            actions=[step_states[i].action for i in range(num_players)],
        )
        for t, step_states in enumerate(env.steps)
    )
    return EpisodeTrace(
        frames=frames,
        result=result,
        num_players=num_players,
        seed=result.seed,
    )


# Env-reported statuses that mean an agent raised, timed out, or returned an
# illegal action — used to detect faults of file/builtin agents we can't guard.
_FAULT_STATUSES = frozenset({"ERROR", "TIMEOUT", "INVALID"})


def _faulted_indices(env, guards: Sequence[Optional[_GuardedAgent]]) -> Set[int]:
    """Player indices that faulted: guarded callables that flagged themselves,
    plus any agent the env marked ERROR/TIMEOUT/INVALID at any step (the final
    step is unreliable — the interpreter resets every status to DONE on
    termination, so we scan the whole episode)."""
    faulted: Set[int] = set()
    for i, g in enumerate(guards):
        if g is not None and g.faulted:
            faulted.add(i)
    for step in env.steps:
        for i, agent_state in enumerate(step):
            if agent_state.status in _FAULT_STATUSES:
                faulted.add(i)
    return faulted


def _extract_result(
    env, num_players: int, guards: Sequence[Optional[_GuardedAgent]]
) -> EpisodeResult:
    """Turn a finished env into an EpisodeResult — the result-extraction half of
    the encapsulated seam."""
    last = env.steps[-1]
    obs = last[0].observation  # full board is shared across all agent slots
    scores = score_board(obs["planets"], obs["fleets"], num_players)
    faulted = _faulted_indices(env, guards)
    placements = compute_placements(scores, faulted)

    outcomes = tuple(
        AgentOutcome(
            index=i,
            placement=placements[i],
            score=int(scores[i]),
            reward=last[i].reward,
            status=last[i].status,
            faulted=i in faulted,
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
# Demo: 2-player, 4-player, and a crashing-stub episode.
#   python -m src.arena
# ---------------------------------------------------------------------------


def _print_result(title: str, result: EpisodeResult, names: Sequence[str]) -> None:
    medals = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}
    print(f"\n{title} — seed {result.seed}, {result.num_steps} steps")
    for o in sorted(result.outcomes, key=lambda o: o.placement):
        flag = "  (faulted)" if o.faulted else ""
        print(
            f"  {medals[o.placement]}  {names[o.index]:<18} "
            f"score={o.score:>6}  reward={o.reward}{flag}"
        )
    print(f"  Winner: {names[result.winner]}")


def _demo() -> None:
    opp = Path(__file__).parent / "opponents"
    p = lambda name: str(opp / f"{name}.py")  # noqa: E731

    # 1) 2-player.
    names2 = ["weakest_first", "production_first"]
    r2 = run_episode(
        [p("weakest_first"), p("production_first")],
        EpisodeConfig(num_players=2, seed=2026),
    )
    _print_result("2-player (1v1)", r2, names2)

    # 2) 4-player FFA — same interface, num_players=4.
    names4 = ["weakest_first", "production_first", "nearest_sniper", "defender"]
    r4 = run_episode(
        [p(n) for n in names4], EpisodeConfig(num_players=4, seed=2026)
    )
    _print_result("4-player (FFA)", r4, names4)

    # 3) Crashing stub — the episode still completes and returns placements,
    #    with the crashing agent forced to last.
    def boom(obs, config=None):
        raise RuntimeError("agent crashed")

    namesC = ["weakest_first", "boom(crashes)"]
    rC = run_episode(
        [p("weakest_first"), boom], EpisodeConfig(num_players=2, seed=2026)
    )
    _print_result("crashing stub", rC, namesC)


if __name__ == "__main__":
    _demo()
