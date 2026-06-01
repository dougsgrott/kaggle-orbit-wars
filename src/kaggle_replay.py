"""
Load a downloaded Kaggle Orbit Wars replay into our own `EpisodeTrace`, so the
existing game-trace debugger (`src/replay.py`) renders it turn by turn — board,
fleet death-cause overlays, and missed-Shot markers — with no re-running of any
agent.

A Kaggle submission's replay JSON (e.g. `submissions/submission_01/<id>.json`)
carries the full per-step record. Each `steps[t]` is a list of per-agent dicts;
the shared board lives in slot 0's `observation` (`planets` / `fleets`), and each
slot's `action` is the moves it submitted that turn. That is exactly the shape of
our `arena.Frame`, so this is a thin field-mapping adapter onto the same
`EpisodeTrace` that `arena.record_episode` produces — the replay viewer doesn't
care which one it gets.

CLI:
    python -m src.kaggle_replay submissions/submission_01/78432179.json
        -> writes <json_dir>/replay.gif and prints a fleet-death summary.

Public API:
    load_kaggle_trace(path)                 -> EpisodeTrace
    summarize(path)                         -> dict   (scores, placements, death tally)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Union

from .arena import (
    AgentOutcome,
    EpisodeResult,
    EpisodeTrace,
    Frame,
    compute_placements,
    score_board,
)

# Env statuses that mean an agent raised / timed out / sent an illegal move.
_FAULT_STATUSES = frozenset({"ERROR", "TIMEOUT", "INVALID"})


def _load_json(path: Union[str, Path]) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _obs_of(step) -> dict:
    """The shared board for a recorded step lives in agent slot 0's observation."""
    return step[0]["observation"]


def load_kaggle_trace(path: Union[str, Path]) -> EpisodeTrace:
    """Parse a Kaggle replay JSON into an `EpisodeTrace` (frames + result).

    Placement is derived from the final board exactly like `arena._extract_result`
    (ships on owned planets + ships in owned fleets, ranked high-first, ties by
    lower index, faulted agents forced last) so a downloaded replay and a locally
    run one are scored identically.
    """
    data = _load_json(path)
    steps = data["steps"]
    num_players = len(steps[0])

    frames = tuple(
        Frame(
            step=t,
            planets=[list(p) for p in _obs_of(step)["planets"]],
            fleets=[list(f) for f in _obs_of(step)["fleets"]],
            actions=[step[i].get("action") for i in range(num_players)],
        )
        for t, step in enumerate(steps)
    )

    # Faults: any agent the env marked ERROR/TIMEOUT/INVALID at any step (the
    # final step is unreliable — terminal resets statuses to DONE).
    faulted = {
        i
        for step in steps
        for i in range(num_players)
        if step[i].get("status") in _FAULT_STATUSES
    }

    final_obs = _obs_of(steps[-1])
    scores = score_board(final_obs["planets"], final_obs["fleets"], num_players)
    placements = compute_placements(scores, faulted)
    rewards = data.get("rewards") or [None] * num_players
    last = steps[-1]

    outcomes = tuple(
        AgentOutcome(
            index=i,
            placement=placements[i],
            score=int(scores[i]),
            reward=rewards[i] if i < len(rewards) else None,
            status=last[i].get("status", "DONE"),
            faulted=i in faulted,
        )
        for i in range(num_players)
    )

    seed = (data.get("info") or {}).get("seed")
    result = EpisodeResult(
        outcomes=outcomes,
        num_players=num_players,
        seed=seed,
        num_steps=len(steps),
    )
    return EpisodeTrace(
        frames=frames, result=result, num_players=num_players, seed=seed
    )


def summarize(path: Union[str, Path]) -> dict:
    """Per-agent final scores/placements/rewards + an episode-wide fleet-death
    tally (combat / out-of-bounds / sun / missed), without rendering anything."""
    from .replay import death_summary

    trace = load_kaggle_trace(path)
    return {
        "num_players": trace.num_players,
        "num_steps": len(trace),
        "seed": trace.seed,
        "outcomes": [
            {
                "index": o.index,
                "placement": o.placement,
                "score": o.score,
                "reward": o.reward,
                "faulted": o.faulted,
            }
            for o in trace.result.outcomes
        ],
        "deaths": death_summary(trace),
    }


# ---------------------------------------------------------------------------
# CLI: render a downloaded replay through the existing viewer.
#   python -m src.kaggle_replay submissions/submission_01/<id>.json [out.gif] [--turns A B]
# ---------------------------------------------------------------------------


def _main(argv: Optional[Sequence[str]] = None) -> None:
    import argparse

    from .replay import render_replay

    ap = argparse.ArgumentParser(description="Render a downloaded Kaggle replay.")
    ap.add_argument("replay", help="path to the submission replay JSON")
    ap.add_argument(
        "out", nargs="?", default=None,
        help="output GIF (default: <replay_dir>/replay.gif); a dir/.png base "
        "writes one PNG per turn",
    )
    ap.add_argument(
        "--turns", nargs=2, type=int, metavar=("START", "END"),
        help="render only turns [START, END) (end exclusive)",
    )
    ap.add_argument("--fps", type=int, default=6)
    args = ap.parse_args(argv)

    replay_path = Path(args.replay)
    out = Path(args.out) if args.out else replay_path.parent / "replay.gif"

    info = summarize(replay_path)
    print(f"{replay_path.name}: {info['num_players']}P, {info['num_steps']} steps, "
          f"seed={info['seed']}")
    for o in sorted(info["outcomes"], key=lambda o: o["placement"]):
        flag = " (faulted)" if o["faulted"] else ""
        print(f"  #{o['placement']}  agent {o['index']}  score={o['score']:>6}  "
              f"reward={o['reward']}{flag}")
    d = info["deaths"]
    print(f"  fleet deaths: combat={d['combat']} sun={d['sun']} "
          f"out_of_bounds={d['out_of_bounds']}  -> missed={d['missed']}")

    trace = load_kaggle_trace(replay_path)
    turns = range(*args.turns) if args.turns else None
    written = render_replay(trace, out, turns=turns, fps=args.fps)
    print("wrote", ", ".join(str(p) for p in written))


if __name__ == "__main__":
    _main()
