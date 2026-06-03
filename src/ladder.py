"""
Ladder evaluation harness + build ledger (E2).

Drives the **Arena** over the tiered **Ladder** (`opponents.LADDER`) and produces
trustworthy, non-saturating signal: each opponent is played across many seeds in
**both formats** — 1v1 both sides, and 4P FFA with the agent rotated through
seats — results are aggregated by **Placement** (via `rating`, E1) with a Wilson
interval, and every run is appended to a persistent ledger CSV keyed by build id
+ timestamp. Runs games in parallel so a full sweep finishes in minutes.

This supersedes `eval.evaluate_agent` (1v1, reward-based) for measuring our own
brains; that function stays for backwards compatibility. The ledger here is a
*separate* file by default (`ladder_log.csv`) so the existing 1v1 `eval_log.csv`
history is preserved untouched — the two schemas differ.

The agent under test and every opponent are whatever `arena.run_episode` accepts
(a brain wrapped as a callable, a `.py` path, or a builtin name like
"random"/"starter").

Public API:
    evaluate_ladder(agent, ladder=opponents.LADDER, n_seeds=8, n_workers=4,
                    build_id=..., csv_path="ladder_log.csv") -> dict
    OPPONENT_RATINGS    -- fixed reference Elo per ladder tier (for the proxy)
"""
from __future__ import annotations

import csv
import multiprocessing as mp
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from . import opponents as _opp
from .arena import EpisodeConfig, run_episode
from .rating import (
    DEFAULT_RATING,
    aggregate_outcomes,
    expected_score,
    update_elo,
)

Agent = Union[str, Callable]

_SEED_BASE = 3000

# Fixed reference ratings per tier — anchors for the single-number Elo proxy.
# Spread so that beating the Boss tier moves the rating meaningfully more than
# beating the floor. These are yardstick anchors, not learned values.
OPPONENT_RATINGS: Dict[str, float] = {
    "floor": 200.0,
    "panel": 500.0,
    "official": 700.0,
    "boss": 1000.0,
    "snapshots": 800.0,
}


def _tier_of(ladder: dict, opponent: Agent) -> str:
    for tier, members in ladder.items():
        if opponent in members:
            return tier
    return "panel"


# ---------------------------------------------------------------------------
# Parallel worker: play ONE game and return the agent's normalized outcome.
# Module-level + picklable so it works with mp.Pool. The agent and opponent must
# both be picklable across the pool — file paths / builtin names always are; a
# brain is passed by registry *name* (resolved in the worker) for the same reason.
# ---------------------------------------------------------------------------


def _resolve(spec):
    """A job spec is ('brain', name) -> resolve via the agents registry in the
    worker; or ('agent', path_or_builtin) -> use as-is for run_episode."""
    kind, value = spec
    if kind == "brain":
        from .agents import REGISTRY

        fn = REGISTRY[value]

        def _agent(obs, config=None):
            return fn(obs, config)

        return _agent
    return value


def _play_job(job):
    """One game. `job` = (agent_spec, opp_spec, opp_name, tier, fmt, seed, seat,
    num_players). Returns (opp_name, tier, fmt, placement, num_players) or None
    if the agent faulted (excluded from aggregation, surfaced in the summary)."""
    agent_spec, opp_spec, opp_name, tier, fmt, seed, seat, num_players = job
    agent = _resolve(agent_spec)
    opp = _resolve(opp_spec)
    # Seat the agent at `seat`; fill the rest with the opponent.
    line = [opp] * num_players
    line[seat] = agent
    result = run_episode(line, EpisodeConfig(num_players=num_players, seed=seed))
    o = result.outcomes[seat]
    if o.faulted:
        return (opp_name, tier, fmt, None, num_players)
    return (opp_name, tier, fmt, o.placement, num_players)


def _build_jobs(agent_spec, ladder, n_seeds):
    """Every (opponent × format × seed × seat) game for the sweep."""
    jobs = []
    for tier, members in ladder.items():
        for opp in members:
            opp_name = opp if isinstance(opp, str) and "/" not in opp else Path(str(opp)).stem
            opp_spec = ("agent", opp)
            for s in range(n_seeds):
                seed = _SEED_BASE + s
                # 1v1, both sides.
                for seat in (0, 1):
                    jobs.append(
                        (agent_spec, opp_spec, opp_name, tier, "1v1", seed, seat, 2)
                    )
                # 4P FFA: rotate the agent's seat across seeds for position balance.
                jobs.append(
                    (agent_spec, opp_spec, opp_name, tier, "4p", seed, s % 4, 4)
                )
    return jobs


def evaluate_ladder(
    agent: Union[str, Tuple[str, str]],
    ladder: Optional[dict] = None,
    n_seeds: int = 8,
    n_workers: int = 4,
    build_id: str = "build",
    csv_path: Optional[str] = "ladder_log.csv",
    verbose: bool = True,
) -> dict:
    """Sweep `agent` over the `ladder` in both formats; aggregate + log.

    `agent` is a registry brain name (str) — resolved per worker — or an explicit
    `("agent", path_or_builtin)` spec. Returns a summary dict and, unless
    `csv_path` is None, appends one row per (opponent, format) plus an overall
    Elo row to the ledger.
    """
    if ladder is None:
        ladder = _opp.LADDER
    agent_spec = ("brain", agent) if isinstance(agent, str) else agent

    jobs = _build_jobs(agent_spec, ladder, n_seeds)
    if verbose:
        print(f"\n=== LADDER EVAL '{build_id}' ===")
        print(f"  {len(jobs)} games over {n_seeds} seeds "
              f"({sum(len(m) for m in ladder.values())} opponents × 1v1+4p)")

    t0 = time.time()
    # (opponent, format) -> list of (placement, num_players); faults tallied apart.
    buckets: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
    tier_of_opp: Dict[str, str] = {}
    faults = 0
    with mp.Pool(processes=n_workers) as pool:
        for opp_name, tier, fmt, placement, num_players in pool.imap_unordered(
            _play_job, jobs
        ):
            tier_of_opp[opp_name] = tier
            if placement is None:
                faults += 1
                continue
            buckets.setdefault((opp_name, fmt), []).append((placement, num_players))
    elapsed = time.time() - t0

    ts = datetime.now().isoformat(timespec="seconds")
    rows: List[dict] = []
    rating = DEFAULT_RATING
    # Process weakest -> strongest tier so the Elo walk is stable/readable.
    tier_order = list(ladder.keys())
    keys = sorted(
        buckets.keys(),
        key=lambda k: (tier_order.index(tier_of_opp[k[0]]), k[0], k[1]),
    )
    for opp_name, fmt in keys:
        agg = aggregate_outcomes(buckets[(opp_name, fmt)])
        tier = tier_of_opp[opp_name]
        # Elo proxy: nudge our rating per (opponent, format) batch toward the
        # mean placement score vs that tier's reference rating.
        exp = expected_score(rating, OPPONENT_RATINGS.get(tier, 500.0))
        rating = update_elo(rating, exp, agg["mean_score"])
        rows.append({
            "build_id": build_id, "timestamp": ts, "tier": tier,
            "opponent": opp_name, "format": fmt, "n": agg["n"],
            "first_rate": round(agg["first_rate"], 4),
            "ci_lo": round(agg["ci_lo"], 4), "ci_hi": round(agg["ci_hi"], 4),
            "mean_placement": round(agg["mean_placement"], 3),
            "mean_score": round(agg["mean_score"], 4),
            "elo_after": round(rating, 1),
        })
        if verbose:
            print(f"  [{tier:<8}] {opp_name:<16} {fmt:<3} : "
                  f"1st={agg['first_rate'] * 100:5.1f}% "
                  f"[{agg['ci_lo'] * 100:4.1f}–{agg['ci_hi'] * 100:4.1f}] "
                  f"meanPlc={agg['mean_placement']:.2f} (n={agg['n']})")

    if verbose:
        print(f"  {'─' * 64}")
        print(f"  Elo proxy: {rating:.0f}   faults={faults}   (in {elapsed:.0f}s)")

    if csv_path is not None and rows:
        path = Path(csv_path)
        write_header = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if write_header:
                w.writeheader()
            for row in rows:
                w.writerow(row)
        if verbose:
            print(f"  ledger updated: {path}")

    return {
        "build_id": build_id,
        "elo": rating,
        "faults": faults,
        "rows": rows,
        "elapsed_s": elapsed,
    }
