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
import math
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
    "strong": 1200.0,   # L2 ported agents (catalogue 727-816), above the Boss
    "sota": 1500.0,     # The Producer benchmark (beats the Boss 6-0), top of the field
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


# ---------------------------------------------------------------------------
# Paired head-to-head A/B vs a single opponent (e.g. the Boss). Two brains face
# the SAME (seed, side) boards, so we get both independent Wilson CIs AND a
# paired (McNemar-style) comparison — far more sensitive for close rates.
# ---------------------------------------------------------------------------


def _sign_test_p(wins: int, n: int) -> float:
    """Two-sided exact binomial sign test p-value for `wins` successes in `n`
    discordant trials under H0 p=0.5. n=0 -> 1.0 (no evidence)."""
    if n <= 0:
        return 1.0
    k = max(wins, n - wins)
    tail = sum(math.comb(n, i) for i in range(k, n + 1))
    return min(1.0, 2.0 * tail * (0.5 ** n))


def ab_compare(
    brain_a: str,
    brain_b: str,
    opponent: Agent,
    n_seeds: int = 30,
    n_workers: int = 8,
    csv_path: Optional[str] = "boss_ab_log.csv",
    verbose: bool = True,
) -> dict:
    """Paired 1v1 A/B of two registry brains vs `opponent`, both sides.

    Both brains play the identical (seed, side) board set, so results pair up:
    per board we record whether each brain placed 1st, then report each brain's
    1st-rate + Wilson CI (independent) and the paired split (a-only / b-only /
    both / neither) with a two-sided sign-test p on the discordant pairs.
    """
    opp_name = (opponent if isinstance(opponent, str) and "/" not in opponent
                else Path(str(opponent)).stem)
    opp_spec = ("agent", opponent)

    keys: List[tuple] = []
    jobs: List[tuple] = []
    for s in range(n_seeds):
        seed = _SEED_BASE + s
        for side in (0, 1):
            for tag, brain in (("a", brain_a), ("b", brain_b)):
                keys.append((s, side, tag))
                jobs.append(
                    (("brain", brain), opp_spec, opp_name, "boss", "1v1", seed, side, 2)
                )

    if verbose:
        print(f"\n=== A/B vs {opp_name}: '{brain_a}' (a) vs '{brain_b}' (b) ===")
        print(f"  {len(jobs)} games ({n_seeds} seeds × 2 sides × 2 brains, paired)")

    t0 = time.time()
    results: Dict[tuple, tuple] = {}
    with mp.Pool(processes=n_workers) as pool:
        for key, res in zip(keys, pool.imap(_play_job, jobs)):  # imap = ordered
            results[key] = res  # res = (opp, tier, fmt, placement, num_players)
    elapsed = time.time() - t0

    a_out: List[Tuple[int, int]] = []
    b_out: List[Tuple[int, int]] = []
    a_only = b_only = both = neither = faults = 0
    for s in range(n_seeds):
        for side in (0, 1):
            ra, rb = results[(s, side, "a")], results[(s, side, "b")]
            pa, pb = ra[3], rb[3]
            if pa is not None:
                a_out.append((pa, ra[4]))
            if pb is not None:
                b_out.append((pb, rb[4]))
            if pa is None or pb is None:
                faults += 1
                continue
            aw, bw = (pa == 1), (pb == 1)
            if aw and not bw:
                a_only += 1
            elif bw and not aw:
                b_only += 1
            elif aw and bw:
                both += 1
            else:
                neither += 1

    agg_a = aggregate_outcomes(a_out)
    agg_b = aggregate_outcomes(b_out)
    discordant = a_only + b_only
    sign_p = _sign_test_p(a_only, discordant)

    summary = {
        "opponent": opp_name,
        "brain_a": brain_a, "brain_b": brain_b,
        "a": agg_a, "b": agg_b,
        "paired": {"a_only": a_only, "b_only": b_only, "both": both,
                   "neither": neither, "discordant": discordant, "sign_p": sign_p},
        "faults": faults, "elapsed_s": elapsed,
    }

    if verbose:
        print(f"  [a] {brain_a:<18} 1st {agg_a['first_rate']*100:5.1f}% "
              f"[{agg_a['ci_lo']*100:4.1f}–{agg_a['ci_hi']*100:4.1f}] (n={agg_a['n']})")
        print(f"  [b] {brain_b:<18} 1st {agg_b['first_rate']*100:5.1f}% "
              f"[{agg_b['ci_lo']*100:4.1f}–{agg_b['ci_hi']*100:4.1f}] (n={agg_b['n']})")
        print(f"  paired: a-only={a_only} b-only={b_only} both={both} neither={neither} "
              f"| sign p={sign_p:.3f}  faults={faults}  (in {elapsed:.0f}s)")

    if csv_path is not None:
        ts = datetime.now().isoformat(timespec="seconds")
        path = Path(csv_path)
        write_header = not path.exists()
        fields = ["timestamp", "opponent", "brain", "n", "first_rate", "ci_lo",
                  "ci_hi", "a_only", "b_only", "both", "neither", "sign_p"]
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                w.writeheader()
            for tag, brain, agg in (("a", brain_a, agg_a), ("b", brain_b, agg_b)):
                w.writerow({
                    "timestamp": ts, "opponent": opp_name, "brain": brain,
                    "n": agg["n"], "first_rate": round(agg["first_rate"], 4),
                    "ci_lo": round(agg["ci_lo"], 4), "ci_hi": round(agg["ci_hi"], 4),
                    "a_only": a_only, "b_only": b_only, "both": both,
                    "neither": neither, "sign_p": round(sign_p, 4),
                })
        if verbose:
            print(f"  ledger updated: {path}")

    return summary
