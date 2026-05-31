"""
Evaluation harness — run a 1v1 game, sweep opponents, report Wilson-CI win-rates.

Adapted from clean_scripts/orbit_wars_validation_ml_robuste_harnais_eval.py
(catalogue score 835.6).

Public API:
    run_match(agent_path, opponent_path, seed=0, side=0)  -> int  (1/0/-1)
    wilson_ci(wins, n, z=1.96)                            -> (p, lo, hi)
    evaluate_agent(agent_path, opponent_paths,
                   opponent_names=None, n_seeds=8,
                   n_workers=4, verbose=True, csv_path=None) -> dict
"""

from __future__ import annotations

import csv
import math
import multiprocessing as mp
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence


def run_match(agent_path: str, opponent_path: str,
              seed: int = 0, side: int = 0) -> int:
    """Play one game. Returns 1 if agent_path wins, 0 if it loses, -1 on
    draw/error. `side` chooses which slot the agent plays (0 or 1)."""
    from kaggle_environments import make
    paths = [agent_path, opponent_path] if side == 0 else [opponent_path, agent_path]
    env = make("orbit_wars", configuration={"randomSeed": seed}, debug=False)
    try:
        env.run(paths)
        final = env.steps[-1]
        my_r = final[side].reward
        opp_r = final[1 - side].reward
        if my_r is None or opp_r is None:
            return -1
        if my_r > opp_r:
            return 1
        if my_r < opp_r:
            return 0
        return -1
    except Exception:
        return -1


def wilson_ci(wins: int, n: int, z: float = 1.96):
    """Wilson 95% score interval — preferred over normal approximation for n<100."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def evaluate_agent(agent_path: str,
                   opponent_paths: Sequence[str],
                   opponent_names: Optional[Sequence[str]] = None,
                   n_seeds: int = 8,
                   n_workers: int = 4,
                   label: str = "build",
                   verbose: bool = True,
                   csv_path: Optional[str] = None,
                   build_id: Optional[str] = None) -> dict:
    """Multi-opponent win-rate eval. Each opponent is played n_seeds*2 games
    (both sides). Returns a dict with per-opponent and overall WR + CI; if
    csv_path is given, appends one row per opponent.
    """
    if opponent_names is None:
        opponent_names = [Path(p).stem for p in opponent_paths]
    if build_id is None:
        build_id = label

    jobs = []
    for opp_path, opp_name in zip(opponent_paths, opponent_names):
        for seed in range(2000, 2000 + n_seeds):
            for side in (0, 1):
                jobs.append((agent_path, opp_path, seed, side, opp_name))

    if verbose:
        print(f"\n=== EVAL '{label}' ===")
        print(f"  {len(opponent_paths)} opponents × {n_seeds} seeds × 2 sides = {len(jobs)} games")

    t0 = time.time()
    results_by_opp = {name: [] for name in opponent_names}
    with mp.Pool(processes=n_workers) as pool:
        ready_jobs = [(j[0], j[1], j[2], j[3]) for j in jobs]
        for i, r in enumerate(pool.imap(_run_match_args, ready_jobs)):
            opp_name = jobs[i][4]
            results_by_opp[opp_name].append(r)
    elapsed = time.time() - t0

    rows = []
    overall_wins, overall_n = 0, 0
    for opp_name in opponent_names:
        results = results_by_opp[opp_name]
        wins   = sum(1 for r in results if r == 1)
        losses = sum(1 for r in results if r == 0)
        n = wins + losses
        wr, lo, hi = wilson_ci(wins, n)
        overall_wins += wins
        overall_n    += n
        rows.append({
            "build_id": build_id, "opponent": opp_name,
            "wins": wins, "losses": losses, "n": n,
            "wr": wr, "ci_lo": lo, "ci_hi": hi,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })
        if verbose:
            print(f"  vs {opp_name:<22} : {wr * 100:5.1f}%  [{lo * 100:4.1f}–{hi * 100:4.1f}]  (n={n})")

    overall_wr, overall_lo, overall_hi = wilson_ci(overall_wins, overall_n)
    if verbose:
        print(f"  {'─' * 60}")
        print(f"  OVERALL                 : {overall_wr * 100:5.1f}%  "
              f"[{overall_lo * 100:4.1f}–{overall_hi * 100:4.1f}]  (n={overall_n})")
        print(f"  (in {elapsed:.0f}s)")

    if csv_path is not None:
        path = Path(csv_path)
        write_header = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if write_header:
                w.writeheader()
            for row in rows:
                w.writerow(row)
        if verbose:
            print(f"  CSV updated: {path}")

    return {
        "build_id": build_id,
        "overall_wr": overall_wr,
        "overall_ci": (overall_lo, overall_hi),
        "by_opp": {r["opponent"]: r for r in rows},
        "elapsed_s": elapsed,
    }


def _run_match_args(args):
    return run_match(*args)
