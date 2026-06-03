"""
Rating & aggregation (E1) — pure, env-free signal from episode results.

Turns a batch of `arena.EpisodeResult`s into measurable per-opponent signal
(placement aggregation + Wilson interval) and maintains a simple Elo as a single
**Skill rating** proxy. Everything here is a pure function of its inputs — no
`kaggle_environments`, no I/O, no global state — so it is unit-tested on synthetic
results and reused by the ladder harness (E2).

Placement is the unit (binary win/loss in 1v1, 1st–4th in 4P FFA), matching how
the leaderboard scores; we normalise it to a [0,1] **placement score** so 1v1 and
4P feed the same Elo update:
    score = (num_players - placement) / (num_players - 1)
i.e. 1st -> 1.0, last -> 0.0, evenly spaced between (1v1: win 1.0 / loss 0.0).

Public API:
    placement_score(placement, num_players)              -> float
    aggregate_outcomes(outcomes)                         -> dict
    aggregate_placements(results, agent_index)           -> dict
    expected_score(rating_a, rating_b)                   -> float
    update_elo(rating, expected, actual, k=24.0)         -> float
"""
from __future__ import annotations

from typing import Sequence, Tuple

from .eval import wilson_ci

DEFAULT_RATING = 600.0  # mirrors the Kaggle default starting skill rating.
DEFAULT_K = 24.0


def placement_score(placement: int, num_players: int) -> float:
    """Normalise a 1-based placement to [0,1] (1st -> 1.0, last -> 0.0).

    Single-player or degenerate fields score 1.0 (nothing to lose to)."""
    if num_players <= 1:
        return 1.0
    return (num_players - placement) / (num_players - 1)


def aggregate_outcomes(outcomes: Sequence[Tuple[int, int]]) -> dict:
    """Summarise a list of `(placement, num_players)` outcomes for one agent.

    This is the aggregation primitive — it does not care which seat the agent
    played, only how it placed and in how large a field — so it works for 1v1
    and 4P alike, and for results passed back from parallel workers as plain
    tuples. Returns:
        n              -- episodes counted
        firsts         -- count of 1st-place finishes
        first_rate     -- fraction of 1st places (the win-rate analogue)
        ci_lo, ci_hi   -- Wilson 95% interval on first_rate
        mean_placement -- average 1-based placement (lower is better)
        mean_score     -- average normalised placement score in [0,1]
    """
    n = len(outcomes)
    if n == 0:
        return {
            "n": 0, "firsts": 0, "first_rate": 0.0, "ci_lo": 0.0, "ci_hi": 0.0,
            "mean_placement": 0.0, "mean_score": 0.0,
        }
    firsts = sum(1 for p, _ in outcomes if p == 1)
    placement_sum = sum(p for p, _ in outcomes)
    score_sum = sum(placement_score(p, np) for p, np in outcomes)
    rate, lo, hi = wilson_ci(firsts, n)
    return {
        "n": n,
        "firsts": firsts,
        "first_rate": rate,
        "ci_lo": lo,
        "ci_hi": hi,
        "mean_placement": placement_sum / n,
        "mean_score": score_sum / n,
    }


def aggregate_placements(results: Sequence, agent_index: int) -> dict:
    """Summarise `agent_index`'s outcomes across a list of episode `results`.

    Convenience over `aggregate_outcomes` for when you hold full results: each
    result is anything exposing `.num_players` and `.placement_of(i)` (an
    `arena.EpisodeResult` or a stub). See `aggregate_outcomes` for the returned
    fields.
    """
    return aggregate_outcomes(
        [(r.placement_of(agent_index), r.num_players) for r in results]
    )


def expected_score(rating_a: float, rating_b: float) -> float:
    """Standard Elo expectation: A's expected score vs B in [0,1]."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_elo(
    rating: float, expected: float, actual: float, k: float = DEFAULT_K
) -> float:
    """Standard Elo update: nudge `rating` toward outcomes that beat expectation.

    `actual` is a [0,1] score (e.g. `placement_score`); `expected` from
    `expected_score`. Moves up when actual > expected, down when actual <
    expected, by `k * (actual - expected)`.
    """
    return rating + k * (actual - expected)
