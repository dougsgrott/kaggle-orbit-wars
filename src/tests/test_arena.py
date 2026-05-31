"""Arena tests.

Two layers:
  * Pure tests for the placement/score helpers — no kaggle_environments, so they
    run anywhere (same spirit as the utils tests).
  * One integration-flavored test that plays a real 2-player episode through
    `run_episode`; it is skipped if the Official env isn't installed.
"""

import pytest

from src.arena import (
    AgentOutcome,
    EpisodeConfig,
    EpisodeResult,
    compute_placements,
    score_board,
)


# --- pure: score_board --------------------------------------------------------


def test_score_board_sums_planets_and_fleets():
    # planets: [id, owner, x, y, radius, ships, production]
    planets = [
        [0, 0, 0, 0, 1, 10, 1],
        [1, 1, 0, 0, 1, 4, 1],
        [2, -1, 0, 0, 1, 99, 1],  # neutral — ignored
        [3, 0, 0, 0, 1, 5, 1],
    ]
    # fleets: [id, owner, x, y, angle, from_planet_id, ships]
    fleets = [
        [0, 0, 0, 0, 0.0, 0, 3],
        [1, 1, 0, 0, 0.0, 1, 7],
    ]
    assert score_board(planets, fleets, 2) == [18, 11]


def test_score_board_ignores_out_of_range_owners():
    planets = [[0, 5, 0, 0, 1, 50, 1]]  # owner beyond num_players
    assert score_board(planets, [], 2) == [0, 0]


# --- pure: compute_placements -------------------------------------------------


def test_compute_placements_orders_by_score_desc():
    assert compute_placements([10, 30, 20]) == [3, 1, 2]


def test_compute_placements_breaks_ties_by_lower_index():
    # equal scores -> lower index gets the better (smaller) placement
    assert compute_placements([5, 5, 5]) == [1, 2, 3]
    assert compute_placements([7, 9, 9]) == [3, 1, 2]


def test_compute_placements_is_a_permutation():
    placements = compute_placements([3, 3, 1, 8])
    assert sorted(placements) == [1, 2, 3, 4]


# --- integration: a real episode through the seam -----------------------------


def _do_nothing(obs, config=None):
    """A legal agent that never launches a fleet."""
    return []


def test_run_episode_two_player_known_outcome():
    pytest.importorskip("kaggle_environments")
    from pathlib import Path

    from src.arena import run_episode

    opp = Path(__file__).resolve().parents[1] / "opponents" / "weakest_first.py"
    result = run_episode([str(opp), _do_nothing], EpisodeConfig(num_players=2, seed=7))

    assert isinstance(result, EpisodeResult)
    assert result.num_players == 2
    assert result.seed == 7
    assert all(isinstance(o, AgentOutcome) for o in result.outcomes)

    # Placements are a valid permutation of 1..2.
    assert sorted(o.placement for o in result.outcomes) == [1, 2]
    # The active expander must beat an agent that never moves.
    assert result.placement_of(0) == 1
    assert result.placement_of(1) == 2
    assert result.winner == 0
    assert result.ranking == [0, 1]
    # do-nothing captures nothing and launches nothing -> zero final ships.
    assert result.score_of(0) > result.score_of(1)
    assert result.score_of(1) == 0
