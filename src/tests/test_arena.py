"""Arena tests.

Two layers:
  * Pure tests for the placement/score helpers — no kaggle_environments, so they
    run anywhere (same spirit as the utils tests).
  * Integration-flavored tests that play real episodes through `run_episode`
    (2P + 4P, reproducibility, and fault tolerance); skipped if the Official env
    isn't installed.
"""

import time
from pathlib import Path

import pytest

from src.arena import (
    AgentOutcome,
    EpisodeConfig,
    EpisodeResult,
    compute_placements,
    score_board,
)


def _opp(name: str) -> str:
    return str(Path(__file__).resolve().parents[1] / "opponents" / f"{name}.py")


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


def test_compute_placements_forces_faulted_below_everyone():
    # idx 0 has the top score but faulted -> it must place last.
    assert compute_placements([10, 5, 8], faulted={0}) == [3, 2, 1]


def test_compute_placements_orders_multiple_faulted_among_themselves():
    # idx 0,1 faulted -> they take the bottom slots, ranked by score then index;
    # idx 2,3 (clean) take the top slots.
    assert compute_placements([1, 2, 3, 4], faulted={0, 1}) == [4, 3, 2, 1]


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


def _boom(obs, config=None):
    """An agent that always raises."""
    raise RuntimeError("agent crashed")


def _slow(obs, config=None):
    """An agent that always blows a small per-turn budget."""
    time.sleep(0.5)
    return []


def test_run_episode_rejects_unsupported_player_count():
    # Validation happens before the env is touched, so no env is needed.
    from src.arena import run_episode

    with pytest.raises(ValueError):
        run_episode([_do_nothing, _do_nothing, _do_nothing], EpisodeConfig(num_players=3))


def test_run_episode_four_player_ffa():
    pytest.importorskip("kaggle_environments")
    from src.arena import run_episode

    agents = [_opp(n) for n in
              ("weakest_first", "production_first", "nearest_sniper", "defender")]
    result = run_episode(agents, EpisodeConfig(num_players=4, seed=11, episode_steps=120))

    assert result.num_players == 4
    # Placements are a valid permutation of 1..4 via the same interface as 2P.
    assert sorted(o.placement for o in result.outcomes) == [1, 2, 3, 4]
    assert result.placement_of(result.winner) == 1
    assert all(o.score >= 0 for o in result.outcomes)


def test_run_episode_is_reproducible():
    pytest.importorskip("kaggle_environments")
    from src.arena import run_episode

    agents = [_opp(n) for n in
              ("weakest_first", "production_first", "nearest_sniper", "defender")]
    cfg = EpisodeConfig(num_players=4, seed=99, episode_steps=120)

    def signature(r):
        return [(o.index, o.placement, o.score, o.reward, o.faulted) for o in r.outcomes]

    r1 = run_episode(agents, cfg)
    r2 = run_episode(agents, cfg)
    assert signature(r1) == signature(r2)
    assert r1.num_steps == r2.num_steps


def test_run_episode_crashing_agent_placed_last():
    pytest.importorskip("kaggle_environments")
    from src.arena import run_episode

    # slot 1 raises every turn; the episode must still complete and return a
    # result, with the crashing agent forced to last Placement.
    result = run_episode([_opp("weakest_first"), _boom], EpisodeConfig(num_players=2, seed=7))

    assert isinstance(result, EpisodeResult)
    assert result.outcomes[1].faulted is True
    assert result.placement_of(1) == 2  # last in a 2-player game
    assert result.outcomes[0].faulted is False
    assert result.placement_of(0) == 1
    assert result.winner == 0


def test_run_episode_timeout_agent_placed_last():
    pytest.importorskip("kaggle_environments")
    from src.arena import run_episode

    # slot 1 sleeps past the Arena's per-turn budget. It is flagged faulted and
    # forced last even though sitting idle may leave it more ships than the
    # winner. Marking it dead after the first timeout keeps the game fast.
    t0 = time.time()
    result = run_episode(
        [_opp("weakest_first"), _slow],
        EpisodeConfig(num_players=2, seed=7, episode_steps=40, act_timeout=0.15),
    )
    assert result.outcomes[1].faulted is True
    assert result.placement_of(1) == 2
    assert result.placement_of(0) == 1
    # One ~0.15s timeout, then the agent is skipped — nowhere near 40 * 0.5s.
    assert time.time() - t0 < 10
