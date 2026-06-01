"""Pure tests for the v0 roi_greedy brain + defense-reserve (no kaggle_environments).

Same style as the utils tests: synthetic observations, behavioral assertions.
"""

import math

from src.agents.roi_greedy import plan_turn, defense_reserve, SHIP_BUFFER
from src.agents import REGISTRY, DEFAULT


# planet row: [id, owner, x, y, radius, ships, production]   (owner -1 = neutral)
# fleet  row: [id, owner, x, y, angle, from_planet_id, ships]
def _obs(player, planets, fleets=()):
    return {
        "player": player,
        "planets": [list(p) for p in planets],
        "fleets": [list(f) for f in fleets],
    }


# --- defense_reserve: sized via combat resolution -----------------------------


def test_reserve_zero_without_threat():
    assert defense_reserve(0, []) == 0


def test_reserve_single_wave():
    # Owner-0 planet vs a 20-ship enemy fleet: needs 20 to hold (tie at remaining=0).
    assert defense_reserve(0, [(1, 20)]) == 20
    assert defense_reserve(0, [(1, 30)]) == 30


def test_reserve_non_decreasing_in_threat():
    assert defense_reserve(0, [(1, 20)]) <= defense_reserve(0, [(1, 30)])


def test_reserve_multi_enemy_net_survivor():
    # Two enemy owners co-arrive: 30 vs 10 leaves a net 20-ship attacker.
    assert defense_reserve(0, [(1, 30), (2, 10)]) == 20


def test_reserve_ffa_tie_neutralises():
    # Equal top attackers annihilate each other -> no reserve needed.
    assert defense_reserve(0, [(1, 20), (2, 20)]) == 0


# --- plan_turn ----------------------------------------------------------------


def _assert_legal(obs, moves):
    """Every Shot launches from a planet I own, with 1..garrison ships."""
    me = obs["player"]
    by_id = {int(p[0]): p for p in obs["planets"]}
    for sid, angle, ships in moves:
        assert int(by_id[sid][1]) == me
        assert 1 <= int(ships) <= int(by_id[sid][5])
        assert math.isfinite(angle)


def test_never_shoots_across_the_sun():
    # Only target sits directly across the sun (10,50)->(90,50): no safe shot.
    obs = _obs(0, [
        [0, 0, 10.0, 50.0, 1.0, 50, 1],
        [1, -1, 90.0, 50.0, 1.0, 5, 1],
    ])
    assert plan_turn(obs) == []


def test_no_shots_when_unaffordable():
    # A lone ship can't capture a 5-ship planet -> no launch.
    obs = _obs(0, [
        [0, 0, 10.0, 10.0, 1.0, 1, 1],
        [1, -1, 40.0, 10.0, 1.0, 5, 1],
    ])
    assert plan_turn(obs) == []


def test_no_shots_when_no_targets():
    obs = _obs(0, [[0, 0, 10.0, 10.0, 1.0, 50, 1]])
    assert plan_turn(obs) == []


def test_prefers_higher_roi_target():
    # Two equally-reachable neutrals; planet A (prod 5) beats planet B (prod 1).
    obs = _obs(0, [
        [0, 0, 10.0, 10.0, 1.0, 50, 1],
        [1, -1, 40.0, 10.0, 2.0, 5, 5],   # high ROI, due east (angle ~ 0)
        [2, -1, 10.0, 40.0, 1.0, 5, 1],   # low ROI, due south (angle ~ pi/2)
    ])
    moves = plan_turn(obs)
    assert len(moves) == 1
    _assert_legal(obs, moves)
    # Chosen Shot aims east at the high-ROI planet, not south at the low-ROI one.
    assert abs(moves[0][1] - 0.0) < 0.2


def test_treats_any_nonneutral_owner_as_enemy():
    # I am player 0; a player-2 planet must be attacked just like any enemy.
    obs = _obs(0, [
        [0, 0, 10.0, 10.0, 1.0, 50, 1],
        [1, 2, 40.0, 10.0, 1.0, 5, 1],
    ])
    moves = plan_turn(obs)
    assert len(moves) == 1
    assert moves[0][0] == 0
    _assert_legal(obs, moves)


def test_never_spends_below_reserve():
    # A 25-ship enemy fleet is inbound to my planet (40 ships) -> reserve 25,
    # so at most 15 are spendable. The agent still attacks a cheap neutral, but
    # never launches more than the 15 it can spare.
    obs = _obs(
        0,
        [
            [0, 0, 10.0, 10.0, 1.0, 40, 1],   # mine, threatened
            [1, -1, 40.0, 10.0, 1.0, 5, 1],   # cheap neutral target
        ],
        fleets=[
            # enemy fleet just west of my planet, heading due east into it.
            [0, 1, 5.0, 10.0, 0.0, 99, 25],
        ],
    )
    moves = plan_turn(obs)
    assert len(moves) == 1
    sid, _, ships = moves[0]
    assert sid == 0
    assert int(ships) <= 40 - 25  # spent strictly within the reserve budget
    _assert_legal(obs, moves)


def test_send_is_capture_estimate_plus_buffer_when_affordable():
    # Plenty of ships, no threat: send = ships_needed + SHIP_BUFFER. A neutral
    # with 5 ships needs 6 to capture, so we send 6 + buffer.
    obs = _obs(0, [
        [0, 0, 10.0, 10.0, 1.0, 80, 1],
        [1, -1, 40.0, 10.0, 1.0, 5, 1],
    ])
    moves = plan_turn(obs)
    assert len(moves) == 1
    assert int(moves[0][2]) == 6 + SHIP_BUFFER
    _assert_legal(obs, moves)


# --- agents registry ----------------------------------------------------------


def test_registry_exposes_this_brain_and_default_resolves():
    # This brain is registered, and DEFAULT names a real entry in the registry.
    assert REGISTRY["roi_greedy"] is plan_turn
    assert DEFAULT in REGISTRY
