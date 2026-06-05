"""Tests for the do-nothing garrison projection + sizing helpers (AG12).

The projection rolls the WorldModel (interpreter) forward, so the property tests
need kaggle_environments; `capture_floor` / `safe_drain` are pure given a
trajectory and are tested directly on a hand-built one.
"""
import pytest

from src import garrison


def _obs(planets, fleets=None):
    # planet schema: [id, owner, x, y, radius, ships, production]
    return {
        "player": 0,
        "planets": [list(p) for p in planets],
        "fleets": [list(f) for f in (fleets or [])],
        "initial_planets": [list(p) for p in planets],
        "angular_velocity": 0.0,
        "comets": [],
        "comet_planet_ids": [],
    }


# --- pure helpers (no env) --------------------------------------------------

def test_capture_floor_reinforcement_is_one():
    # If the projection says the target is mine at k, the floor is 1.
    traj = [{}, {5: (0, 20.0)}]  # me = 0 owns planet 5 at turn 1
    assert garrison.capture_floor(traj, 5, me=0, k=1) == 1


def test_capture_floor_clears_projected_defenders_plus_overhead():
    traj = [{}, {5: (1, 7.0)}]  # enemy holds 7 ships at arrival
    # ceil(7 + 1) = 8
    assert garrison.capture_floor(traj, 5, me=0, k=1, overhead=1.0) == 8
    # never below 1, even for an empty neutral
    assert garrison.capture_floor([{}, {5: (-1, 0.0)}], 5, me=0, k=1, overhead=0.0) == 1


def test_capture_floor_clamps_k_into_horizon():
    traj = [{}, {5: (1, 3.0)}]
    # k beyond the horizon clamps to the last frame (still 3 defenders).
    assert garrison.capture_floor(traj, 5, me=0, k=99, overhead=1.0) == 4


def test_safe_drain_is_min_of_held_trajectory_capped_at_now():
    # owned all the way; ships dip to 4 at turn 2 then recover -> can shed 4.
    traj = [
        {5: (0, 10.0)},   # now
        {5: (0, 6.0)},
        {5: (0, 4.0)},    # worst held turn
        {5: (0, 9.0)},
    ]
    assert garrison.safe_drain(traj, 5, me=0, H=3) == 4.0


def test_safe_drain_capped_at_current_ships():
    # trajectory only grows -> min over held >= now, so cap is current ships.
    traj = [{5: (0, 8.0)}, {5: (0, 10.0)}, {5: (0, 12.0)}]
    assert garrison.safe_drain(traj, 5, me=0, H=2) == 8.0


def test_safe_drain_doomed_source_sends_all():
    # lost by turn 1 (enemy owns it) -> nothing to protect, shed everything.
    traj = [{5: (0, 9.0)}, {5: (1, 5.0)}, {5: (1, 6.0)}]
    assert garrison.safe_drain(traj, 5, me=0, H=2) == 9.0


# --- projection (needs the interpreter) -------------------------------------

def test_projection_turn0_matches_obs():
    pytest.importorskip("kaggle_environments")
    planets = [
        [0, 0, 30.0, 50.0, 2.0, 12.0, 2.0],   # mine
        [1, -1, 70.0, 50.0, 2.0, 5.0, 1.0],   # neutral
    ]
    traj = garrison.garrison_projection(_obs(planets), H=5, num_players=2)
    assert len(traj) == 6
    assert traj[0][0] == (0, 12.0)
    assert traj[0][1] == (-1, 5.0)


def test_projection_grows_owned_ships_by_production():
    pytest.importorskip("kaggle_environments")
    planets = [[0, 0, 30.0, 50.0, 2.0, 12.0, 3.0]]  # mine, prod 3, no threats
    traj = garrison.garrison_projection(_obs(planets), H=3, num_players=2)
    owner1, ships1 = traj[1][0]
    assert owner1 == 0
    assert ships1 > 12.0  # production accrued under do-nothing
