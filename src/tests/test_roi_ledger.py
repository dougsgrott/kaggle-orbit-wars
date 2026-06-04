"""Pure tests for the M1c `roi_ledger` brain — arrival-ledger planning.

No kaggle_environments. Contract: legal motion-aimed Shots; behaves like v1 when
nothing is in flight; skips targets its own fleets already cover (sending force to
uncovered planets instead); sizes launches net of in-flight fleets. Coordinates
confirmed against src.utils.
"""

import math

from src.agents.roi_greedy_predict import plan_turn as v1
from src.agents.roi_ledger import plan_turn as ledger, build_arrival_ledger
from src.agents import REGISTRY


# planet row: [id, owner, x, y, radius, ships, production]   (owner -1 = neutral)
# fleet  row: [id, owner, x, y, angle, from_planet_id, ships]
def _obs(player, planets, fleets=()):
    return {
        "player": player,
        "planets": [list(p) for p in planets],
        "fleets": [list(f) for f in fleets],
    }


def _assert_legal(obs, moves):
    me = obs["player"]
    by_id = {int(p[0]): p for p in obs["planets"]}
    for sid, angle, ships in moves:
        assert int(by_id[sid][1]) == me
        assert 1 <= int(ships) <= int(by_id[sid][5])
        assert math.isfinite(angle)


# --- ledger builder -----------------------------------------------------------


def test_build_ledger_attributes_fleet_to_target_with_eta():
    planets = [[0, 0, 90.0, 90.0, 1.0, 80, 1], [1, -1, 60.0, 90.0, 1.0, 5, 1]]
    # My 10-ship fleet at (70,90) heading west (pi) toward planet 1 at (60,90).
    fleets = [[0, 0, 70.0, 90.0, math.pi, 0, 10]]
    led = build_arrival_ledger(0, planets, fleets)
    assert 1 in led
    eta, owner, ships = led[1][0]
    assert owner == 0 and ships == 10
    assert eta >= 1  # ~10 units / fleet_speed(10) -> a handful of turns


def test_empty_when_no_fleets():
    planets = [[0, 0, 90.0, 90.0, 1.0, 80, 1], [1, -1, 60.0, 90.0, 1.0, 5, 1]]
    assert build_arrival_ledger(0, planets, []) == {}


# --- parity & the core fix ----------------------------------------------------


def test_no_inflight_behaves_like_v1():
    obs = _obs(0, [[0, 0, 90.0, 90.0, 1.0, 80, 1], [1, -1, 60.0, 90.0, 1.0, 5, 1]])
    assert ledger(obs) == v1(obs)


def test_skips_target_already_covered_by_my_fleet_and_expands_elsewhere():
    # My planet 0 (80) with TWO reachable neutrals; my 10-ship fleet already
    # inbound to planet 1 (enough to take its 5). v1 relaunches at the nearest
    # (planet 1); ledger skips it and sends to the uncovered planet 2.
    planets = [
        [0, 0, 90.0, 90.0, 1.0, 80, 1],
        [1, -1, 60.0, 90.0, 1.0, 5, 1],   # already covered
        [2, -1, 90.0, 60.0, 1.0, 5, 1],   # uncovered
    ]
    fleets = [[0, 0, 70.0, 90.0, math.pi, 0, 10]]  # mine -> planet 1
    obs = _obs(0, planets, fleets)
    v1_move = v1(obs)
    led_move = ledger(obs)
    _assert_legal(obs, led_move)
    assert len(led_move) == 1
    # v1 aims west at planet 1 (~pi); ledger aims north at planet 2 (~ -pi/2).
    assert abs(v1_move[0][1] - math.pi) < 0.3
    assert abs(led_move[0][1] - (-math.pi / 2)) < 0.3


def test_sizes_launch_net_of_friendly_inflight():
    # Same single target, but a friendly fleet already covers PART of the need.
    # Bare need for a 40-ship enemy is large; with a 30-ship friendly fleet
    # co-inbound, ledger should send fewer ships than v1 (which ignores it).
    planets = [
        [0, 0, 90.0, 90.0, 1.0, 90, 1],
        [1, 1, 60.0, 90.0, 1.0, 40, 1],   # enemy target, 40 ships
    ]
    # Friendly 30-ship fleet close in, ~same ETA as a fresh launch.
    fleets = [[0, 0, 68.0, 90.0, math.pi, 0, 30]]
    obs = _obs(0, planets, fleets)
    v1_sent = sum(m[2] for m in v1(obs))
    led_moves = ledger(obs)
    _assert_legal(obs, led_moves)
    led_sent = sum(m[2] for m in led_moves)
    assert led_moves, "ledger should still launch the remainder"
    assert led_sent < v1_sent  # credits the friendly fleet en route


def test_all_shots_legal_multi_planet():
    planets = [
        [0, 0, 90.0, 90.0, 1.0, 60, 1],
        [1, 0, 85.0, 95.0, 1.0, 40, 1],
        [2, -1, 60.0, 90.0, 1.0, 5, 1],
        [3, 1, 90.0, 60.0, 1.0, 8, 1],
    ]
    obs = _obs(0, planets)
    _assert_legal(obs, ledger(obs))


def test_registered_in_registry():
    assert REGISTRY["roi_ledger"] is ledger
