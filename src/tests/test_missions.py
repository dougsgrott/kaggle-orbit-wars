"""Pure tests for the M1a `missions` brain — global multi-source allocation.

No kaggle_environments. Contract: legal motion-aimed Shots, identical to v1 on
degenerate single-source boards, but combines force across planets to take a
target v1 (per-planet greedy) abandons. Coordinates confirmed against src.utils.
"""

import math

from src.agents.roi_greedy_predict import plan_turn as v1
from src.agents.missions import plan_turn as missions
from src.agents import REGISTRY


# planet row: [id, owner, x, y, radius, ships, production]   (owner -1 = neutral)
def _obs(player, planets, fleets=(), motion=False):
    o = {
        "player": player,
        "planets": [list(p) for p in planets],
        "fleets": [list(f) for f in fleets],
    }
    if motion:
        o["initial_planets"] = [list(p) for p in planets]
        o["angular_velocity"] = 0.05
        o["comets"] = []
        o["comet_planet_ids"] = []
    return o


def _assert_legal(obs, moves):
    me = obs["player"]
    by_id = {int(p[0]): p for p in obs["planets"]}
    for sid, angle, ships in moves:
        assert int(by_id[sid][1]) == me                 # launch from a planet I own
        assert 1 <= int(ships) <= int(by_id[sid][5])    # legal ship count
        assert math.isfinite(angle)


def test_no_targets_returns_empty():
    assert missions(_obs(0, [[0, 0, 90.0, 90.0, 1.0, 50, 1]])) == []


def test_single_source_matches_v1_on_static_board():
    # One owned planet, one reachable neutral, static board -> same Shot as v1.
    obs = _obs(0, [[0, 0, 90.0, 90.0, 1.0, 80, 1], [1, -1, 60.0, 90.0, 1.0, 5, 1]])
    assert missions(obs) == v1(obs)


def test_combines_two_planets_to_take_target_v1_abandons():
    # An 80-ship enemy neither of my 50-ship planets can capture alone, but can
    # together. v1 (per-planet greedy) sends nothing; missions concentrates force.
    planets = [
        [0, 0, 85.0, 95.0, 1.0, 50, 1],   # my planet A
        [1, 0, 95.0, 85.0, 1.0, 50, 1],   # my planet B
        [2, 1, 60.0, 95.0, 1.0, 80, 1],   # enemy target (clear of the sun)
    ]
    obs = _obs(0, planets)
    assert v1(obs) == []                  # v1 abandons it
    moves = missions(obs)
    _assert_legal(obs, moves)
    # Both my planets fire at the same enemy target.
    assert {m[0] for m in moves} == {0, 1}
    # Combined fleet covers the capture need (>= the 80-ship garrison).
    assert sum(m[2] for m in moves) >= 80


def test_does_not_overshoot_need_plus_buffer():
    # Two big planets, a weak neutral: missions must not pour both planets' full
    # garrisons in — total sent is bounded by need + buffer (v1 would have each
    # co-firing planet send the full amount).
    planets = [
        [0, 0, 85.0, 95.0, 1.0, 90, 1],
        [1, 0, 95.0, 85.0, 1.0, 90, 1],
        [2, -1, 60.0, 95.0, 1.0, 5, 1],   # weak neutral: need ~6
    ]
    obs = _obs(0, planets)
    moves = missions(obs)
    _assert_legal(obs, moves)
    assert sum(m[2] for m in moves) <= 6 + 2  # need(6) + SHIP_BUFFER(2)


def test_budget_is_shared_across_targets_not_double_spent():
    # One small planet, two neutral targets: its limited budget can't fully fund
    # both, so total ships launched never exceeds what the planet has.
    planets = [
        [0, 0, 90.0, 90.0, 1.0, 12, 1],   # my planet, only 12 ships
        [1, -1, 60.0, 90.0, 1.0, 5, 1],
        [2, -1, 90.0, 60.0, 1.0, 5, 1],
    ]
    obs = _obs(0, planets)
    moves = missions(obs)
    _assert_legal(obs, moves)
    assert sum(m[2] for m in moves if m[0] == 0) <= 12


def test_registered_in_registry():
    assert REGISTRY["missions"] is missions
