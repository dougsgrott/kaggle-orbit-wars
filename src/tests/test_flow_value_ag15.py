"""Tests for AG15 — friendly-flip defense + pressure regroup (flow_value_def/_dr).

The `_flip_targets` / `_enemy_pressure` helpers are pure; the brains' legality
tests need kaggle_environments (the WorldModel projection).
"""
import importlib

import pytest

from src.agents import REGISTRY

_FV = importlib.import_module("src.agents.flow_value")
flow_value_def = REGISTRY.get("flow_value_def")
flow_value_dr = REGISTRY.get("flow_value_dr")


def test_registered():
    assert flow_value_def is not None and flow_value_dr is not None
    assert REGISTRY["flow_value_def"] is flow_value_def
    assert REGISTRY["flow_value_dr"] is flow_value_dr


def test_flip_targets_picks_projected_loss():
    # planet 7 (mine, prod 2) flips to an enemy at turn 2 in the projection;
    # planet 3 (mine) stays mine throughout -> only 7 is a defensive target.
    me = 0
    my_planets = [
        [7, 0, 30.0, 30.0, 2.0, 5.0, 2.0],
        [3, 0, 60.0, 60.0, 2.0, 9.0, 1.0],
    ]
    traj = [
        {7: (0, 5.0), 3: (0, 9.0)},
        {7: (0, 4.0), 3: (0, 10.0)},
        {7: (1, 6.0), 3: (0, 11.0)},   # 7 lost at turn 2
        {7: (1, 7.0), 3: (0, 12.0)},
    ]
    out = _FV._flip_targets(traj, my_planets, me, H=3)
    ids = [int(p[0]) for p in out]
    assert ids == [7]


def test_flip_targets_none_when_all_held():
    me = 0
    my_planets = [[7, 0, 30.0, 30.0, 2.0, 5.0, 2.0]]
    traj = [{7: (0, 5.0)}, {7: (0, 7.0)}, {7: (0, 9.0)}]
    assert _FV._flip_targets(traj, my_planets, me, H=2) == []


def test_enemy_pressure_higher_near_enemy():
    me = 0
    planets = [
        [0, 0, 20.0, 20.0, 2.0, 10.0, 1.0],   # mine, far from enemy
        [1, 0, 78.0, 78.0, 2.0, 10.0, 1.0],   # mine, next to the enemy
        [2, 1, 82.0, 82.0, 2.0, 30.0, 1.0],   # enemy
    ]
    pr = _FV._enemy_pressure(planets, me, H=14)
    assert pr[1] > pr[0]  # the planet beside the enemy is more pressured


@pytest.mark.parametrize("name", ["flow_value_def", "flow_value_dr"])
def test_brain_emits_legal_shots(name):
    pytest.importorskip("kaggle_environments")
    from kaggle_environments import make

    env = make("orbit_wars", debug=False)
    env.reset(2)
    obs = env.steps[0][0].observation
    moves = REGISTRY[name](obs)
    owners = {int(p[0]): (int(p[1]), float(p[5])) for p in obs["planets"]}
    for m in moves:
        sid, _angle, n = int(m[0]), float(m[1]), int(m[2])
        assert owners[sid][0] == 0
        assert 1 <= n <= owners[sid][1]
