"""Tests for the continuous-intercept aimer (AG14) + the `flow_value_ia` brain.

`intercept_aim` and the swept-pair helper are pure (no env); the brain's
`plan_turn` runs the WorldModel projection, so its legality test needs
kaggle_environments.
"""
import math

import pytest

from src.agents import REGISTRY
from src import utils

flow_value_ia = REGISTRY.get("flow_value_ia")


def test_registered_in_registry():
    assert flow_value_ia is not None
    assert REGISTRY["flow_value_ia"] is flow_value_ia


def test_swept_pair_hits_moving_target():
    # Point goes (0,0)->(10,0); circle r=1 centred at (10,0) stationary -> contact.
    assert utils._swept_point_circle_hit(0, 0, 10, 0, 10, 0, 10, 0, 1.0)
    # Same fleet step, circle far away the whole step -> no contact.
    assert not utils._swept_point_circle_hit(0, 0, 10, 0, 0, 50, 0, 50, 1.0)
    # Crossing paths within the step (relative motion brings them together).
    assert utils._swept_point_circle_hit(0, 0, 10, 0, 5, 5, 5, -5, 1.0)


def test_intercept_aim_static_target_clear_shot():
    # Static target straight ahead, off-centre so the path avoids the sun.
    world = {"initial_by_id": {1: {"x": 40.0, "y": 30.0}},
             "angular_velocity": 0.0, "comets": [], "comet_ids": set()}
    res = utils.intercept_aim(20.0, 25.0, 2.0, 1, 40.0, 30.0, 2.0, 15, **world)
    assert res is not None
    angle, turns, px, py = res
    # aims roughly toward the target
    assert abs(math.atan2(30.0 - 25.0, 40.0 - 20.0) - angle) < 0.3
    assert turns >= 1


def test_intercept_aim_rejects_sun_blocked_shot():
    # Source and target on opposite sides of the sun (centre 50,50): the straight
    # shot grazes the sun, so the swept verify must reject it (None).
    world = {"initial_by_id": {1: {"x": 70.0, "y": 50.0}},
             "angular_velocity": 0.0, "comets": [], "comet_ids": set()}
    res = utils.intercept_aim(30.0, 50.0, 2.0, 1, 70.0, 50.0, 2.0, 15, **world)
    assert res is None


def test_brain_emits_legal_shots():
    pytest.importorskip("kaggle_environments")
    from kaggle_environments import make

    env = make("orbit_wars", debug=False)
    env.reset(2)
    obs = env.steps[0][0].observation
    moves = flow_value_ia(obs)
    owners = {int(p[0]): (int(p[1]), float(p[5])) for p in obs["planets"]}
    for m in moves:
        sid, _angle, n = int(m[0]), float(m[1]), int(m[2])
        assert owners[sid][0] == 0
        assert 1 <= n <= owners[sid][1]
