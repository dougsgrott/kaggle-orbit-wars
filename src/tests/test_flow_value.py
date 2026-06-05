"""Tests for the `flow_value` brain (AG13 — competitive flow-diff value).

Behavioural: the value is computed by rolling the WorldModel (interpreter), so the
property tests need kaggle_environments. We assert registration, legal shots, and
the two defining properties of the value (a real capture scores positive; a launch
the projection shows redundant scores ~0).
"""
import importlib

import pytest

from src.agents import REGISTRY

flow_value = REGISTRY.get("flow_value")
_FV = importlib.import_module("src.agents.flow_value")  # module (constants + helpers)


def _obs(planets, fleets=None, player=0):
    return {
        "player": player,
        "planets": [list(p) for p in planets],
        "fleets": [list(f) for f in (fleets or [])],
        "initial_planets": [list(p) for p in planets],
        "angular_velocity": 0.0,
        "comets": [],
        "comet_planet_ids": [],
    }


def test_registered_in_registry():
    assert flow_value is not None
    assert REGISTRY["flow_value"] is flow_value


def test_emits_legal_shots():
    pytest.importorskip("kaggle_environments")
    from kaggle_environments import make

    env = make("orbit_wars", debug=False)
    env.reset(2)
    obs = env.steps[0][0].observation
    moves = flow_value(obs)
    owners = {int(p[0]): (int(p[1]), float(p[5])) for p in obs["planets"]}
    for m in moves:
        sid, _angle, n = int(m[0]), float(m[1]), int(m[2])
        assert owners[sid][0] == 0
        assert 1 <= n <= owners[sid][1]


def test_capture_scores_positive():
    pytest.importorskip("kaggle_environments")
    from src.garrison import project_with_baseline
    from src.utils import aim_with_prediction
    from src.agents.roi_greedy_predict import _build_world

    # mine: big garrison; a small static neutral nearby and reachable. Both planets
    # sit in the lower-left so the straight shot does not cross the sun (centre 50,50).
    planets = [
        [0, 0, 25.0, 25.0, 2.0, 40.0, 2.0],    # mine, 40 ships
        [1, -1, 40.0, 32.0, 2.0, 3.0, 2.0],    # neutral, 3 ships, close
    ]
    obs = _obs(planets)
    H = _FV.H_DEFAULT
    world = _build_world(obs)
    fstate0, traj, base = project_with_baseline(obs, H, num_players=2)
    aim = aim_with_prediction(25.0, 25.0, 2.0, 1, 40.0, 32.0, 2.0, 20, **world)
    assert aim is not None
    val = _FV._candidate_value(fstate0, 0, 2, [0, float(aim[0]), 20], base, H)
    # capturing a producing neutral adds net ships over the horizon -> clearly > 0.
    assert val > _FV.SCORE_THRESHOLD


def test_redundant_launch_scores_near_zero():
    pytest.importorskip("kaggle_environments")
    from src.garrison import project_with_baseline
    from src.utils import aim_with_prediction
    from src.agents.roi_greedy_predict import _build_world

    # I already own both planets: a launch from 0 to 1 only relocates my own ships,
    # so the net-ship delta over the horizon is ~0 (well below threshold). Off-centre
    # so the shot is sun-safe.
    planets = [
        [0, 0, 25.0, 25.0, 2.0, 40.0, 2.0],
        [1, 0, 40.0, 32.0, 2.0, 10.0, 2.0],
    ]
    obs = _obs(planets)
    H = _FV.H_DEFAULT
    world = _build_world(obs)
    fstate0, traj, base = project_with_baseline(obs, H, num_players=2)
    aim = aim_with_prediction(25.0, 25.0, 2.0, 1, 40.0, 32.0, 2.0, 10, **world)
    assert aim is not None
    val = _FV._candidate_value(fstate0, 0, 2, [0, float(aim[0]), 10], base, H)
    assert abs(val) <= _FV.SCORE_THRESHOLD
