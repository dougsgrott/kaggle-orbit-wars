"""AG17 — byte-exact aim from the projection.

(1) `project_with_baseline` exposes per-turn (x,y) that byte-match an independent
    interpreter rollout. (2) `intercept_aim_exact` returns swept-verified, legal
    aims. (3) `flow_value_xa` is registered and emits legal shots in 2P/4P.
"""
import importlib

import pytest

from src.agents import REGISTRY, flow_value_xa


def test_registered():
    assert REGISTRY["flow_value_xa"] is flow_value_xa


def test_traj_xy_matches_interpreter():
    pytest.importorskip("kaggle_environments")
    from kaggle_environments import make
    from src import worldmodel as wm
    from src.garrison import project_with_baseline

    env = make("orbit_wars", debug=False)
    env.reset(2)
    obs = env.steps[0][0].observation
    H = 12
    _f0, _traj, _base, traj_xy = project_with_baseline(obs, H, num_players=2)
    assert len(traj_xy) == H + 1

    # turn 0 must equal the observed positions exactly.
    obs_xy = {int(p[0]): (float(p[2]), float(p[3])) for p in obs["planets"]}
    assert traj_xy[0] == obs_xy

    # every later turn must equal an independent do-nothing interpreter rollout.
    cur = wm.from_obs(obs, num_players=2)
    noop = [[], []]
    for k in range(1, H + 1):
        cur = wm.step(cur, noop)
        ref = {int(p[0]): (float(p[2]), float(p[3])) for p in wm.planets_of(cur)}
        assert traj_xy[k] == ref, f"turn {k} positions diverge from the interpreter"

    # orbiting planets actually move (the snapshot is not frozen).
    assert any(traj_xy[H].get(pid) != traj_xy[0].get(pid) for pid in obs_xy)


def test_intercept_aim_exact_verified():
    pytest.importorskip("kaggle_environments")
    from kaggle_environments import make
    from src.garrison import project_with_baseline
    from src.utils import intercept_aim_exact, _verify_swept_pos, _interp_xy

    env = make("orbit_wars", debug=False)
    env.reset(2)
    obs = env.steps[0][0].observation
    H = 14
    _f0, _traj, _base, traj_xy = project_with_baseline(obs, H, num_players=2)
    mine = [p for p in obs["planets"] if int(p[1]) == 0][0]
    sx, sy, sr = float(mine[2]), float(mine[3]), float(mine[4])
    # aim at every other planet; any non-None result must be swept-verified + sane.
    hit = 0
    for t in obs["planets"]:
        tid = int(t[0])
        if tid == int(mine[0]):
            continue
        tr = float(t[4])
        res = intercept_aim_exact(sx, sy, sr, tid, tr, 20, traj_xy, horizon=H)
        if res is None:
            continue
        ang, turns, px, py = res
        assert 1 <= turns <= H
        assert _verify_swept_pos(sx, sy, sr, ang, turns, 20, tr,
                                 lambda tt: _interp_xy(traj_xy, tid, tt))
        hit += 1
    assert hit >= 1  # at least one reachable target on the start board


@pytest.mark.parametrize("nP", [2, 4])
def test_emits_legal_shots(nP):
    pytest.importorskip("kaggle_environments")
    from kaggle_environments import make

    env = make("orbit_wars", debug=False)
    env.reset(nP)
    obs = env.steps[0][0].observation
    moves = flow_value_xa(obs)
    owners = {int(p[0]): (int(p[1]), float(p[5])) for p in obs["planets"]}
    for m in moves:
        sid, _angle, n = int(m[0]), float(m[1]), int(m[2])
        assert owners[sid][0] == 0
        assert 1 <= n <= owners[sid][1]
