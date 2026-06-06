"""AG16 — flow_value_cfg (Producer per-format config) emits legal, format-aware shots."""
import pytest

from src.agents import REGISTRY, flow_value_cfg
from src.agents.flow_value_cfg import _num_players, _CFG_2P, _CFG_4P


def test_registered():
    assert REGISTRY["flow_value_cfg"] is flow_value_cfg


@pytest.mark.parametrize("nP", [2, 4])
def test_emits_legal_shots(nP):
    pytest.importorskip("kaggle_environments")
    from kaggle_environments import make

    env = make("orbit_wars", debug=False)
    env.reset(nP)
    obs = env.steps[0][0].observation
    assert _num_players(obs) == nP
    moves = flow_value_cfg(obs)
    owners = {int(p[0]): (int(p[1]), float(p[5])) for p in obs["planets"]}
    for m in moves:
        sid, _angle, n = int(m[0]), float(m[1]), int(m[2])
        assert owners[sid][0] == 0          # only launch from my own planets
        assert 1 <= n <= owners[sid][1]     # legal ship count


def test_presets_distinct():
    # 4P narrows sources + horizon + defensive quota relative to 2P.
    assert _CFG_4P["H"] < _CFG_2P["H"]
    assert _CFG_4P["max_sources"] < _CFG_2P["max_sources"]
    assert _CFG_4P["max_def"] < _CFG_2P["max_def"]
