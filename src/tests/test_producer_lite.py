"""Tests for the `roi_projected` brain (AG12).

Behavioural: the brain rolls the WorldModel projection, so it needs
kaggle_environments. We assert it is registered and emits legal Shots.
"""
import pytest

from src.agents import REGISTRY

roi_projected = REGISTRY.get("roi_projected")


def test_registered_in_registry():
    assert roi_projected is not None
    assert REGISTRY["roi_projected"] is roi_projected


def test_emits_legal_shots():
    pytest.importorskip("kaggle_environments")
    from kaggle_environments import make

    env = make("orbit_wars", debug=False)
    env.reset(2)
    # advance a few turns so fleets are in flight and captures resolve
    obs = env.steps[0][0].observation
    moves = roi_projected(obs)
    owners = {int(p[0]): (int(p[1]), float(p[5])) for p in obs["planets"]}
    for m in moves:
        sid, _angle, n = int(m[0]), float(m[1]), int(m[2])
        assert owners[sid][0] == 0, "may only launch from an owned planet"
        assert 1 <= n <= owners[sid][1], "ship count must be in [1, available]"
