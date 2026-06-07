"""producer_port — the full Producer vendored as our champion base.

Written AFTER the port was battle-validated vs the pristine original (2P 24/48 = 50%
mirror, n=48, 0 faults). These pin registration, legality, the DEFAULT promotion, and
a small parity smoke. All require torch (the vendored engine); skipped where absent.
"""
import pytest

pytest.importorskip("torch")
pytest.importorskip("kaggle_environments")

from src.agents import REGISTRY, DEFAULT, producer_port


def test_registered_and_default():
    assert REGISTRY["producer_port"] is producer_port
    assert DEFAULT == "producer_port"


@pytest.mark.parametrize("nP", [2, 4])
def test_emits_legal_shots(nP):
    from kaggle_environments import make
    env = make("orbit_wars", debug=False)
    env.reset(nP)
    obs = env.steps[0][0].observation
    moves = producer_port(obs)
    owners = {int(p[0]): (int(p[1]), float(p[5])) for p in obs["planets"]}
    for m in moves:
        sid, _angle, n = int(m[0]), float(m[1]), int(m[2])
        assert owners[sid][0] == 0
        assert 1 <= n <= owners[sid][1]


def test_battle_parity_smoke():
    """A couple of seeds vs the pristine original: no fault, and not a blowout either
    way (same engine -> mirror). Full battle (n=48 -> 50%) is in /tmp/battle.py."""
    from src import diag
    from src.opponents.producer import agent as pristine
    wins = 0
    for seed in (9000, 9001):
        for seat in (0, 1):
            g = diag.play_logged(producer_port, pristine, seed=seed,
                                 num_players=2, our_seat=seat)
            assert g.placement in (1, 2)        # finished, no fault
            wins += int(g.placement == 1)
    assert 1 <= wins <= 3                        # mirror-ish over 4 paired games
