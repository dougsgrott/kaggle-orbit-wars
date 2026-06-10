"""Tests for ``producer_open`` (AG20 — early-game position-building layer) and the AG19
contribution-gate helpers. Game-playing checks are torch/kaggle skip-guarded."""
import importlib
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("kaggle_environments")

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def test_registered():
    from src.agents import REGISTRY

    assert "producer_open" in REGISTRY


def test_position_term_neutral_expand():
    """The neutral_expand term gives a production-scaled bonus to NEUTRAL targets only;
    own/enemy targets get 0 (the greedy's attack/defense candidates are unchanged)."""
    M = importlib.import_module("src.agents.producer_open")
    # planets [P,7] = [id, owner, x, y, radius, ships, production].
    planets = torch.tensor([
        [0, -1, 50, 50, 2, 10, 5.0],   # neutral, prod 5
        [1, 0, 10, 10, 2, 10, 3.0],    # mine (owner 0), prod 3
        [2, 1, 90, 90, 2, 10, 2.0],    # enemy, prod 2
    ])
    cand_tgt = torch.tensor([0, 1, 2])
    ref = torch.zeros(3)
    old_term, old_w = M._TERM, M._WEIGHT
    M._TERM, M._WEIGHT = "neutral_expand", 2.0
    M._CTX["player_id"] = 0
    try:
        t = M._position_term(cand_tgt, planets, ref=ref)
    finally:
        M._TERM, M._WEIGHT = old_term, old_w
    assert abs(t[0].item() - 10.0) < 1e-5  # 2.0 * prod 5
    assert t[1].item() == 0.0 and t[2].item() == 0.0


def test_open_plays_legal_no_fault():
    """Legal play vs the sota Producer file (env-isolated seats → no runtime sharing),
    2P + 4P, 0 faults."""
    from src.arena import EpisodeConfig, run_episode
    from src.agents import REGISTRY
    from src.opponents import PRODUCER

    op = REGISTRY["producer_open"]
    r2 = run_episode([op, PRODUCER], EpisodeConfig(num_players=2, seed=5000, episode_steps=40))
    r4 = run_episode([op, PRODUCER, PRODUCER, PRODUCER],
                     EpisodeConfig(num_players=4, seed=5001, episode_steps=40))
    assert not any(o.faulted for o in r2.outcomes)
    assert not any(o.faulted for o in r4.outcomes)


def test_gate_min_sig_k():
    """The gate's power helper: smallest favourable discordant split that is significant at
    p<0.05 under the sign test — None when there are too few discordant pairs to ever resolve."""
    from src.gate import _min_sig_k

    assert _min_sig_k(5) is None          # 5/5 -> p=0.0625, never significant
    assert _min_sig_k(6) == 6             # 6/6 -> p=0.03125 < 0.05
    k = _min_sig_k(40)
    assert k is not None and 20 < k < 40  # needs a clear majority, not unanimity
