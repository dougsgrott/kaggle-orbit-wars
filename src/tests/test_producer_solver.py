"""Tests for ``producer_solver`` (AG23 — Track B): registration, the exact-ILP
selector's correctness (the measurement instrument), and behaviour-neutral off-mode
play. The Stage-0 premise outcome was an **honest negative** — the Producer's greedy
selection already realises the ILP optimum on ~99% of turns (see wiki/measured_log.md
and docs/issues/agent/AG23-solver-turn-allocation.md) — so the live solver is not built;
these tests guard the reusable building blocks (`_solve_ilp`) and the isolation.
"""
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only (matches local play)


def test_registered():
    from src.agents import REGISTRY

    assert "producer_solver" in REGISTRY


def test_solve_ilp_detects_capacity_gap():
    """The ILP must beat the greedy exactly when a per-source budget binds: greedy takes
    the top-scoring wave and starves a higher-total pair (the instrument's validity)."""
    from src.agents.producer_solver import _solve_ilp

    # One source (planet 0). A:0->t0 send10 sc3.0  B:0->t1 send6 sc2.5  C:0->t0 send5 sc2.0
    sc = torch.tensor([3.0, 2.5, 2.0])
    src = torch.tensor([[0], [0], [0]])
    send = torch.tensor([[10.0], [6.0], [5.0]])
    tsh = torch.tensor([0, 1, 0])
    tsl = torch.tensor([1, 2, 1])
    # budget 11: greedy picks A(3.0, uses 10), can't fund B -> 3.0; optimum C+B = 4.5.
    bnd = _solve_ilp(score=sc, cand_src=src, cand_send=send, cand_tgt_short=tsh,
                     cand_tgt_slot=tsl, source_budget=torch.tensor([11.0, 99, 99]),
                     roi_threshold=1.5, W=6)
    assert abs(bnd[0] - 4.5) < 1e-6 and set(bnd[1]) == {1, 2}
    # budget inf: A+B = 5.5 (one per target; C blocked by t0).
    free = _solve_ilp(score=sc, cand_src=src, cand_send=send, cand_tgt_short=tsh,
                      cand_tgt_slot=tsl, source_budget=torch.tensor([99.0, 99, 99]),
                      roi_threshold=1.5, W=6)
    assert abs(free[0] - 5.5) < 1e-6 and set(free[1]) == {0, 1}


def test_solve_ilp_cardinality_and_roi_threshold():
    """Cardinality caps at W; sub-ROI candidates are ineligible."""
    from src.agents.producer_solver import _solve_ilp

    sc = torch.tensor([5.0, 4.0, 3.0, 1.0])          # 4th below roi 1.5
    src = torch.tensor([[0], [1], [2], [3]])
    send = torch.tensor([[1.0], [1.0], [1.0], [1.0]])
    tsh = torch.tensor([0, 1, 2, 3])
    tsl = torch.tensor([10, 11, 12, 13])
    r = _solve_ilp(score=sc, cand_src=src, cand_send=send, cand_tgt_short=tsh,
                   cand_tgt_slot=tsl, source_budget=torch.tensor([9.0] * 14),
                   roi_threshold=1.5, W=2)
    assert abs(r[0] - 9.0) < 1e-6 and set(r[1]) == {0, 1}  # top-2 (5+4); 4th excluded


def test_solve_ilp_empty():
    """No eligible candidate -> zero objective, no selection."""
    from src.agents.producer_solver import _solve_ilp

    sc = torch.tensor([1.0, 0.5])                     # both below roi 1.5
    out = _solve_ilp(score=sc, cand_src=torch.tensor([[0], [1]]),
                     cand_send=torch.tensor([[1.0], [1.0]]), cand_tgt_short=torch.tensor([0, 1]),
                     cand_tgt_slot=torch.tensor([2, 3]), source_budget=torch.tensor([9.0] * 4),
                     roi_threshold=1.5, W=6)
    assert out == (0.0, [], 0)


def test_off_mode_plays_legal_no_fault():
    """Off mode (default) is the pure Producer on an isolated runtime copy: legal play,
    0 faults, and producer_port is undisturbed in the same process."""
    from src.arena import EpisodeConfig, run_episode
    from src.agents import REGISTRY

    r = run_episode(
        [REGISTRY["producer_solver"], REGISTRY["producer_port"]],
        EpisodeConfig(num_players=2, seed=3000, episode_steps=40),
    )
    assert not any(o.faulted for o in r.outcomes)
