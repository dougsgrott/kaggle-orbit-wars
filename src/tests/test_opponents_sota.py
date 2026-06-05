"""Smoke tests for the 'sota' ladder tier (The Producer benchmark).

The ported top submission ("The Producer", slawekbiel) must run as a file-path
opponent in our arena with no fault, in both 1v1 and 4P — the same contract the
Boss and the L2 'strong' tier meet. It REQUIRES torch (the rest of the ladder is
stdlib), so the whole module is skipped where torch is absent, and the tier is
only wired into LADDER when torch imports (opponents.HAVE_TORCH). Benchmark-only:
this agent is never part of our shippable solution and is never resubmitted.
"""
import pytest

pytest.importorskip("torch")

from src import opponents


def test_sota_tier_is_wired_when_torch_present():
    # With torch importable the tier is non-empty and sits at the top of the field.
    assert opponents.HAVE_TORCH is True
    assert opponents.SOTA == [opponents.PRODUCER]
    assert opponents.LADDER["sota"] == opponents.SOTA
    tiers = list(opponents.LADDER)
    assert tiers.index("sota") > tiers.index("strong") > tiers.index("boss")
    assert opponents.PRODUCER in opponents.LADDER_ALL


def test_producer_runs_clean_1v1():
    pytest.importorskip("kaggle_environments")
    from src.arena import run_episode, EpisodeConfig

    # vs a do-nothing seat; the Producer must play a legal, fault-free game.
    result = run_episode([opponents.PRODUCER, opponents.STARTER],
                         EpisodeConfig(num_players=2, seed=4201, episode_steps=80))
    assert result.outcomes[0].faulted is False
    assert result.outcomes[0].status == "DONE"


def test_producer_runs_clean_4p():
    pytest.importorskip("kaggle_environments")
    from src.arena import run_episode, EpisodeConfig

    result = run_episode([opponents.PRODUCER, opponents.LB1200,
                          opponents.STARTER, opponents.STARTER],
                         EpisodeConfig(num_players=4, seed=4202, episode_steps=80))
    assert all(not result.outcomes[i].faulted for i in (0, 1))
    assert sorted(o.placement for o in result.outcomes) == [1, 2, 3, 4]
