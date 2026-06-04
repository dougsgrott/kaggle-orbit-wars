"""Smoke tests for the L2 'strong' ladder tier (docs/issues/eval/L2).

The ported higher-rated agents (lb1200 ~815.9, lb1224 ~727.5) must run as
file-path opponents in our arena with no fault, in both 1v1 and 4P — the same
contract the Boss meets. Behavioural; skipped without kaggle_environments.
"""
import pytest

from src import opponents


def test_strong_tier_is_wired():
    # Pure: the tier exists, is non-empty, and is part of the Ladder.
    assert opponents.STRONG, "strong tier should list the ported agents"
    assert opponents.LADDER["strong"] == opponents.STRONG
    # Ordered above the Boss in the Ladder (signal stays non-saturating).
    tiers = list(opponents.LADDER)
    assert tiers.index("strong") > tiers.index("boss")
    assert set(opponents.STRONG) <= set(opponents.LADDER_ALL)


@pytest.mark.parametrize("opp", [opponents.LB1200, opponents.LB1224])
def test_strong_opponent_runs_clean_1v1(opp):
    pytest.importorskip("kaggle_environments")
    from src.arena import run_episode, EpisodeConfig

    # vs a do-nothing seat; the strong agent must play a legal, fault-free game.
    result = run_episode([opp, opponents.STARTER],
                         EpisodeConfig(num_players=2, seed=4201, episode_steps=80))
    assert result.outcomes[0].faulted is False
    assert result.outcomes[0].status == "DONE"


def test_strong_opponent_runs_clean_4p():
    pytest.importorskip("kaggle_environments")
    from src.arena import run_episode, EpisodeConfig

    result = run_episode([opponents.LB1200, opponents.LB1224,
                          opponents.STARTER, opponents.STARTER],
                         EpisodeConfig(num_players=4, seed=4202, episode_steps=80))
    assert all(not result.outcomes[i].faulted for i in (0, 1))
    assert sorted(o.placement for o in result.outcomes) == [1, 2, 3, 4]
