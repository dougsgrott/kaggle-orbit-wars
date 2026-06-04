"""Tests for the MCTS brain (AG9, M4).

Like lookahead, `plan_turn` simulates via the WorldModel (interpreter), so the
behavioural tests need kaggle_environments (skipped otherwise). We test the search
*mechanics* and contracts, not move superiority — with a do-nothing opponent model
deeper search legitimately diverges from lookahead (see the module docstring), so
there's no clean "MCTS beats greedy" assertion to make; that's what the boss A/B is for.
"""
import importlib
import math
import time

import pytest

from src.agents import REGISTRY

_mcts_mod = importlib.import_module("src.agents.mcts")
mcts = REGISTRY["mcts"]


def _obs(player, planets, fleets=()):
    return {
        "player": player,
        "planets": [list(p) for p in planets],
        "fleets": [list(f) for f in fleets],
        "initial_planets": [list(p) for p in planets],
        "angular_velocity": 0.0,
        "comets": [],
        "comet_planet_ids": [],
    }


def _assert_legal(obs, moves):
    me = obs["player"]
    by_id = {int(p[0]): p for p in obs["planets"]}
    for sid, angle, ships in moves:
        assert int(by_id[sid][1]) == me
        assert 1 <= int(ships) <= int(by_id[sid][5])
        assert math.isfinite(angle)


def test_registered_in_registry():
    assert REGISTRY["mcts"] is mcts


def test_no_targets_returns_empty():
    # No enemy/neutral targets -> single candidate (hold) -> the early return path.
    pytest.importorskip("kaggle_environments")
    assert mcts(_obs(0, [[0, 0, 90.0, 90.0, 1.0, 50, 1]])) == []


def test_returns_legal_move():
    pytest.importorskip("kaggle_environments")
    obs = _obs(0, [
        [0, 0, 90.0, 90.0, 1.0, 80, 1],
        [1, -1, 60.0, 90.0, 1.0, 5, 1],
        [2, 1, 20.0, 20.0, 1.0, 30, 2],
    ])
    _assert_legal(obs, _mcts_mod.mcts_plan(obs, num_players=2, budget_s=0.2))


def test_leaf_estimator_matches_lookahead_horizon():
    # The search core must agree with the *proven* lookahead leaf at depth 1:
    # one expansion step + ROLLOUT_TURNS == lookahead's horizon-10 leaf.
    pytest.importorskip("kaggle_environments")
    from src import worldmodel as wm
    la = importlib.import_module("src.agents.lookahead")
    assert _mcts_mod.ROLLOUT_TURNS + 1 == la.LOOKAHEAD_TURNS

    obs = _obs(0, [
        [0, 0, 90.0, 90.0, 1.0, 80, 1],
        [1, -1, 60.0, 90.0, 1.0, 5, 1],
        [2, 1, 30.0, 30.0, 1.0, 40, 1],
    ])
    base = wm.from_obs(obs, 2)
    for mv in la._candidate_moves(obs):
        # lookahead's leaf for mv
        fs = wm.step(base, [mv, []])
        fs = wm.rollout(fs, [None, None], la.LOOKAHEAD_TURNS - 1)
        la_val = la._leaf_value(fs, 0)
        # mcts's leaf for the same mv (do-nothing opponent)
        child = _mcts_mod._apply(base, 0, mv, 2, None)
        mc_val = _mcts_mod._rollout_value(child, 0, 2)
        assert mc_val == la_val


def test_opponent_policy_hook_moves_rivals():
    # The AG10-enabling hook: with an opponent policy, rivals actually act during
    # the simulation; with None they hold. Distinguish via the resulting fleets.
    pytest.importorskip("kaggle_environments")
    from src import worldmodel as wm
    from src.agents.roi_greedy_predict import plan_turn as v1

    # Player 1 has a strong home next to a cheap neutral -> v1 will launch.
    obs = _obs(0, [
        [0, 0, 90.0, 90.0, 1.0, 40, 1],   # my home
        [1, 1, 20.0, 20.0, 1.0, 80, 1],   # opponent home (will launch under v1)
        [2, -1, 35.0, 20.0, 1.0, 5, 1],   # cheap neutral near the opponent
    ])
    base = wm.from_obs(obs, 2)
    hold = _mcts_mod._apply(base, 0, [], 2, None)            # everyone holds
    modeled = _mcts_mod._apply(base, 0, [], 2, v1)           # opponent plays v1
    assert len(wm.fleets_of(hold)) == 0
    assert len(wm.fleets_of(modeled)) >= 1                   # rival actually launched


def test_budget_is_respected():
    # Anytime: a turn must finish near the budget, not run away. Warm the
    # interpreter import first (one-time, ~seconds locally; free on Kaggle).
    pytest.importorskip("kaggle_environments")
    from src import worldmodel as wm
    obs = _obs(0, [
        [0, 0, 90.0, 90.0, 1.0, 80, 1],
        [1, -1, 60.0, 90.0, 1.0, 5, 1],
        [2, 1, 30.0, 30.0, 1.0, 40, 1],
    ])
    wm.step(wm.from_obs(obs, 2), [[], []])  # warm import
    t0 = time.monotonic()
    _mcts_mod.mcts_plan(obs, num_players=2, budget_s=0.2)
    elapsed = time.monotonic() - t0
    assert elapsed < 0.6, f"budget overrun: {elapsed:.3f}s for a 0.2s budget"
