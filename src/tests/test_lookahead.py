"""Tests for the greedy lookahead brain (AG8, M3).

`plan_turn` simulates via the WorldModel (interpreter), so these need
kaggle_environments (skipped otherwise, like the worldmodel/arena tests). The
candidate-set construction is testable without the env.
"""
import importlib
import math

import pytest

from src.agents import REGISTRY

# The package binds `lookahead` to the plan_turn *function*, shadowing the
# submodule attribute — reach the module via importlib for its internals.
_la_mod = importlib.import_module("src.agents.lookahead")
lookahead = REGISTRY["lookahead"]


def _obs(player, planets, fleets=(), motion=True):
    o = {
        "player": player,
        "planets": [list(p) for p in planets],
        "fleets": [list(f) for f in fleets],
    }
    if motion:
        o["initial_planets"] = [list(p) for p in planets]
        o["angular_velocity"] = 0.0
        o["comets"] = []
        o["comet_planet_ids"] = []
    return o


def _assert_legal(obs, moves):
    me = obs["player"]
    by_id = {int(p[0]): p for p in obs["planets"]}
    for sid, angle, ships in moves:
        assert int(by_id[sid][1]) == me
        assert 1 <= int(ships) <= int(by_id[sid][5])
        assert math.isfinite(angle)


def test_candidate_set_dedupes_and_includes_hold():
    # Candidate construction is pure (no env). On a board with one obvious move,
    # the brains mostly agree, so we expect few unique candidates incl. hold ([]).
    obs = _obs(0, [[0, 0, 90.0, 90.0, 1.0, 80, 1], [1, -1, 60.0, 90.0, 1.0, 5, 1]])
    cands = _la_mod._candidate_moves(obs)
    # Deduped: identical brain proposals collapse; hold is present.
    assert [] in cands
    assert len(cands) == len({tuple(tuple(m) for m in c) for c in cands})


def test_no_targets_returns_empty():
    pytest.importorskip("kaggle_environments")
    assert lookahead(_obs(0, [[0, 0, 90.0, 90.0, 1.0, 50, 1]])) == []


def test_returns_legal_move():
    pytest.importorskip("kaggle_environments")
    obs = _obs(0, [
        [0, 0, 90.0, 90.0, 1.0, 80, 1],
        [1, -1, 60.0, 90.0, 1.0, 5, 1],
        [2, 1, 20.0, 20.0, 1.0, 30, 2],
    ])
    _assert_legal(obs, lookahead(obs))


def test_picks_move_with_better_simulated_future_than_greedy():
    # Budget-limited planet, a high-production neutral worth committing more force
    # to. One-turn greedy sizes the launch at need+buffer; lookahead, seeing the
    # production compound over K turns, commits more ships — a strictly higher
    # leaf score. We assert lookahead's chosen move's simulated value is >= every
    # other candidate's (its selection criterion) and that it diverges from v1.
    pytest.importorskip("kaggle_environments")
    from src import worldmodel as wm
    from src.agents.roi_greedy_predict import plan_turn as v1

    planets = [
        [0, 0, 90.0, 90.0, 1.0, 30, 1],
        [1, -1, 75.0, 90.0, 1.0, 5, 1],   # cheap, low production
        [2, -1, 90.0, 75.0, 1.0, 8, 5],   # high production — compounds
    ]
    obs = _obs(0, planets)
    chosen = lookahead(obs)
    _assert_legal(obs, chosen)

    # The chosen move maximises the K-turn leaf value over the candidate set.
    base = wm.from_obs(obs, 2)

    def leaf_value(mv):
        acts = [[], []]
        acts[0] = mv
        fs = wm.step(base, acts)
        fs = wm.rollout(fs, [None, None], _la_mod.LOOKAHEAD_TURNS - 1)
        sc = wm.score(fs)
        return sc[0] - sc[1]

    chosen_val = leaf_value(chosen)
    for cand in _la_mod._candidate_moves(obs):
        assert chosen_val >= leaf_value(cand) - 1e-9
    # And it genuinely diverges from one-turn greedy (commits more force here).
    assert chosen != v1(obs)
    assert chosen_val >= leaf_value(v1(obs))


def test_registered_in_registry():
    assert REGISTRY["lookahead"] is lookahead
