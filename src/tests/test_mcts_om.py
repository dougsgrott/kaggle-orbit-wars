"""Tests for the opponent-model + 4P-aware MCTS brain (AG10, M4b).

`value_placement` is env-free (from_obs + score_board are pure), so its properties
test anywhere; the brain's `plan_turn` runs the search and needs kaggle_environments.
"""
import importlib
import math

import pytest

from src.agents import REGISTRY
from src.agents.mcts import value_placement
from src import worldmodel as wm

mcts_om = REGISTRY["mcts_om"]


def _fstate(planets, num_players):
    # planet schema: [id, owner, x, y, radius, ships, production]
    obs = {
        "player": 0,
        "planets": [list(p) for p in planets],
        "fleets": [],
        "initial_planets": [list(p) for p in planets],
        "angular_velocity": 0.0,
        "comets": [],
        "comet_planet_ids": [],
    }
    return wm.from_obs(obs, num_players=num_players)


def test_registered_in_registry():
    assert REGISTRY["mcts_om"] is mcts_om


# --- value_placement (pure) ---------------------------------------------------


def test_placement_value_1v1_orders_by_ships():
    ahead = _fstate([[0, 0, 0, 0, 1, 100, 1], [1, 1, 0, 0, 1, 50, 1]], 2)
    behind = _fstate([[0, 0, 0, 0, 1, 50, 1], [1, 1, 0, 0, 1, 100, 1]], 2)
    assert value_placement(ahead, 0) > value_placement(behind, 0)
    assert value_placement(ahead, 0) > 1.0          # 1st place (1.0) + share tiebreak
    assert value_placement(behind, 0) < 0.5         # last place ~0 + tiny tiebreak


def test_placement_value_4p_rewards_better_placement():
    # scores: p0=100 (1st), p1=80 (2nd), p2=60 (3rd), p3=40 (4th)
    fs = _fstate([
        [0, 0, 0, 0, 1, 100, 1],
        [1, 1, 0, 0, 1, 80, 1],
        [2, 2, 0, 0, 1, 60, 1],
        [3, 3, 0, 0, 1, 40, 1],
    ], 4)
    v1st, v2nd, v3rd, v4th = (value_placement(fs, i) for i in range(4))
    assert v1st > v2nd > v3rd > v4th
    assert math.isclose(round(v1st, 1), 1.0)        # (4-1)/3 = 1.0
    assert 0.3 < v3rd < 0.4                          # (4-3)/3 = 0.333


def test_placement_tiebreak_never_flips_placement():
    # A 2nd-place board with a huge ship share must still rank below any 1st-place
    # board, however thin its lead -> the 0.01 share term can't flip a placement.
    second_rich = _fstate([
        [0, 0, 0, 0, 1, 95, 1],    # me 2nd but ship-rich
        [1, 1, 0, 0, 1, 96, 1],    # leader by a hair
        [2, 2, 0, 0, 1, 1, 1],
        [3, 3, 0, 0, 1, 1, 1],
    ], 4)
    first_thin = _fstate([
        [0, 0, 0, 0, 1, 30, 1],    # me 1st by one ship
        [1, 1, 0, 0, 1, 29, 1],
        [2, 2, 0, 0, 1, 1, 1],
        [3, 3, 0, 0, 1, 1, 1],
    ], 4)
    assert value_placement(first_thin, 0) > value_placement(second_rich, 0)


# --- brain integration --------------------------------------------------------


def test_returns_legal_move():
    pytest.importorskip("kaggle_environments")
    obs = {
        "player": 0,
        "planets": [
            [0, 0, 90.0, 90.0, 1.0, 80, 1],
            [1, -1, 60.0, 90.0, 1.0, 5, 1],
            [2, 1, 20.0, 20.0, 1.0, 30, 2],
        ],
        "fleets": [],
        "initial_planets": [
            [0, 0, 90.0, 90.0, 1.0, 80, 1],
            [1, -1, 60.0, 90.0, 1.0, 5, 1],
            [2, 1, 20.0, 20.0, 1.0, 30, 2],
        ],
        "angular_velocity": 0.0, "comets": [], "comet_planet_ids": [],
    }
    moves = mcts_om(obs)
    by_id = {int(p[0]): p for p in obs["planets"]}
    for sid, angle, ships in moves:
        assert int(by_id[sid][1]) == 0
        assert 1 <= int(ships) <= int(by_id[sid][5])
        assert math.isfinite(angle)
