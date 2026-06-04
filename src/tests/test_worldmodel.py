"""Tests for the WorldModel forward model (M3, ADR-0003).

Pure-board transition tests need no kaggle_environments... except `step` calls
the interpreter, so these require the env (skipped otherwise, like the arena
integration tests). The conformance test asserts WorldModel byte-matches a fresh
interpreter advance for a fixed seed.
"""
from types import SimpleNamespace

import pytest

from src import worldmodel as wm


def _obs(planets, fleets=(), **kw):
    o = {
        "player": 0,
        "planets": [list(p) for p in planets],
        "fleets": [list(f) for f in fleets],
        "initial_planets": [list(p) for p in planets],
        "angular_velocity": 0.0,
        "comets": [],
        "comet_planet_ids": [],
    }
    o.update(kw)
    return o


def test_from_obs_does_not_mutate_input_and_sets_next_fleet_id():
    obs = _obs([[0, 0, 90.0, 90.0, 1.0, 80, 1]], fleets=[[3, 0, 50.0, 50.0, 0.0, 0, 5]])
    fs = wm.from_obs(obs, num_players=2)
    # next_fleet_id derived above the max existing fleet id (3) -> 4.
    assert fs.state[0].observation.next_fleet_id == 4
    # Source obs is untouched and the forward state is an independent copy.
    fs.state[0].observation.planets[0][5] = 999
    assert obs["planets"][0][5] == 80


def test_score_reuses_score_board():
    obs = _obs([[0, 0, 0, 0, 1, 10, 1], [1, 1, 0, 0, 1, 4, 1]],
               fleets=[[0, 0, 0, 0, 0.0, 0, 3]])
    fs = wm.from_obs(obs, num_players=2)
    # player 0: 10 + 3 fleet = 13 ; player 1: 4
    assert wm.score(fs) == [13, 4]


# --- env-backed: interpreter transitions ------------------------------------


def test_step_applies_production_tick():
    pytest.importorskip("kaggle_environments")
    # Lone owned planet, production 4: one step adds production (no combat).
    obs = _obs([[0, 0, 90.0, 90.0, 1.0, 10, 4]])
    fs = wm.from_obs(obs, num_players=2)
    fs1 = wm.step(fs, [[], []])
    assert wm.planets_of(fs1)[0][5] == 14  # 10 + 4
    assert wm.step_of(fs1) == 1
    # original forward state unchanged (step returns a fresh copy)
    assert wm.step_of(fs) == 0


def test_step_simulates_a_capture():
    pytest.importorskip("kaggle_environments")
    import math
    obs = _obs([[0, 0, 90.0, 90.0, 1.0, 80, 1], [1, -1, 80.0, 90.0, 1.0, 5, 1]])
    fs = wm.from_obs(obs, num_players=2)
    # Launch 8 ships west at the weak neutral, then let the fleet arrive.
    fs = wm.step(fs, [[[0, math.pi, 8]], []])
    for _ in range(25):
        fs = wm.step(fs, [[], []])
    owners = {int(p[0]): int(p[1]) for p in wm.planets_of(fs)}
    assert owners.get(1) == 0  # neutral captured by me


def test_rollout_with_do_nothing_policies_advances_turns():
    pytest.importorskip("kaggle_environments")
    obs = _obs([[0, 0, 90.0, 90.0, 1.0, 10, 2]])
    fs = wm.from_obs(obs, num_players=2)
    fs2 = wm.rollout(fs, policies=[None, None], turns=5)
    assert wm.step_of(fs2) == 5
    assert wm.planets_of(fs2)[0][5] == 10 + 2 * 5  # production each turn


def test_conformance_byte_matches_fresh_interpreter():
    pytest.importorskip("kaggle_environments")
    from kaggle_environments.envs.orbit_wars.orbit_wars import interpreter

    def fresh_init(seed, n=2):
        st = [SimpleNamespace(observation=SimpleNamespace(step=0), action=[],
                              status="ACTIVE", reward=0)]
        for i in range(1, n):
            st.append(SimpleNamespace(observation=SimpleNamespace(player=i), action=[],
                                      status="ACTIVE", reward=0))
        e = SimpleNamespace(
            configuration=SimpleNamespace(shipSpeed=6.0, episodeSteps=500,
                                          cometSpeed=4.0, seed=seed),
            done=False, info={})
        interpreter(st, e)
        return st, e

    ref, refe = fresh_init(7)
    snap, _ = fresh_init(7)  # independent same-seed board for WM's step-0 snapshot
    b = snap[0].observation
    obs0 = {
        "planets": [list(p) for p in b.planets], "fleets": [],
        "initial_planets": [list(p) for p in b.initial_planets],
        "angular_velocity": b.angular_velocity, "comets": [],
        "comet_planet_ids": list(b.comet_planet_ids),
        "next_fleet_id": b.next_fleet_id, "step": 0,
    }
    fs = wm.from_obs(obs0, num_players=2)
    for t in range(1, 6):
        for s in ref:
            s.observation.step = t
        interpreter(ref, refe)
        fs = wm.step(fs, [[], []])
        assert [list(p) for p in ref[0].observation.planets] == \
               [list(p) for p in wm.planets_of(fs)], f"divergence at t={t}"
