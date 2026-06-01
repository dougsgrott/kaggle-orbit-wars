"""Pure tests for the motion-aware brain roi_greedy_predict (AG4).

No kaggle_environments. The contract: on static boards it is byte-identical to
roi_greedy; on orbiting targets it feeds real motion to the verified solver and
lands shots the static brain whiffs or skips. Coordinates here were confirmed
against src.utils (aim_with_prediction / _verify_shot_hits).
"""

from src.utils import aim_with_prediction, _verify_shot_hits
from src.agents.roi_greedy import plan_turn as static_brain
from src.agents.roi_greedy_predict import plan_turn as motion_brain, _build_world
from src.agents import REGISTRY


# planet row: [id, owner, x, y, radius, ships, production]   (owner -1 = neutral)
# fleet  row: [id, owner, x, y, angle, from_planet_id, ships]
def _obs(player, planets, fleets=(), motion=False, angular_velocity=0.05,
         comets=(), comet_ids=()):
    o = {
        "player": player,
        "planets": [list(p) for p in planets],
        "fleets": [list(f) for f in fleets],
    }
    if motion:
        o["initial_planets"] = [list(p) for p in planets]
        o["angular_velocity"] = angular_velocity
        o["comets"] = list(comets)
        o["comet_planet_ids"] = list(comet_ids)
    return o


# --- parity on static boards --------------------------------------------------


def test_parity_with_roi_greedy_on_static_board_no_motion_fields():
    # Static planets (orbital_radius + r >= 50), motion fields absent => the
    # motion brain's world collapses to empty, so decisions must match v0 exactly.
    obs = _obs(0, [[0, 0, 90.0, 90.0, 1.0, 80, 1], [1, -1, 5.0, 90.0, 1.0, 5, 1]])
    assert motion_brain(obs) == static_brain(obs)


def test_parity_with_roi_greedy_on_static_board_with_motion_fields():
    # Motion fields present but every planet is static => angle is geometric
    # (ship-count independent), so the output is still byte-identical to v0.
    obs = _obs(0, [[0, 0, 90.0, 90.0, 1.0, 80, 1], [1, -1, 5.0, 90.0, 1.0, 5, 1]],
               motion=True)
    assert motion_brain(obs) == static_brain(obs)


def test_no_targets_returns_empty():
    obs = _obs(0, [[0, 0, 90.0, 90.0, 1.0, 50, 1]], motion=True)
    assert motion_brain(obs) == []


# --- the fix: motion changes aim on orbiting targets --------------------------


def test_captures_orbiting_target_the_static_brain_skips():
    # Orbiting neutral at (50,35): there is no sun-safe *static* shot from the
    # corner (aim_with_prediction returns None for the static world), but a
    # motion-aware aim finds a verified intercept.
    planets = [[0, 0, 90.0, 90.0, 1.0, 80, 1], [1, -1, 50.0, 35.0, 1.0, 5, 1]]
    obs = _obs(0, planets, motion=True)
    assert static_brain(obs) == []           # v0 throws nothing — it can't aim
    moves = motion_brain(obs)
    assert len(moves) == 1                    # v1 lands a shot
    assert moves[0][0] == 0
    assert 1 <= moves[0][2] <= 80


def test_emitted_shot_is_verified_against_real_motion():
    # The shot v1 emits must actually land under the real motion (the whole point
    # of AG4): re-derive the verification the solver guarantees.
    planets = [[0, 0, 90.0, 90.0, 1.0, 80, 1], [1, -1, 60.0, 30.0, 1.0, 5, 1]]
    obs = _obs(0, planets, motion=True)
    moves = motion_brain(obs)
    assert len(moves) == 1
    sid, angle, ships = moves[0]
    src = planets[0]
    tgt = planets[1]
    world = _build_world(obs)
    # Find the arrival turn the solver settled on for this fleet size, then check
    # the emitted (angle, turns) lands against the moving target.
    aim = aim_with_prediction(
        src[2], src[3], src[4], int(tgt[0]),
        tgt[2], tgt[3], tgt[4], int(ships), **world,
    )
    assert aim is not None
    assert abs(angle - aim[0]) < 1e-9
    assert _verify_shot_hits(
        src[2], src[3], src[4], angle, aim[1], int(ships),
        int(tgt[0]), tgt[2], tgt[3], tgt[4], **world,
    )


def test_static_aim_would_have_missed_this_moving_target():
    # Guards the premise of the test above: the *static* aim for the same target
    # does NOT land once the target is actually moving.
    sx, sy, sr = 90.0, 90.0, 1.0
    tx, ty, tr = 60.0, 30.0, 1.0
    ships = 7
    static_world = {"initial_by_id": {}, "angular_velocity": 0.0,
                    "comets": [], "comet_ids": set()}
    real_world = {"initial_by_id": {99: {"x": tx, "y": ty}},
                  "angular_velocity": 0.05, "comets": [], "comet_ids": set()}
    a_static = aim_with_prediction(sx, sy, sr, 99, tx, ty, tr, ships, **static_world)
    assert a_static is not None
    # The static angle, flown against the real motion, misses.
    assert not _verify_shot_hits(
        sx, sy, sr, a_static[0], a_static[1], ships, 99, tx, ty, tr, **real_world
    )


# --- world builder + registry -------------------------------------------------


def test_build_world_maps_obs_fields():
    obs = _obs(0, [[7, 0, 10.0, 20.0, 1.0, 5, 1]], motion=True,
               angular_velocity=0.033, comet_ids=[7])
    w = _build_world(obs)
    assert w["initial_by_id"][7] == {"x": 10.0, "y": 20.0}
    assert w["angular_velocity"] == 0.033
    assert w["comet_ids"] == {7}


def test_build_world_defaults_to_empty_when_fields_absent():
    obs = _obs(0, [[0, 0, 10.0, 10.0, 1.0, 5, 1]])  # no motion fields
    w = _build_world(obs)
    assert w["initial_by_id"] == {}
    assert w["angular_velocity"] == 0.0
    assert w["comets"] == [] and w["comet_ids"] == set()


def test_registered_in_registry():
    assert REGISTRY["roi_greedy_predict"] is motion_brain
