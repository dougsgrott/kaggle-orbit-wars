"""aim_with_prediction verification + Wilson CI sanity."""

import math

from src.utils import (
    aim_with_prediction, _verify_shot_hits, _aim_raw,
    estimate_arrival,
)
from src.eval import wilson_ci


def _empty_world():
    # Static-planet world: empty initial_by_id (predict_planet_position returns
    # current position when planet_id missing) + no comets. All "predictions"
    # then degenerate to current position, which is fine for an aim test.
    return {"initial_by_id": {}, "angular_velocity": 0.035,
            "comets": [], "comet_ids": set()}


def test_aim_clear_path_returns_verified_shot():
    w = _empty_world()
    # Source at (10, 50), target at (90, 50). Sun at (50,50) blocks straight line.
    # So a *straight* shot returns None → aim should also return None for this geometry.
    result = aim_with_prediction(
        sx=10.0, sy=50.0, sr=1.0,
        target_id=99,
        tx=90.0, ty=50.0, tr=1.0,
        ships=50,
        **w)
    assert result is None


def test_aim_off_axis_returns_hit():
    w = _empty_world()
    # Bottom edge of the board: (10,10) → (80,10). Perpendicular distance from
    # the sun centre (50,50) to this segment is 40 — well outside the safety
    # buffer of 11.5.
    result = aim_with_prediction(
        sx=10.0, sy=10.0, sr=1.0,
        target_id=99,
        tx=80.0, ty=10.0, tr=2.0,
        ships=100,
        **w)
    assert result is not None
    angle, turns, _, _ = result
    est = estimate_arrival(10.0, 10.0, 1.0, 80.0, 10.0, 2.0, 100)
    assert est is not None
    assert abs(angle - est[0]) < 0.2  # close to direct aim


def test_verify_gate_can_reject():
    # Construct an obviously-bogus shot: aim 180° away from target.
    w = _empty_world()
    ok = _verify_shot_hits(
        sx=20.0, sy=20.0, sr=1.0,
        angle=math.pi,         # wrong direction
        turns=5, ships=50,
        target_id=99,
        tx=80.0, ty=80.0, tr=2.0,
        **w)
    assert ok is False


def test_aim_raw_then_verify_gate_consistency():
    """The public aim_with_prediction must NEVER return a shot that
    _verify_shot_hits would reject — that's the v7 contract."""
    w = _empty_world()
    cases = [
        (10.0, 10.0, 1.0, 80.0, 10.0, 2.0, 100),
        (10.0, 10.0, 1.0, 90.0, 30.0, 1.5, 60),
        (50.0, 5.0,  1.0, 90.0, 5.0,  3.0, 200),
    ]
    for sx, sy, sr, tx, ty, tr, ships in cases:
        result = aim_with_prediction(sx, sy, sr, 99, tx, ty, tr, ships, **w)
        if result is None:
            continue
        angle, turns, _, _ = result
        assert _verify_shot_hits(sx, sy, sr, angle, turns, ships,
                                 99, tx, ty, tr, **w), (
            f"aim_with_prediction returned an UNverified shot for "
            f"src=({sx},{sy}) tgt=({tx},{ty}) ships={ships}")


def test_wilson_ci_known_values():
    # 7/10 wins → point estimate 0.7, CI roughly [0.40, 0.89].
    p, lo, hi = wilson_ci(7, 10)
    assert p == 0.7
    assert 0.35 < lo < 0.45
    assert 0.85 < hi < 0.92
    # Edge: n=0 returns zeros.
    p, lo, hi = wilson_ci(0, 0)
    assert (p, lo, hi) == (0.0, 0.0, 0.0)
