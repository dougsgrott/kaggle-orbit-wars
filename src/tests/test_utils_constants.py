"""Verify physics constants and the fleet_speed spec formula."""

import math

from src.utils import (
    BOARD_SIZE, CENTER_X, CENTER_Y, SUN_RADIUS, SUN_SAFETY,
    MAX_SHIP_SPEED, ROTATION_LIMIT, ROUTE_SEARCH_HORIZON,
    fleet_speed,
)


def test_constants_match_spec():
    assert BOARD_SIZE == 100.0
    assert CENTER_X == 50.0 and CENTER_Y == 50.0
    assert SUN_RADIUS == 10.0
    assert SUN_SAFETY == 1.5
    assert MAX_SHIP_SPEED == 6.0
    assert ROTATION_LIMIT == 50.0
    assert ROUTE_SEARCH_HORIZON >= 150  # v7 fix; must not be lowered


def test_fleet_speed_endpoints():
    assert fleet_speed(1) == 1.0
    assert math.isclose(fleet_speed(1000), 6.0, rel_tol=1e-9)
    assert fleet_speed(0) == 1.0          # clamp at low end
    assert fleet_speed(10_000) <= 6.0     # capped at MAX_SHIP_SPEED


def test_fleet_speed_monotonic():
    prev = fleet_speed(1)
    for n in (2, 5, 10, 50, 100, 500, 999, 1000):
        cur = fleet_speed(n)
        assert cur >= prev
        prev = cur
