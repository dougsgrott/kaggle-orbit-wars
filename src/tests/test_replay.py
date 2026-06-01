"""Replay (game-trace debugger) tests — pure death-cause inference only.

These never import matplotlib or kaggle_environments: rendering is a by-eye dev
tool, but the cause inference is real physics and worth pinning down. Fleets are
[id, owner, x, y, angle, from_planet_id, ships]; planets are
[id, owner, x, y, radius, ships, production].
"""

import math

from src.arena import Frame
from src.replay import (
    COMBAT,
    OUT_OF_BOUNDS,
    SUN,
    FleetDeath,
    fleet_deaths,
    infer_death_cause,
)


def test_infer_sun():
    # Sits at distance 12 from center (outside the sun), heading inward; one
    # step dips within the 10-unit sun radius.
    fleet = [1, 0, 50.0, 38.0, math.pi / 2, -1, 50]
    cause, _, _ = infer_death_cause(fleet, planets=[])
    assert cause == SUN


def test_infer_out_of_bounds():
    # Near the right edge, heading +x, no planet in the way -> leaves the board.
    fleet = [1, 0, 98.0, 50.0, 0.0, -1, 50]
    cause, x, _ = infer_death_cause(fleet, planets=[])
    assert cause == OUT_OF_BOUNDS
    assert x > 100.0


def test_infer_combat_on_planet_hit():
    fleet = [1, 0, 55.0, 50.0, 0.0, -1, 500]  # fast, heading +x into the planet
    planets = [[9, -1, 60.0, 50.0, 2.0, 30, 1]]
    cause, x, y = infer_death_cause(fleet, planets)
    assert cause == COMBAT
    assert (x, y) == (60.0, 50.0)  # death position snaps to the struck planet


def test_combat_takes_precedence_over_out_of_bounds():
    # Would fly off-board, but a planet is in the path first -> combat, not OOB.
    fleet = [1, 0, 95.0, 50.0, 0.0, -1, 500]
    planets = [[9, -1, 98.0, 50.0, 2.0, 30, 1]]
    cause, _, _ = infer_death_cause(fleet, planets)
    assert cause == COMBAT


def test_fleet_deaths_flags_vanished_fleet():
    prev = Frame(
        step=5,
        planets=[],
        fleets=[[1, 0, 50.0, 38.0, math.pi / 2, -1, 50],   # heads into the sun
                [2, 1, 10.0, 10.0, 0.0, 5, 9]],             # survives
        actions=[],
    )
    curr = Frame(step=6, planets=[], fleets=[[2, 1, 12.0, 10.0, 0.0, 5, 9]], actions=[])

    deaths = fleet_deaths(prev, curr)
    assert len(deaths) == 1
    assert deaths[0].fleet_id == 1
    assert deaths[0].cause == SUN
    assert deaths[0].missed is True


def test_fleet_death_missed_property():
    assert FleetDeath(1, 0, SUN, 0, 0, 5).missed is True
    assert FleetDeath(1, 0, OUT_OF_BOUNDS, 0, 0, 5).missed is True
    assert FleetDeath(1, 0, COMBAT, 0, 0, 5).missed is False
