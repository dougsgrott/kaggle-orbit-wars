"""Sun-blocking, static-planet detection, combat resolution."""

from src.utils import (
    segment_hits_sun, is_path_clear,
    is_static_planet, orbital_radius,
    resolve_combat,
    ships_needed_to_capture_simple,
    ships_needed_to_capture_timeline,
)


def test_segment_through_sun_blocks():
    # Horizontal line from (0,50) to (100,50) passes through sun centre.
    assert segment_hits_sun(0.0, 50.0, 100.0, 50.0)
    assert not is_path_clear(0.0, 50.0, 100.0, 50.0)


def test_segment_along_edge_clear():
    # Top edge of the board — nowhere near the sun.
    assert not segment_hits_sun(0.0, 0.0, 0.0, 100.0)
    assert is_path_clear(0.0, 0.0, 0.0, 100.0)


def test_static_planet_threshold():
    # Planet at (50, 90) → orbital_radius=40, + r=11 → 51 >= 50 → static.
    assert is_static_planet(50.0, 90.0, 11.0)
    # Planet at (50, 60) → orbital_radius=10, + r=1 → 11 < 50 → rotating.
    assert not is_static_planet(50.0, 60.0, 1.0)


def test_orbital_radius_known_points():
    assert orbital_radius(50.0, 50.0) == 0.0
    assert abs(orbital_radius(50.0, 90.0) - 40.0) < 1e-9


def test_resolve_combat_capture():
    # Defender P1 owns 20 ships; attacker P0 sends 50. Survivor=50, defender=20 → captured with 30.
    new_owner, new_ships = resolve_combat(1, 20, [(0, 50)])
    assert new_owner == 0 and new_ships == 30


def test_resolve_combat_failed():
    new_owner, new_ships = resolve_combat(1, 60, [(0, 50)])
    assert new_owner == 1 and new_ships == 10


def test_resolve_combat_tie():
    # Single attacker exact tie → defender holds with 0 (planet_ships - survivor_ships).
    new_owner, new_ships = resolve_combat(1, 50, [(0, 50)])
    assert new_owner == 1 and new_ships == 0


def test_resolve_combat_ffa_tie_neutralises():
    # Two equal top attackers → survivor=0 → planet unchanged.
    new_owner, new_ships = resolve_combat(1, 10, [(0, 80), (2, 80)])
    assert new_owner == 1 and new_ships == 10


def test_resolve_combat_reinforce_own():
    new_owner, new_ships = resolve_combat(0, 30, [(0, 50)])
    assert new_owner == 0 and new_ships == 80


def test_ships_needed_simple_basic():
    # Defender owns 20 ships, attacker is owner 0. Capture needs ≥ 21.
    n = ships_needed_to_capture_simple(20, 1, 0)
    assert n == 21


def test_ships_needed_simple_already_owned():
    # Attacker already owns the planet and no other arrivals → 0 ships needed.
    n = ships_needed_to_capture_simple(20, 0, 0)
    assert n == 0


def test_ships_needed_timeline_production():
    # Defender 0 ships, production 2/turn, arrival_turn=10 → planet has 18 by t=10
    # (production accrues from t=2). Attacker needs 19.
    n = ships_needed_to_capture_timeline(
        defender_ships=0, defender_owner=1, defender_production=2,
        attacker_owner=0, arrival_turn=10)
    assert n == 19


def test_ships_needed_timeline_neutral_no_production():
    # Neutral defender (owner=-1) does NOT produce ships in resolve_combat
    # accounting; need just enough to overcome current ships.
    n = ships_needed_to_capture_timeline(
        defender_ships=15, defender_owner=-1, defender_production=3,
        attacker_owner=0, arrival_turn=5)
    assert n == 16
