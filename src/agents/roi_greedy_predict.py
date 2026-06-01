"""
v1 "ROI-greedy + motion-aware aim" brain (issue AG4).

Identical strategy to the v0 `roi_greedy` brain — greedy, per-planet, multi-enemy
aware, defense-reserved, ETA-discounted ROI — with one change: it feeds the
verified aim solver the **real planet/comet motion** the observation exposes,
instead of v0's empty static world. That single change is the measured #1 fix:
v0 aimed at planets' *current* positions while ~half the planets orbit
(`angular_velocity` up to 0.05; home planets can orbit), so its multi-turn fleets
arrived where the target *was* — submission_01 missed 41% of all fleets and
captured nothing until turn 224 (see wiki/measured_log.md).

`aim_with_prediction` already accepts `initial_by_id` / `angular_velocity` /
`comets` / `comet_ids` and forward-verifies every returned shot against the real
motion; v0 simply starved it. Here we build that world once per turn from the obs
and pass it to every aim call.

Ship-count subtlety (why we re-aim before emitting):
    For a *moving* target the solver's angle and arrival turn depend on the fleet
    size, because `fleet_speed(ships)` sets how far the fleet travels per turn and
    therefore where it intercepts. We size the fleet (`send`) *after* an initial
    aim, so the emitted shot is re-aimed with the actual `send` count — otherwise
    a slower, smaller fleet would be verified at the wrong speed and miss. On a
    *static* target the solver's angle is purely geometric (ship-count
    independent), so this re-aim is a no-op and the output is byte-identical to
    `roi_greedy` (the parity the tests lock in).

This is a separate coexisting brain (ADR-0002), A/B-tested against `roi_greedy`
via the registry — not a rewrite of v0. All physics/aim/combat comes from
`utils`; reserve/threat/ROI helpers are reused from `roi_greedy` (nothing
re-derived).

Decision schema and output are exactly as `roi_greedy.plan_turn`.

Public API:
    plan_turn(obs, config=None)   -> list[list]
"""
from __future__ import annotations

from typing import List

from ..utils import aim_with_prediction, ships_needed_to_capture_timeline
from .roi_greedy import (
    SHIP_BUFFER,
    _field,
    _inbound_threats,
    defense_reserve,
)


def _field_opt(obs, key, default):
    """Read an *optional* obs field (dict or attr bag), falling back to `default`
    when absent — synthetic test observations may omit the motion fields, in
    which case the world is empty and aim degenerates to the static case."""
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _build_world(obs) -> dict:
    """Assemble the motion world `aim_with_prediction` needs from the obs.

    `initial_by_id` maps planet id -> {"x","y"} of its *start* position (the
    solver uses it to decide static-vs-orbiting and to rotate orbiting planets);
    `angular_velocity`, `comets`, and `comet_ids` drive planet/comet prediction.
    Missing fields collapse to the static world (empty), so this is always safe.
    """
    initial = _field_opt(obs, "initial_planets", []) or []
    initial_by_id = {
        int(p[0]): {"x": float(p[2]), "y": float(p[3])} for p in initial
    }
    return {
        "initial_by_id": initial_by_id,
        "angular_velocity": float(_field_opt(obs, "angular_velocity", 0.0) or 0.0),
        "comets": _field_opt(obs, "comets", []) or [],
        "comet_ids": {int(c) for c in (_field_opt(obs, "comet_planet_ids", []) or [])},
    }


def plan_turn(obs, config=None) -> List[list]:
    """Return this turn's list of Shots `[from_planet_id, angle, num_ships]`,
    aiming with real planet/comet motion. See the module docstring."""
    me = int(_field(obs, "player"))
    planets = _field(obs, "planets")
    fleets = _field(obs, "fleets")

    world = _build_world(obs)
    threats = _inbound_threats(me, planets, fleets)
    my_planets = [p for p in planets if int(p[1]) == me]

    moves: List[list] = []
    for src in my_planets:
        sid = int(src[0])
        sx, sy, sr = float(src[2]), float(src[3]), float(src[4])
        spendable = int(src[5]) - defense_reserve(me, threats.get(sid, []))
        if spendable < 1:
            continue

        best = None  # (score, tid, tx, ty, tr, send)
        for t in planets:
            t_owner = int(t[1])
            if t_owner == me:
                continue  # only non-owned planets are targets (neutral + enemies).
            tid = int(t[0])
            tx, ty, tr = float(t[2]), float(t[3]), float(t[4])
            t_ships, t_prod = int(t[5]), int(t[6])

            aim = aim_with_prediction(
                sx, sy, sr, tid, tx, ty, tr, spendable, **world
            )
            if aim is None:
                continue  # no sun-safe shot lands — skip rather than waste a fleet.
            turns = aim[1]

            need = ships_needed_to_capture_timeline(
                t_ships, t_owner, t_prod, me, turns
            )
            send = min(spendable, need + SHIP_BUFFER)
            if send < need:
                continue  # can't afford the capture after the reserve.

            score = t_prod / max(1, need) / max(1, turns)
            if best is None or score > best[0]:
                best = (score, tid, tx, ty, tr, send)

        if best is None:
            continue

        # Re-aim the chosen target with the fleet size we will actually send, so
        # the emitted shot is verified at the right speed (no-op on static
        # targets; the correctness fix on moving ones). Skip if the sized fleet
        # has no verified intercept.
        _, tid, tx, ty, tr, send = best
        refined = aim_with_prediction(sx, sy, sr, tid, tx, ty, tr, send, **world)
        if refined is None:
            continue
        moves.append([sid, float(refined[0]), int(send)])

    return moves
