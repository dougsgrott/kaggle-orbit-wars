"""
v0 "ROI-greedy" agent brain — the project's first decision core (PRD M0 / AG1).

Given a single observation, decide which **Shots** to launch this turn. A Shot is
`[from_planet_id, angle, num_ships]`; a turn's action is a list of Shots. This
module is a *pure* function of the observation (no I/O, no randomness, no
module-level mutable state) so it is unit-testable on synthetic observations
without `kaggle_environments`. It is one candidate brain among potentially many
in `src/agents/` (see that package's docstring); wiring the current default into
the Kaggle entry point and measuring it is AG2.

Strategy (greedy, per-planet, multi-enemy-aware):
  * Targets are every planet not owned by me (neutral + *every* non-me owner, so
    the same logic works in 1v1 and 4P FFA).
  * Each of my planets keeps a **defense reserve** sized from the actual inbound
    enemy threat via combat resolution (`defense_reserve`), and only spends what
    is left over.
  * For each spendable planet, score every reachable target by ETA-discounted
    return-on-investment `production / ships_needed / eta`, and launch a single
    Shot at the best one, sized by the production-aware capture estimate plus a
    small buffer.
  * Every Shot is routed through the verified, sun-safe aim solver
    (`utils.aim_with_prediction`); a target with no safe shot is skipped, so the
    agent never throws a fleet at the sun and never knowingly misses.

All physics / aim / combat / capture math comes from `utils` (PRD: nothing is
re-derived here).

Static-world approximation (deliberate v0 simplification — confirmed decision):
    The observation *does* expose `angular_velocity`, `initial_planets`,
    `comets`, and `comet_planet_ids`, so full orbital/comet motion is available.
    v0 nonetheless aims at every planet's *current* position by passing an empty
    world (`initial_by_id={}`, `comets=[]`, `comet_ids=set()`) to the aim solver
    — with no initial positions, `predict_*` degenerates to the current position,
    so the solver reduces to verified straight-line aim at where the planet is
    now. This keeps the core minimal and matches the existing aim tests and the
    corpus opponents (which all treat planets as static); the map also guarantees
    at least three static planet groups. Feeding the real motion to the solver
    (for orbiting planets and comets) is a later (M1) upgrade. A welcome
    consequence: for a static target the returned aim *angle* is purely geometric
    (independent of the ship count), so we can size the fleet after solving the
    angle without re-aiming.

Decision schema:
    Input:  obs  -- dict-or-attr with "player" (int), "planets"
                    ([id, owner, x, y, radius, ships, production]), and "fleets"
                    ([id, owner, x, y, angle, from_planet_id, ships]).
            config -- accepted and ignored (Kaggle entry-point signature parity).
    Output: list of Shots [from_planet_id, angle, num_ships]; [] when nothing is
            affordable or safe. Every Shot is legal by construction.

Public API:
    plan_turn(obs, config=None)                 -> list[list]
    defense_reserve(planet_owner, inbound_fleets) -> int   (pure)
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

from ..utils import (
    aim_with_prediction,
    resolve_combat,
    ships_needed_to_capture_timeline,
)
from ..features import find_target_via_ray

# A small margin added on top of the capture estimate so the agent stops
# chronically undershooting while the target keeps producing (story 25).
SHIP_BUFFER = 2

# Empty "static world" passed to the aim solver — see the module docstring.
_STATIC_WORLD = {
    "initial_by_id": {},
    "angular_velocity": 0.0,
    "comets": [],
    "comet_ids": set(),
}


def _field(obs, key):
    """Read `key` from an observation that may be a dict or an attribute bag
    (the Kaggle env passes a Struct; opponents handle both — we mirror them)."""
    if isinstance(obs, dict):
        return obs[key]
    return getattr(obs, key)


def defense_reserve(
    planet_owner: int, inbound_fleets: Sequence[Tuple[int, int]]
) -> int:
    """Minimum garrison `planet_owner` must keep to still own the planet after
    the given enemy fleets land, computed through the engine's own
    `resolve_combat` (story 28).

    `inbound_fleets` is a list of `(owner, ships)` for *enemy* fleets treated as
    co-arriving in a single worst-case wave (v0 ignores arrival timing and the
    planet's own production accrual — a deliberate simplification).

    Returns 0 when there is no threat and is non-decreasing as the threat grows.
    The "holds" predicate is monotonic in the garrison, so we binary-search the
    smallest garrison `g` for which the planet's owner is unchanged.
    """
    fleets = [(int(o), int(s)) for o, s in inbound_fleets if int(s) > 0]
    if not fleets:
        return 0
    if resolve_combat(planet_owner, 0, fleets)[0] == planet_owner:
        return 0  # attackers annihilate each other (e.g. FFA tie) — no reserve.
    lo, hi = 1, sum(s for _, s in fleets) + 1
    while lo < hi:
        mid = (lo + hi) // 2
        if resolve_combat(planet_owner, mid, fleets)[0] == planet_owner:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _inbound_threats(me: int, planets, fleets) -> dict:
    """Map each of my planet ids to the list of `(owner, ships)` enemy fleets
    inbound to it. A fleet's target is implicit, so we recover it by ray-casting
    from the fleet along its heading (`features.find_target_via_ray`)."""
    threats: dict = {}
    for f in fleets:
        f_owner = int(f[1])
        if f_owner < 0 or f_owner == me:
            continue  # neutral fleets don't exist; skip my own.
        target_id = find_target_via_ray((float(f[2]), float(f[3])), float(f[4]), planets)
        if target_id >= 0:
            threats.setdefault(int(target_id), []).append((f_owner, int(f[6])))
    return threats


def plan_turn(obs, config=None) -> List[list]:
    """Return this turn's list of Shots `[from_planet_id, angle, num_ships]`.

    See the module docstring for the full decision schema and strategy.
    """
    me = int(_field(obs, "player"))
    planets = _field(obs, "planets")
    fleets = _field(obs, "fleets")

    threats = _inbound_threats(me, planets, fleets)
    my_planets = [p for p in planets if int(p[1]) == me]

    moves: List[list] = []
    for src in my_planets:
        sid = int(src[0])
        sx, sy, sr = float(src[2]), float(src[3]), float(src[4])
        spendable = int(src[5]) - defense_reserve(me, threats.get(sid, []))
        if spendable < 1:
            continue

        best = None  # (score, angle, send)
        for t in planets:
            t_owner = int(t[1])
            if t_owner == me:
                continue  # only non-owned planets are targets (neutral + enemies).
            tid = int(t[0])
            tx, ty, tr = float(t[2]), float(t[3]), float(t[4])
            t_ships, t_prod = int(t[5]), int(t[6])

            aim = aim_with_prediction(
                sx, sy, sr, tid, tx, ty, tr, spendable, **_STATIC_WORLD
            )
            if aim is None:
                continue  # no sun-safe shot lands — skip rather than waste a fleet.
            angle, turns = aim[0], aim[1]

            need = ships_needed_to_capture_timeline(
                t_ships, t_owner, t_prod, me, turns
            )
            send = min(spendable, need + SHIP_BUFFER)
            if send < need:
                continue  # can't afford the capture after the reserve.

            score = t_prod / max(1, need) / max(1, turns)
            if best is None or score > best[0]:
                best = (score, angle, send)

        if best is not None:
            moves.append([sid, float(best[1]), int(best[2])])

    return moves
