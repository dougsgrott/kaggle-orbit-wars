"""
M1c "roi_ledger" brain — arrival-ledger planning (don't double-spend).

Same motion-aim + ROI + defense reserve as v1 `roi_greedy_predict`, but it stops
v1's biggest measured waste: re-deciding every target from scratch each turn,
blind to its own fleets already in flight. A v1-vs-Boss diagnostic (seed 3001)
showed **61% of v1's launches hit a planet it already had fleets flying toward**
(1234 redundant ships) while uncovered planets sat free — so it gets
out-expanded.

The fix is an **arrival ledger**: once per turn, map each planet to the fleets
(mine + enemy) currently inbound to it, with their ETA, owner, and size. Then,
when planning launches:
  * **Skip** a target my in-flight fleets already capture on their own — no point
    piling on; that force should expand elsewhere.
  * **Size net of in-flight**: feed the ledger as `scheduled_arrivals` to
    `ships_needed_to_capture_timeline`, so the launch credits friendly fleets en
    route (smaller `need`) and debits incoming enemy waves (larger `need`).

This mirrors the Boss's `simulate_timeline` at small scale, reusing utils — no
custom Engine, no cross-turn state (in-flight fleets live in the obs). Pure
`plan_turn(obs, config)`; per-planet greedy allocation is kept (allocation tuning
was already measured neutral in AG5 — this slice isolates the ledger). Every Shot
legal and motion-verified.

A fleet's implicit target is recovered with `find_target_via_ray` and its ETA
estimated as remaining-distance / `fleet_speed` (a small approximation: fleets
fly straight at constant size-scaled speed, exactly the interpreter's model).

Public API:
    plan_turn(obs, config=None)   -> list[list]
    build_arrival_ledger(me, planets, fleets) -> dict[int, list[(eta, owner, ships)]]
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

from ..utils import (
    aim_with_prediction,
    dist,
    fleet_speed,
    ships_needed_to_capture_timeline,
)
from ..features import find_target_via_ray
from .roi_greedy import SHIP_BUFFER, _field, _inbound_threats, defense_reserve
from .roi_greedy_predict import _build_world


def build_arrival_ledger(me: int, planets, fleets) -> Dict[int, List[Tuple[int, int, int]]]:
    """Map planet id -> list of (eta_turn, fleet_owner, ships) for every fleet
    currently in flight toward it. The target is implicit, recovered by ray-cast;
    ETA = ceil(remaining distance / fleet_speed(ships)), min 1."""
    by_id = {int(p[0]): p for p in planets}
    ledger: Dict[int, List[Tuple[int, int, int]]] = {}
    for f in fleets:
        fx, fy, fang, fships = float(f[2]), float(f[3]), float(f[4]), int(f[6])
        tid = find_target_via_ray((fx, fy), fang, planets)
        if tid < 0 or tid not in by_id:
            continue
        tgt = by_id[tid]
        d = dist(fx, fy, float(tgt[2]), float(tgt[3]))
        eta = max(1, int(math.ceil(d / fleet_speed(max(1, fships)))))
        ledger.setdefault(tid, []).append((eta, int(f[1]), fships))
    return ledger


def plan_turn(obs, config=None) -> List[list]:
    """Return this turn's Shots, accounting for in-flight fleets. See module docstring."""
    me = int(_field(obs, "player"))
    planets = _field(obs, "planets")
    fleets = _field(obs, "fleets")

    world = _build_world(obs)
    threats = _inbound_threats(me, planets, fleets)
    ledger = build_arrival_ledger(me, planets, fleets)

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

            aim = aim_with_prediction(sx, sy, sr, tid, tx, ty, tr, spendable, **world)
            if aim is None:
                continue  # no sun-safe shot lands — skip rather than waste a fleet.
            turns = aim[1]

            inbound = ledger.get(tid, [])
            # (a) Skip targets my own fleets already capture — those ships are
            #     committed; this planet's budget belongs elsewhere.
            mine_inbound = [(eta, o, s) for (eta, o, s) in inbound if o == me]
            if mine_inbound:
                horizon = max(turns, max(eta for eta, _, _ in mine_inbound))
                if ships_needed_to_capture_timeline(
                    t_ships, t_owner, t_prod, me, horizon,
                    scheduled_arrivals=mine_inbound,
                ) == 0:
                    continue

            # (b) Size net of ALL in-flight (friendly credit + enemy debit).
            need = ships_needed_to_capture_timeline(
                t_ships, t_owner, t_prod, me, turns, scheduled_arrivals=inbound
            )
            send = min(spendable, need + SHIP_BUFFER)
            if send < need:
                continue  # can't afford the capture after the reserve.

            score = t_prod / max(1, need) / max(1, turns)
            if best is None or score > best[0]:
                best = (score, tid, tx, ty, tr, send)

        if best is None:
            continue

        # Re-aim the chosen target with the actual send count (moving-target
        # correctness — the AG4 lesson). Skip if the sized fleet has no intercept.
        _, tid, tx, ty, tr, send = best
        refined = aim_with_prediction(sx, sy, sr, tid, tx, ty, tr, send, **world)
        if refined is None:
            continue
        moves.append([sid, float(refined[0]), int(send)])

    return moves
