"""
M1b "roi_defense" brain — active defense + recapture + mobilization.

v1 `roi_greedy_predict` loses to the Boss not for lack of aim (fixed) or offense
concentration (AG5 showed allocation alone doesn't help) but because, once
out-expanded, it **hoards ships and mounts no defense**: a recorded v1-vs-Boss
game collapses from ~9 planets to 0 by t100 while holding ~530 ships idle. This
brain keeps v1's motion-aim + ROI offense and adds three things the diag called
for:

  1. **Active reinforce.** A planet whose inbound enemy threat exceeds its
     garrison (it can't meet its own `defense_reserve`) is reinforced from the
     nearest friendly planet with spare ships, via a verified in-time shot —
     so threatened planets get held instead of silently lost.
  2. **Recapture** falls out of v1's existing behavior (just-lost planets are
     non-owned, hence normal targets); the real fix is not going silent, i.e.
  3. **Mobilization.** Stop hoarding: after reinforcing and taking affordable
     captures, spend each planet's remaining garrison down toward a small floor
     onto its best verified target. Idle ships win nothing.

A per-phase ordering acts as the budget gate (defense first, then ROI offense,
then mobilize the dregs) so defense never fully starves expansion and vice
versa. Reinforcement Shots target *my own* planets; everything else targets
non-owned planets. Pure; same `plan_turn(obs, config)` contract and legal Shots.

Coexisting brain (ADR-0002), A/B-tested vs v1 on the boss tier. Reuses v1/utils
helpers — nothing re-derived.

Public API:
    plan_turn(obs, config=None)   -> list[list]
"""
from __future__ import annotations

import math
from typing import List

from ..utils import aim_with_prediction, ships_needed_to_capture_timeline
from .roi_greedy import SHIP_BUFFER, _field, _inbound_threats, defense_reserve
from .roi_greedy_predict import _build_world

# Keep at least this many ships home when mobilizing idle garrisons, so a planet
# is never stripped completely bare on a whim (real threats are handled by the
# reserve/reinforce phases above).
MOBILIZE_FLOOR = 3


def _aim(src, tid, t, ships, world):
    """Verified motion aim from planet row `src` to target row `t`; (angle, turns) or None."""
    res = aim_with_prediction(
        float(src[2]), float(src[3]), float(src[4]),
        tid, float(t[2]), float(t[3]), float(t[4]), ships, **world,
    )
    return (res[0], res[1]) if res is not None else None


def plan_turn(obs, config=None) -> List[list]:
    """Return this turn's Shots with active defense + mobilization. See module docstring."""
    me = int(_field(obs, "player"))
    planets = _field(obs, "planets")
    fleets = _field(obs, "fleets")

    world = _build_world(obs)
    threats = _inbound_threats(me, planets, fleets)
    by_id = {int(p[0]): p for p in planets}
    my_ids = [int(p[0]) for p in planets if int(p[1]) == me]

    # Spendable budget per owned planet = ships above its own defense reserve.
    reserve = {sid: defense_reserve(me, threats.get(sid, [])) for sid in my_ids}
    budget = {sid: int(by_id[sid][5]) - reserve[sid] for sid in my_ids}

    moves: List[list] = []

    # --- Phase 1: reinforce planets that can't meet their own reserve ----------
    # A planet is under-defended when its garrison < the reserve its threat
    # demands; the shortfall is what we try to ship in from a spare neighbour.
    for tid in my_ids:
        shortfall = reserve[tid] - int(by_id[tid][5])
        if shortfall <= 0:
            continue
        tgt = by_id[tid]
        # nearest spare friendly source with a verified in-time shot
        helpers = []
        for sid in my_ids:
            if sid == tid or budget.get(sid, 0) < 1:
                continue
            a = _aim(by_id[sid], tid, tgt, min(budget[sid], shortfall), world)
            if a is not None:
                helpers.append((a[1], sid, a[0]))  # (turns, source, angle)
        helpers.sort()
        for _turns, sid, angle in helpers:
            if shortfall <= 0:
                break
            send = min(budget[sid], shortfall)
            if send < 1:
                continue
            moves.append([sid, float(angle), int(send)])
            budget[sid] -= send
            shortfall -= send

    # --- Phase 2: ROI offense with remaining budget (v1's per-planet logic) ----
    # A planet acts once per turn: if it can afford a capture, it sends the
    # capture amount PLUS any surplus above the mobilize floor in one fleet
    # (concentrate, don't dribble two fleets from the same planet at the same
    # target). `acted` excludes a planet from the mobilize phase below.
    targets = [p for p in planets if int(p[1]) != me]
    acted = {sid for sid in my_ids if budget.get(sid, 0) != int(by_id[sid][5]) - reserve[sid]}
    for sid in my_ids:
        if sid in acted or budget.get(sid, 0) < 1:
            continue
        src = by_id[sid]
        spend = budget[sid]
        best = None  # (score, angle, send)
        for t in targets:
            tid = int(t[0])
            a = _aim(src, tid, t, spend, world)
            if a is None:
                continue
            angle, turns = a
            need = ships_needed_to_capture_timeline(
                int(t[5]), int(t[1]), int(t[6]), me, turns
            )
            if spend < need:
                continue
            # Send the capture cost, and roll surplus above the floor into the
            # same fleet rather than leaving it idle or splitting it off.
            send = max(need + SHIP_BUFFER, spend - MOBILIZE_FLOOR)
            send = min(spend, send)
            score = int(t[6]) / max(1, need) / max(1, turns)
            if best is None or score > best[0]:
                best = (score, angle, send)
        if best is not None:
            moves.append([sid, float(best[1]), int(best[2])])
            budget[sid] -= best[2]
            acted.add(sid)

    # --- Phase 3: mobilize idle garrisons that did nothing above ---------------
    # A planet that neither reinforced nor could afford a capture still sends its
    # surplus (above the floor) to its best verified target — idle ships win
    # nothing, and this is exactly the hoarding the Boss diag flagged.
    for sid in my_ids:
        if sid in acted:
            continue
        surplus = budget.get(sid, 0) - MOBILIZE_FLOOR
        if surplus < 1:
            continue
        src = by_id[sid]
        best = None  # (score, angle)
        for t in targets:
            tid = int(t[0])
            a = _aim(src, tid, t, surplus, world)
            if a is None:
                continue
            angle, turns = a
            score = int(t[6]) / max(1, turns)
            if best is None or score > best[0]:
                best = (score, angle)
        if best is not None:
            moves.append([sid, float(best[1]), int(surplus)])
            budget[sid] -= surplus

    return moves
