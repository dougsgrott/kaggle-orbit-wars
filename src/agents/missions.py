"""
M1a "missions" brain — global multi-source allocation (tempo).

Same motion-aware aim and defense reserve as v1 `roi_greedy_predict`, but it
replaces v1's **per-planet greed** with a **global target plan**. The measured
problem (see wiki/measured_log.md): v1 lets each owned planet independently pick
its own best target, so it never *concentrates* force to take contested/defended
planets and gets out-expanded by the Boss, then collapses. This brain ranks all
non-owned targets once by ROI and fills each target's need from the *cheapest
combination of my planets' verified shots*, sharing each planet's spendable
budget across targets — so several planets can combine on one target, and ships
aren't wasted overshooting (v1 had every co-firing planet send the full
`need + buffer`).

Algorithm (pure; same `plan_turn(obs, config)` contract and Shot output as v1):
  1. Per owned planet, budget = ships - defense_reserve(inbound threats) (>= 0).
  2. For each non-owned target, collect each budgeted planet's *verified* aim
     (motion-aware); rank targets by ETA-discounted ROI
     `production / need / eta`, where `need` / `eta` use the soonest-arriving
     source (a documented approximation — fleets from different planets don't
     perfectly co-arrive; the per-turn replan + buffer absorb it).
  3. Walk targets best-ROI first; fill `need + SHIP_BUFFER` from sources
     cheapest (soonest) first, each sending `min(remaining_budget, remaining_need)`,
     **re-aiming with the actual send count** (moving-target correctness, the
     AG4 lesson). A source's budget is consumed across targets, so high-ROI
     targets get force first and nothing double-spends.

Degenerate single-source/single-target boards reduce to exactly v1's shot.

Reuses v1/utils helpers (nothing re-derived). Coexisting brain (ADR-0002),
A/B-tested vs v1 on the boss tier.

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
from .roi_greedy_predict import _build_world


def plan_turn(obs, config=None) -> List[list]:
    """Return this turn's Shots via global multi-source allocation. See module docstring."""
    me = int(_field(obs, "player"))
    planets = _field(obs, "planets")
    fleets = _field(obs, "fleets")

    world = _build_world(obs)
    threats = _inbound_threats(me, planets, fleets)

    # Per-planet spendable budget after its defense reserve.
    budget: dict = {}
    src_by_id: dict = {}
    for p in planets:
        if int(p[1]) != me:
            continue
        sid = int(p[0])
        b = int(p[5]) - defense_reserve(me, threats.get(sid, []))
        if b >= 1:
            budget[sid] = b
            src_by_id[sid] = p

    if not budget:
        return []

    # For each non-owned target, gather verified shots from each budgeted source.
    # plan: list of (roi, need, target_row, [(turns, sid)] sorted by turns)
    plan = []
    for t in planets:
        t_owner = int(t[1])
        if t_owner == me:
            continue
        tid = int(t[0])
        tx, ty, tr = float(t[2]), float(t[3]), float(t[4])
        t_ships, t_prod = int(t[5]), int(t[6])

        shots = []
        for sid, src in src_by_id.items():
            sx, sy, sr = float(src[2]), float(src[3]), float(src[4])
            aim = aim_with_prediction(sx, sy, sr, tid, tx, ty, tr, budget[sid], **world)
            if aim is not None:
                shots.append((aim[1], sid))  # (turns, source id)
        if not shots:
            continue  # no sun-safe shot from anywhere — skip target.
        shots.sort()
        soonest = shots[0][0]
        need = ships_needed_to_capture_timeline(t_ships, t_owner, t_prod, me, soonest)
        roi = t_prod / max(1, need) / max(1, soonest)
        plan.append((roi, need, t, shots))

    # Highest ROI first; deterministic tie-break by target id.
    plan.sort(key=lambda e: (-e[0], int(e[2][0])))

    moves: List[list] = []
    for _roi, need, t, shots in plan:
        tid = int(t[0])
        tx, ty, tr = float(t[2]), float(t[3]), float(t[4])
        remaining = need + SHIP_BUFFER
        for _turns, sid in shots:
            if remaining <= 0:
                break
            avail = budget.get(sid, 0)
            if avail < 1:
                continue
            send = min(avail, remaining)
            src = src_by_id[sid]
            refined = aim_with_prediction(
                float(src[2]), float(src[3]), float(src[4]),
                tid, tx, ty, tr, send, **world,
            )
            if refined is None:
                continue  # sized fleet has no verified intercept — try next source.
            moves.append([sid, float(refined[0]), int(send)])
            budget[sid] = avail - send
            remaining -= send

    return moves
