"""``roi_projected`` brain (AG12) — v1's sizing on a byte-exact garrison projection.

This is the first move-quality lever derived from "The Producer" (slawekbiel; see
wiki/producer_analysis.md). It is a **minimal, isolated diff** from v1
(`roi_greedy_predict`): identical target selection, ROI score, and motion-aware
aim — only the two *sizing* inputs change, both now read from an exact do-nothing
garrison projection instead of v1's crude estimates:

  * **spendable** (how much a source may launch) ← ``safe_drain`` over the
    projection (the min of the planet's held ship trajectory), replacing v1's
    threat-list ``defense_reserve``;
  * **need** (defenders to clear at the target) ← ``capture_floor`` at the fleet's
    projected arrival turn, replacing v1's ``ships_needed_to_capture_timeline``.

The projection (interpreter rolled forward H do-nothing turns) accounts for
production, every in-flight fleet's arrival, and combat — so both inputs are
byte-exact rather than estimated. Everything else, including the send rule
``min(spendable, need + SHIP_BUFFER)`` and the re-aim before emit, is v1 unchanged,
so the A/B isolates the projection-sizing lever. (The Producer's bigger
"send-everything + competitive flow-diff score" philosophy is a separate later
lever — see AG12's Outcome.)

Requires ``kaggle_environments`` (via the WorldModel projection). Pure plan_turn.

Public API:
    plan_turn(obs, config=None) -> list[list]
"""
from __future__ import annotations

from typing import List

from ..utils import aim_with_prediction
from ..garrison import capture_floor, garrison_projection, safe_drain
from .roi_greedy import SHIP_BUFFER, _field
from .roi_greedy_predict import _build_world

# Projection horizon. Matches the Producer's 2P default; long enough to see most
# captures/recaptures resolve, short enough to stay cheap (~0.5 ms/step).
H_DEFAULT = 18
CAPTURE_OVERHEAD = 1.0


def plan_turn(obs, config=None) -> List[list]:
    """Return this turn's Shots `[from_planet_id, angle, num_ships]`, sized from a
    do-nothing garrison projection. See the module docstring."""
    me = int(_field(obs, "player"))
    planets = _field(obs, "planets")

    num_players = max(2, max((int(p[1]) for p in planets), default=0) + 1)
    world = _build_world(obs)
    traj = garrison_projection(
        obs, H_DEFAULT, num_players=num_players,
        config=config if isinstance(config, dict) else None,
    )

    my_planets = [p for p in planets if int(p[1]) == me]
    moves: List[list] = []
    for src in my_planets:
        sid = int(src[0])
        sx, sy, sr = float(src[2]), float(src[3]), float(src[4])
        spendable = int(safe_drain(traj, sid, me, H_DEFAULT))
        if spendable < 1:
            continue

        best = None  # (score, tid, tx, ty, tr, send)
        for t in planets:
            t_owner = int(t[1])
            if t_owner == me:
                continue  # only non-owned planets are targets (neutral + enemies).
            tid = int(t[0])
            tx, ty, tr = float(t[2]), float(t[3]), float(t[4])

            aim = aim_with_prediction(sx, sy, sr, tid, tx, ty, tr, spendable, **world)
            if aim is None:
                continue  # no sun-safe shot lands — skip rather than waste a fleet.
            turns = aim[1]

            # Exact projected defenders at the fleet's arrival turn (1 if the
            # projection already shows the target ours by then).
            need = capture_floor(traj, tid, me, turns, overhead=CAPTURE_OVERHEAD)
            send = min(spendable, need + SHIP_BUFFER)
            if send < need:
                continue  # can't afford the capture after the reserve.

            t_prod = int(t[6])
            score = t_prod / max(1, need) / max(1, turns)
            if best is None or score > best[0]:
                best = (score, tid, tx, ty, tr, send)

        if best is None:
            continue

        # Re-aim with the fleet size we will actually send (no-op on static
        # targets; the correctness fix on moving ones).
        _, tid, tx, ty, tr, send = best
        refined = aim_with_prediction(sx, sy, sr, tid, tx, ty, tr, send, **world)
        if refined is None:
            continue
        moves.append([sid, float(refined[0]), int(send)])

    return moves
