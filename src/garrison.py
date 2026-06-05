"""Do-nothing garrison projection + sizing helpers (AG12).

A numpy/stdlib-friendly port of the two move-quality primitives that power "The
Producer" (slawekbiel; see wiki/producer_analysis.md): a per-planet projection of
owner/ships over a future horizon assuming *no new launches*, and the two sizing
rules built on it —

  * ``capture_floor`` — how many ships must arrive at turn ``k`` to clear the
    target's *projected* defenders (1 if the projection says it is already ours);
  * ``safe_drain`` — the most a source can shed now while staying held over the
    whole horizon (the min of its held ship trajectory).

The Producer computes the projection analytically from an in-house byte-exact
forward model. We instead roll our **WorldModel** (the official interpreter, ADR-
0003) forward ``H`` do-nothing turns and read each planet's owner/ships per turn:
this is byte-exact *by construction* (it is the real engine) and accounts for
production, every in-flight fleet's arrival, and combat — which v1's crude
``ships_needed_to_capture_timeline`` / threat-list ``defense_reserve`` estimates do
not. Building the projection is one ~9 ms rollout per turn (≈0.5 ms/step × H);
every ``capture_floor`` / ``safe_drain`` lookup after that is a dict read.

Requires ``kaggle_environments`` (via the WorldModel), like the other search-class
brains.

Public API:
    garrison_projection(obs, H, num_players=2, config=None) -> list[dict]
    capture_floor(traj, target_pid, me, k, overhead=1.0)    -> int
    safe_drain(traj, source_pid, me, H)                     -> float
    defenders_at(traj, pid, k)                              -> (owner, ships)
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from . import worldmodel as _wm

# traj[k] maps planet_id -> (owner, ships) under the do-nothing projection at turn k.
Snapshot = Dict[int, Tuple[int, float]]


def _snapshot(fstate) -> Snapshot:
    """Planet id -> (owner, ships) for the current board of ``fstate``."""
    return {int(p[0]): (int(p[1]), float(p[5])) for p in _wm.planets_of(fstate)}


def garrison_projection(
    obs,
    H: int,
    num_players: int = 2,
    config: Optional[dict] = None,
) -> List[Snapshot]:
    """Do-nothing per-planet owner/ships trajectory over ``H`` turns.

    Returns ``traj`` with ``len == H + 1``: ``traj[0]`` is the current board and
    ``traj[k]`` is the board after ``k`` turns in which **every** player launches
    nothing — so it reflects production plus the resolution of every fleet already
    in flight. ``traj[k][pid] = (owner, ships)``; a planet absent at turn ``k`` is
    treated as ``(-1, 0)`` by the lookups below.
    """
    H = max(0, int(H))
    fstate = _wm.from_obs(obs, num_players=int(num_players), config=config)
    traj: List[Snapshot] = [_snapshot(fstate)]
    noop = [[] for _ in range(int(num_players))]
    cur = fstate
    for _ in range(H):
        if getattr(cur.env, "done", False):
            traj.append(traj[-1])  # pad: the game ended, board is frozen
            continue
        cur = _wm.step(cur, noop)
        traj.append(_snapshot(cur))
    return traj


def project_with_baseline(
    obs,
    H: int,
    num_players: int = 2,
    config: Optional[dict] = None,
):
    """One do-nothing rollout, reused for both sizing and the flow-diff baseline.

    Returns ``(fstate0, traj, base_totals)``:
      * ``fstate0`` — the un-stepped initial ``ForwardState`` (callers fork it with
        ``worldmodel.step`` to score a hypothetical launch; ``step`` deep-copies, so
        it is safe to reuse across candidates);
      * ``traj`` — the do-nothing owner/ships trajectory (``garrison_projection``),
        for ``capture_floor`` / ``safe_drain``;
      * ``base_totals`` — per-player **total ships** (on owned planets + in owned
        fleets) at horizon ``H`` under do-nothing = the flow-diff baseline a
        candidate launch is scored against.
    """
    import copy

    H = max(0, int(H))
    n = int(num_players)
    fstate0 = _wm.from_obs(obs, num_players=n, config=config)
    # Fork once, then roll the do-nothing chain IN PLACE on the fork (so fstate0 is
    # preserved for candidate forks and we pay one deep copy, not H).
    cur = copy.deepcopy(fstate0)
    traj: List[Snapshot] = [_snapshot(cur)]
    noop = [[] for _ in range(n)]
    for _ in range(H):
        if getattr(cur.env, "done", False):
            traj.append(traj[-1])
            continue
        _wm.advance_inplace(cur, noop)
        traj.append(_snapshot(cur))
    base_totals = _wm.score(cur)
    return fstate0, traj, base_totals


def defenders_at(traj: List[Snapshot], pid: int, k: int) -> Tuple[int, float]:
    """``(owner, ships)`` of planet ``pid`` at projected turn ``k`` (clamped)."""
    k = max(0, min(int(k), len(traj) - 1))
    return traj[k].get(int(pid), (-1, 0.0))


def capture_floor(
    traj: List[Snapshot],
    target_pid: int,
    me: int,
    k: int,
    overhead: float = 1.0,
) -> int:
    """Ships that must arrive at turn ``k`` to take/hold ``target_pid``.

    If the projection says the target is **mine** at ``k`` (an in-flight fleet of
    mine captures it, or I already hold it), arriving ships only reinforce — the
    floor is ``1``. Otherwise it is ``ceil(projected_defenders@k + overhead)``,
    never below ``1``. ``k`` is clamped into the projection horizon.
    """
    owner, ships = defenders_at(traj, target_pid, k)
    if owner == int(me):
        return 1
    return max(1, math.ceil(ships + float(overhead)))


def safe_drain(traj: List[Snapshot], source_pid: int, me: int, H: int) -> float:
    """Most ``source_pid`` can shed now while staying held over the horizon.

    Over turns ``1..H`` where the do-nothing projection still has me holding the
    planet (``owner == me`` and ``ships > 0``), the largest amount removable now
    that keeps the projected garrison ``>= 0`` on every such turn is the **min of
    that held ship trajectory** (leaving it at 0 on the worst held turn is allowed),
    capped at the ships held *now*. A doomed source (no held turn within ``H``) has
    nothing to protect, so the cap collapses to its current garrison.
    """
    pid, me = int(source_pid), int(me)
    cur = traj[0].get(pid)
    if cur is None:
        return 0.0
    cur_ships = cur[1]
    H = min(int(H), len(traj) - 1)
    held = [
        ships
        for k in range(1, H + 1)
        for (owner, ships) in (traj[k].get(pid, (-1, 0.0)),)
        if owner == me and ships > 0.0
    ]
    if not held:
        return cur_ships  # doomed: send it all, there is nothing to keep
    return max(0.0, min(min(held), cur_ships))
