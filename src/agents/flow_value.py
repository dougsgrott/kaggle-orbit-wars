"""``flow_value`` brain (AG13) — the Producer's competitive flow-diff value.

The keystone move-quality lever mined from "The Producer" (see
wiki/producer_analysis.md). Three prior results (AG9 search, AG10 opponent-model +
4P value, AG12 sizing) all point to the same bottleneck: the **candidate value
function**. This brain ports the Producer's value —

    score(launch) = Δnet_me − Σ_opp Δnet_opp        (net = total ships at horizon H)

— measured as the difference between the **do-nothing projection** and the
projection *with this one launch applied*, via the WorldModel (the interpreter, so
production + every in-flight fleet + combat are byte-exact). Two properties make it
the value every prior lever missed:

  * **4P-aware by construction** — the opponent term sums over *all* rivals, so a
    move is graded by my net gain minus the whole field's (capturing an enemy planet
    earns its denied production as a bonus).
  * **exact marginal value** — a redundant launch (the projection already shows the
    target becoming mine) gains ~0 net ships and falls below threshold, so it never
    fires. This is exactly what fixes AG12's `capture_floor` token-fleet regression:
    sizing and scoring are one mechanism.

Sizing reuses AG12's [src/garrison.py] `safe_drain` (spendable) + `capture_floor`
(defenders at the fleet's arrival turn); aim reuses v1's `aim_with_prediction`
(intercept aim is the separate AG14 lever). Candidates are a (source × target)
shortlist; each is scored independently vs the shared baseline, then a greedy picks
best-first (one fleet per target, source-budget aware) above a net-ships threshold.

Budget: every turn is kept **under the 1 s soft actTimeout** — the do-nothing
rollout is ~6 ms and candidate scoring is wall-clock-guarded — so the 60 s overage
bank is never drained, for any game length. Requires `kaggle_environments`.

Public API:
    plan_turn(obs, config=None) -> list[list]
"""
from __future__ import annotations

import copy
import math
import time
from typing import List

from .. import worldmodel as _wm
from ..garrison import capture_floor, project_with_baseline, safe_drain
from ..utils import aim_with_prediction, fleet_speed
from .roi_greedy import _field
from .roi_greedy_predict import _build_world

# --- knobs (tunable; measured for both strength and budget) -----------------
H_DEFAULT = 14            # projection + scoring horizon (Producer uses 18/13)
MAX_SOURCES = 8           # owned planets considered as launch sources (by safe_drain)
MAX_TARGETS = 8           # non-owned planets considered as targets (by proximity)
MAX_DEF_TARGETS = 4       # AG15: owned planets the projection shows flipping (by urgency)
# --- AG15 regroup ---
REGROUP_PRESSURE_MIN = 0.25   # only regroup toward a materially more-stressed planet
MAX_REGROUP_TIME = 7.0        # a regroup fleet must arrive within this many turns
CAPTURE_OVERHEAD = 1.0
SCORE_THRESHOLD = 2.0     # fire only if marginal net ships over H exceeds this
MIN_SPEND = 1
PER_TURN_BUDGET_S = 0.40  # soft cap: stop scoring EXTRA candidates past MIN_SCORED
HARD_BUDGET_S = 0.85      # hard cap: stop scoring entirely (but always score >= 1),
#                           so even a heavy/comet turn stays near the 1 s soft limit.
MIN_SCORED = 3            # always score this many top candidates (never pass silently),
#                           but bounded by HARD_BUDGET_S so a slow board can't balloon.


def _dist(ax, ay, bx, by) -> float:
    return math.hypot(ax - bx, ay - by)


def _candidate_value(fstate0, me, num_players, launch, base_totals, H) -> float:
    """Δnet_me − Σ_opp Δnet_opp for one launch vs the do-nothing baseline.

    ``net_p`` = player ``p``'s total ships (planets + fleets) at horizon ``H``.
    Forks the current state **once**, plays ``launch`` (mine) on turn 0 and
    do-nothing thereafter — stepping IN PLACE on the fork (cheap; no per-step deep
    copy) — and diffs each player's total against ``base_totals``.
    """
    fork = copy.deepcopy(fstate0)
    turn0 = [[] for _ in range(num_players)]
    turn0[me] = [launch]
    _wm.advance_inplace(fork, turn0)
    noop = [[] for _ in range(num_players)]
    for _ in range(max(0, H - 1)):
        if getattr(fork.env, "done", False):
            break
        _wm.advance_inplace(fork, noop)
    cand = _wm.score(fork)
    d_me = cand[me] - base_totals[me]
    d_opp = sum(cand[p] - base_totals[p] for p in range(num_players) if p != me)
    return float(d_me - d_opp)


def _flip_targets(traj, my_planets, me, H, max_def=MAX_DEF_TARGETS):
    """Owned planets the do-nothing projection shows flipping within H, ranked by
    urgency ≈ projected ships lost (prod·(H−flip_turn) + garrison_now). [AG15]"""
    K = min(int(H), len(traj) - 1)
    out = []
    for p in my_planets:
        pid = int(p[0])
        flip_turn = None
        for k in range(1, K + 1):
            if traj[k].get(pid, (-1, 0.0))[0] != me:
                flip_turn = k
                break
        if flip_turn is not None:
            urgency = float(p[6]) * (H - flip_turn) + float(p[5])
            out.append((urgency, p))
    out.sort(key=lambda z: -z[0])
    return [p for _, p in out[:max_def]]


def _enemy_pressure(planets, me, H):
    """Per owned-planet reachable-enemy-mass proxy: Σ_enemy ships·(1−d/(speed·H))₊.
    The regroup gradient — higher means more contested. [AG15]"""
    enemies = [(float(p[2]), float(p[3]), float(p[5]))
               for p in planets if int(p[1]) >= 0 and int(p[1]) != me]
    pressure = {}
    for p in planets:
        if int(p[1]) != me:
            continue
        px, py = float(p[2]), float(p[3])
        tot = 0.0
        for ex, ey, esh in enemies:
            reach = max(1e-6, fleet_speed(max(1.0, esh)) * float(H))
            decay = 1.0 - _dist(px, py, ex, ey) / reach
            if decay > 0.0:
                tot += esh * decay
        pressure[int(p[0])] = tot
    return pressure


def _regroup(planets, me, H, traj, leftover, world, aimer):
    """Move leftover ships up the enemy-pressure gradient toward a materially more
    stressed owned planet that is still mine at the fleet's arrival turn. [AG15]"""
    pressure = _enemy_pressure(planets, me, H)
    owned = {int(p[0]): p for p in planets if int(p[1]) == me}
    H_axis = len(traj) - 1
    moves = []
    taken_dst = set()
    for sid, spare in sorted(leftover.items(), key=lambda z: -z[1]):
        if spare < MIN_SPEND:
            continue
        s = owned.get(sid)
        if s is None:
            continue
        sx, sy, sr = float(s[2]), float(s[3]), float(s[4])
        sp = pressure.get(sid, 0.0)
        best = None  # (gap, did, angle, send)
        for did, d in owned.items():
            if did == sid or did in taken_dst:
                continue
            gap = pressure.get(did, 0.0) - sp
            if gap <= REGROUP_PRESSURE_MIN:
                continue
            aim = aimer(sx, sy, sr, did, float(d[2]), float(d[3]), float(d[4]), int(spare), **world)
            if aim is None or aim[1] > MAX_REGROUP_TIME:
                continue
            k = min(int(math.ceil(aim[1])), H_axis)
            if traj[k].get(did, (-1, 0.0))[0] != me:  # must still be mine on arrival
                continue
            if best is None or gap > best[0]:
                best = (gap, did, float(aim[0]), int(spare))
        if best is not None:
            _, did, angle, send = best
            moves.append([sid, angle, send])
            taken_dst.add(did)
    return moves


def _plan(obs, config, aimer, *, enable_defense=False, enable_regroup=False,
          H=None, max_sources=None, max_targets=None, max_def=None,
          threshold=None, min_spend=None) -> List[list]:
    """Core planner, parameterised by the `aimer` and the AG15 levers. `plan_turn`
    runs the AG13 baseline (both levers off); variants flip them on.

    The five tuning knobs (`H`, `max_sources`, `max_targets`, `max_def`,
    `threshold`, `min_spend`) default to the module constants — so `flow_value` /
    `flow_value_def` behaviour is unchanged. They are exposed only so the ablation
    brains in [flow_value_abl.py] can step the planner toward the Producer's tuned
    config and measure the gap (see wiki/producer_diff.md). See the module docstring."""
    me = int(_field(obs, "player"))
    planets = _field(obs, "planets")

    num_players = max(2, max((int(p[1]) for p in planets), default=0) + 1)
    H = H_DEFAULT if H is None else int(H)
    max_sources = MAX_SOURCES if max_sources is None else int(max_sources)
    max_targets = MAX_TARGETS if max_targets is None else int(max_targets)
    max_def = MAX_DEF_TARGETS if max_def is None else int(max_def)
    threshold = SCORE_THRESHOLD if threshold is None else float(threshold)
    min_spend = MIN_SPEND if min_spend is None else int(min_spend)
    world = _build_world(obs)
    cfg = config if isinstance(config, dict) else None
    fstate0, traj, base_totals = project_with_baseline(obs, H, num_players=num_players, config=cfg)

    my_planets = [p for p in planets if int(p[1]) == me]
    if not my_planets:
        return []

    # --- shortlist: top sources by safe_drain, top targets by proximity --------
    sized_sources = []
    for s in my_planets:
        spend = int(safe_drain(traj, int(s[0]), me, H))
        if spend >= min_spend:
            sized_sources.append((spend, s))
    if not sized_sources:
        return []
    sized_sources.sort(key=lambda z: -z[0])
    sources = sized_sources[:max_sources]

    def _proximity(t):
        tx, ty = float(t[2]), float(t[3])
        return min(_dist(float(s[2]), float(s[3]), tx, ty) for _, s in sources)

    attack = sorted((p for p in planets if int(p[1]) != me), key=_proximity)[:max_targets]
    # AG15: owned planets projected to flip become *defensive* targets (the flow-diff
    # value rewards holding them — kept production + denied to the enemy).
    defense = _flip_targets(traj, my_planets, me, H, max_def) if enable_defense else []
    targets = [(t, False) for t in attack] + [(t, True) for t in defense]
    if not targets:
        return []

    # --- build candidates (size = full safe_drain; gate on capture_floor) ------
    cands = []  # (priority, src_pid, tgt_pid, angle, send, is_def)
    for spendable, s in sources:
        sid = int(s[0])
        sx, sy, sr = float(s[2]), float(s[3]), float(s[4])
        for t, is_def in targets:
            tid = int(t[0])
            if tid == sid:
                continue
            send = int(spendable)
            aim = aimer(sx, sy, sr, tid, float(t[2]), float(t[3]), float(t[4]), send, **world)
            if aim is None:
                continue
            k = aim[1]
            need = capture_floor(traj, tid, me, k, overhead=CAPTURE_OVERHEAD)
            if send < need or send < min_spend:
                continue
            prio = float(t[6]) / max(1, need) / max(1, k)
            cands.append((prio, sid, tid, float(aim[0]), send, is_def))

    if not cands:
        return []
    cands.sort(key=lambda z: -z[0])

    # --- score by the flow-diff value (MIN_SCORED floor + hard budget cap) -----
    t0 = time.monotonic()
    soft = t0 + PER_TURN_BUDGET_S
    hard = t0 + HARD_BUDGET_S
    scored = []  # (value, src_pid, tgt_pid, angle, send, is_def)
    for i, (prio, sid, tid, angle, send, is_def) in enumerate(cands):
        now = time.monotonic()
        if i >= 1 and now > hard:
            break
        if i >= MIN_SCORED and now > soft:
            break
        val = _candidate_value(fstate0, me, num_players, [sid, angle, send], base_totals, H)
        scored.append((val, sid, tid, angle, send, is_def))

    # --- greedy: best-first, one fleet per target, source-budget aware ---------
    # Role mutex (AG15): a reinforced planet can't also be a source, and a planet
    # drained as a source can't be reinforced this turn.
    scored.sort(key=lambda z: -z[0])
    orig = {int(p[0]): int(p[5]) for p in my_planets}
    budget = dict(orig)
    taken = set()
    reinforced = set()
    used_src = set()
    moves: List[list] = []
    for val, sid, tid, angle, send, is_def in scored:
        if val <= threshold:
            break
        if tid in taken or budget.get(sid, 0) < send:
            continue
        if sid in reinforced or (is_def and tid in used_src):
            continue
        moves.append([sid, angle, int(send)])
        budget[sid] -= send
        taken.add(tid)
        used_src.add(sid)
        if is_def:
            reinforced.add(tid)

    # --- AG15 regroup: marshal leftover ships up the pressure gradient ---------
    if enable_regroup:
        leftover = {}
        for spend, s in sources:
            sid = int(s[0])
            if sid in reinforced:
                continue
            committed = orig.get(sid, 0) - budget.get(sid, 0)
            spare = int(spend) - committed
            if spare >= MIN_SPEND:
                leftover[sid] = spare
        if leftover:
            moves.extend(_regroup(planets, me, H, traj, leftover, world, aimer))

    return moves


def plan_turn(obs, config=None) -> List[list]:
    """Return this turn's Shots `[from_planet_id, angle, num_ships]`, ranked by the
    competitive flow-diff value (AG13 baseline: no defense/regroup). See docstring."""
    return _plan(obs, config, aim_with_prediction)
