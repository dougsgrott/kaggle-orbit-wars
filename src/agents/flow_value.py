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
from ..utils import aim_with_prediction
from .roi_greedy import _field
from .roi_greedy_predict import _build_world

# --- knobs (tunable; measured for both strength and budget) -----------------
H_DEFAULT = 14            # projection + scoring horizon (Producer uses 18/13)
MAX_SOURCES = 8           # owned planets considered as launch sources (by safe_drain)
MAX_TARGETS = 8           # non-owned planets considered as targets (by proximity)
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


def _plan(obs, config, aimer) -> List[list]:
    """Core planner, parameterised by the `aimer` (signature of
    `aim_with_prediction`). `plan_turn` uses the default aimer; the AG14 variant
    `flow_value_ia` injects the continuous-intercept aimer. See the module docstring."""
    me = int(_field(obs, "player"))
    planets = _field(obs, "planets")

    num_players = max(2, max((int(p[1]) for p in planets), default=0) + 1)
    H = H_DEFAULT
    world = _build_world(obs)
    cfg = config if isinstance(config, dict) else None
    fstate0, traj, base_totals = project_with_baseline(obs, H, num_players=num_players, config=cfg)

    my_planets = [p for p in planets if int(p[1]) == me]
    if not my_planets:
        return []
    targets_all = [p for p in planets if int(p[1]) != me]
    if not targets_all:
        return []

    # --- shortlist: top sources by safe_drain, top targets by proximity --------
    sized_sources = []
    for s in my_planets:
        sid = int(s[0])
        spend = int(safe_drain(traj, sid, me, H))
        if spend >= MIN_SPEND:
            sized_sources.append((spend, s))
    if not sized_sources:
        return []
    sized_sources.sort(key=lambda z: -z[0])
    sources = sized_sources[:MAX_SOURCES]

    def _proximity(t):
        tx, ty = float(t[2]), float(t[3])
        return min(_dist(float(s[2]), float(s[3]), tx, ty) for _, s in sources)

    targets = sorted(targets_all, key=_proximity)[:MAX_TARGETS]

    # --- build candidates (size by capture_floor; re-aim with the real send) ---
    cands = []  # (priority, src_pid, tgt_pid, angle, send)
    for spendable, s in sources:
        sid = int(s[0])
        sx, sy, sr = float(s[2]), float(s[3]), float(s[4])
        for t in targets:
            tid = int(t[0])
            if tid == sid:
                continue
            tx, ty, tr = float(t[2]), float(t[3]), float(t[4])
            # Size = the source's full safe_drain (the Producer's rule): a capture
            # becomes a forward base with surplus, which is what keeps the agent
            # expanding instead of stalling once the first ring is taken. The
            # flow-diff value still suppresses wasteful sends (a launch to a target
            # the projection shows already becoming mine scores ~0).
            send = int(spendable)
            aim = aimer(sx, sy, sr, tid, tx, ty, tr, send, **world)
            if aim is None:
                continue
            k = aim[1]
            need = capture_floor(traj, tid, me, k, overhead=CAPTURE_OVERHEAD)
            if send < need or send < MIN_SPEND:
                continue  # can't clear the projected defenders at arrival
            # cheap priority so the most promising are scored first under the guard.
            prio = float(t[6]) / max(1, need) / max(1, k)
            cands.append((prio, sid, tid, float(aim[0]), send))

    if not cands:
        return []
    cands.sort(key=lambda z: -z[0])

    # --- score by the flow-diff value -----------------------------------------
    # cands is priority-sorted. Score the top MIN_SCORED unconditionally (never pass
    # silently — the earlier guard's failure mode), then keep going until the soft
    # budget; but stop at the HARD budget regardless (always scoring at least one),
    # so a heavy/comet turn can't balloon far past the 1 s soft actTimeout. On normal
    # boards every candidate scores well within budget.
    t0 = time.monotonic()
    soft = t0 + PER_TURN_BUDGET_S
    hard = t0 + HARD_BUDGET_S
    scored = []  # (value, src_pid, tgt_pid, angle, send)
    for i, (prio, sid, tid, angle, send) in enumerate(cands):
        now = time.monotonic()
        if i >= 1 and now > hard:
            break
        if i >= MIN_SCORED and now > soft:
            break
        val = _candidate_value(fstate0, me, num_players, [sid, angle, send], base_totals, H)
        scored.append((val, sid, tid, angle, send))

    # --- greedy: best-first, one fleet per target, source-budget aware ---------
    scored.sort(key=lambda z: -z[0])
    budget = {int(p[0]): int(p[5]) for p in my_planets}
    taken = set()
    moves: List[list] = []
    for val, sid, tid, angle, send in scored:
        if val <= SCORE_THRESHOLD:
            break  # sorted desc — the rest are no better
        if tid in taken or budget.get(sid, 0) < send:
            continue
        moves.append([sid, angle, int(send)])
        budget[sid] -= send
        taken.add(tid)
    return moves


def plan_turn(obs, config=None) -> List[list]:
    """Return this turn's Shots `[from_planet_id, angle, num_ships]`, ranked by the
    competitive flow-diff value (default motion-aware aimer). See the module docstring."""
    return _plan(obs, config, aim_with_prediction)
