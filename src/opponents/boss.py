
"""
Orbit Wars — Physics-Complete Agent v8
=======================================

PHYSICS PRINCIPLES IMPLEMENTED
───────────────────────────────
Fleet mechanics
  • Logarithmic speed: v(n) = 1 + 5·(ln(n)/ln(1000))^1.5  [engine-exact]
  • Launch from planet surface: origin = planet_centre + (r+ε)·d̂
  • Effective travel distance = dist(launch, target_surface) = total - r_src - r_tgt
  • Travel time  T = ⌈dist / v(n)⌉  (discrete turns)

Sun collision (continuous)
  • Point-to-segment distance from sun centre to fleet path segment < sun_radius + margin
  • Uses algebraic projection, handles degenerate zero-length segments

Board boundary
  • Parametric ray vs AABB [0,100]²  → max safe travel distance before exit
  • Any aim that exits the board before reaching target is rejected

Planet position prediction by type
  Static  → always at current (x, y); no prediction needed
  Orbiting → circular orbit: P(t) = C + r·(cos(φ+ω·t), sin(φ+ω·t))
              orbit radius r taken from initial_by_id (not current position,
              which may be mid-rotation and give wrong result)
  Comet   → precomputed elliptical path; position = paths[idx][path_index + t]
              return None when comet has left the board

Intercept solver (orbiting planets & comets)
  • Primary: iterative refinement — aim at current pos, advance ETA,
    re-aim at predicted pos, repeat until Δpos < tolerance
  • Fallback: exhaustive scan from t_min = ⌈(d_min/v_max)⌉ to HORIZON
    t_min avoids wasted scan of turns where planet is physically unreachable
  • Both paths confirm with radius-corrected geometry

Minimum-ETA lower bound
  • t_min_physics = ⌈max(0, dist_centres - r_src - r_tgt) / MAX_SPEED⌉
  • Scan starts here instead of 1 → skips impossible early turns

STRATEGIC FRAMEWORK
────────────────────
WorldModel (single source of truth per turn)
  • Arrival ledger: maps every in-flight fleet to its target via geometric
    projection (perpendicular distance to heading vector)
  • Event-driven timeline simulator with exact multi-attacker combat resolution
  • Binary-search minimum garrison (smallest keep that survives all waves)
  • Proactive multi-enemy keep: sliding-window max over stacked threats in
    MULTI_ENEMY_PROACTIVE_HORIZON turns

Mission planner
  single     – one planet covers full need
  snipe      – arrive just before enemy captures a neutral
  swarm-2/3/4 – multiple sources gang up; ETA-matched within tight tolerance
  reinforce  – ship threatened friendly planet before it falls
  crash_exploit – exploit two enemy fleets destroying each other
  followup   – spend leftover ships after main dispatch

2-player vs 4-player adaptation
  is_four_player = (active owners ≥ 4) — exact engine definition
  4P: prefer neutrals, avoid 2-front wars, attack weakest enemy
  2P: more aggressive opening, smaller margins, faster expansion

Counter-attack (from_planet_id)
  Fleet observations carry from_planet_id — if an enemy planet just
  launched a large fleet, its garrison is temporarily depleted.
  A weakening ratio ≥ WEAKENED_RATIO triggers a score bonus.

Elimination acceleration
  Any enemy with total ships ≤ KILL_THRESHOLD gets KILL_MULT applied
  to all attack scores against their planets.
"""

import math
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field

# ─── Engine constants (match orbit_wars v1.0.9) ───────────────────────────────
BOARD           = 100.0
CENTER_X        = 50.0
CENTER_Y        = 50.0
SUN_R           = 10.0
MAX_SPEED       = 6.0
SUN_SAFETY      = 1.5      # extra clearance on sun radius
ROTATION_LIMIT  = 50.0     # orbital_radius + planet_radius threshold
TOTAL_STEPS     = 500
HORIZON         = 110      # max turns we simulate ahead
LAUNCH_CLEARANCE = 0.1     # extra gap beyond planet radius on launch

# ─── Phase thresholds ─────────────────────────────────────────────────────────
EARLY_TURN_LIMIT          = 40
OPENING_TURN_LIMIT        = 80
LATE_REMAINING_TURNS      = 60
VERY_LATE_REMAINING_TURNS = 25

# ─── Intercept solver ─────────────────────────────────────────────────────────
INTERCEPT_TOLERANCE   = 1   # turns; intercept valid if |T_actual - T_pred| ≤ this
ITERATIVE_REFINE_ITERS = 6  # iterations before fallback to scan
POSITION_CONVERGE_EPS  = 0.25  # units; position convergence threshold

# ─── Combat margins ───────────────────────────────────────────────────────────
SAFE_NEUTRAL_MARGIN      = 2
CONTESTED_NEUTRAL_MARGIN = 2

# ─── Opening filter thresholds ────────────────────────────────────────────────
SAFE_OPENING_PROD_THRESHOLD        = 4
SAFE_OPENING_TURN_LIMIT            = 10
ROTATING_OPENING_MAX_TURNS         = 13
ROTATING_OPENING_LOW_PROD          = 2
FOUR_PLAYER_ROTATING_REACTION_GAP  = 3
FOUR_PLAYER_ROTATING_SEND_RATIO    = 0.62
FOUR_PLAYER_ROTATING_TURN_LIMIT    = 10

# ─── Comet parameters ─────────────────────────────────────────────────────────
COMET_CHASE_BASE      = 10   # minimum chase turns
COMET_CHASE_PROD_MULT = 6    # per production-unit of comet
COMET_LIFE_MARGIN     = 5    # min (life - arrival) to bother capturing

# ─── Scoring weights ──────────────────────────────────────────────────────────
ATTACK_COST_TURN_WEIGHT = 0.55
SNIPE_COST_TURN_WEIGHT  = 0.45
INDIRECT_VALUE_SCALE    = 0.15
INDIRECT_FRIENDLY_W     = 0.35
INDIRECT_NEUTRAL_W      = 0.90
INDIRECT_ENEMY_W        = 1.25

# ─── Value multipliers ────────────────────────────────────────────────────────
STATIC_NEUTRAL_MULT             = 1.40
STATIC_HOSTILE_MULT             = 1.55
ROTATING_OPENING_MULT           = 0.90
HOSTILE_MULT                    = 1.85
OPENING_HOSTILE_MULT            = 1.45
SAFE_NEUTRAL_MULT               = 1.20
CONTESTED_NEUTRAL_MULT          = 0.70
EARLY_NEUTRAL_MULT              = 1.20
COMET_MULT                      = 0.65
SNIPE_MULT                      = 1.12
SWARM_MULT                      = 1.05
REINFORCE_MULT                  = 1.35
CRASH_EXPLOIT_MULT              = 1.18
FINISHING_HOSTILE_MULT          = 1.15
BEHIND_ROTATING_NEUTRAL_MULT    = 0.92

# ─── Counter-attack (from_planet_id weakening) ────────────────────────────────
WEAKENED_RATIO     = 0.25  # ships_launched/ships_total must exceed this
WEAKENED_MULT      = 1.30

# ─── Elimination acceleration ─────────────────────────────────────────────────
KILL_THRESHOLD = 35
KILL_MULT      = 1.45

# ─── Score multipliers ────────────────────────────────────────────────────────
STATIC_SCORE_MULT             = 1.18
EARLY_STATIC_NEUTRAL_S_MULT   = 1.25
FOUR_P_ROTATING_NEUTRAL_S_MULT = 0.84
DENSE_STATIC_NEUTRAL_COUNT    = 4
DENSE_ROTATING_NEUTRAL_S_MULT = 0.86
SNIPE_SCORE_MULT              = 1.12
SWARM_SCORE_MULT              = 1.06

# ─── Margins ──────────────────────────────────────────────────────────────────
NEUTRAL_MARGIN_BASE   = 2; NEUTRAL_MARGIN_PROD_W = 2; NEUTRAL_MARGIN_CAP   = 8
HOSTILE_MARGIN_BASE   = 3; HOSTILE_MARGIN_PROD_W = 2; HOSTILE_MARGIN_CAP   = 12
STATIC_TARGET_MARGIN  = 4; CONTESTED_TARGET_MARGIN = 5; FOUR_P_TARGET_MARGIN = 3
LONG_TRAVEL_MARGIN_START = 18; LONG_TRAVEL_MARGIN_DIV = 3; LONG_TRAVEL_MARGIN_CAP = 8
COMET_MARGIN_RELIEF   = 6
FINISHING_HOSTILE_SEND_BONUS = 3

# ─── Domination / mode thresholds ─────────────────────────────────────────────
BEHIND_DOM    = -0.20; AHEAD_DOM   = 0.18
FINISHING_DOM =  0.35; FINISHING_PROD_RATIO = 1.25
AHEAD_MARGIN_BONUS    = 0.08
BEHIND_MARGIN_PENALTY = 0.05
FINISHING_MARGIN_BONUS = 0.08

# ─── Late-game scoring extras ─────────────────────────────────────────────────
LATE_IMMEDIATE_SHIP_VALUE = 0.60
WEAK_ENEMY_THRESHOLD      = 45
ELIMINATION_BONUS         = 18.0

# ─── Reinforcement ────────────────────────────────────────────────────────────
REINFORCE_ENABLED          = True
REINFORCE_MIN_PROD         = 2
REINFORCE_MAX_TRAVEL       = 22
REINFORCE_SAFETY_MARGIN    = 2
REINFORCE_MAX_SRC_FRACTION = 0.75
REINFORCE_MIN_FUTURE_TURNS = 40

# ─── Multi-source swarm ───────────────────────────────────────────────────────
PARTIAL_SRC_MIN        = 7
MULTI_SRC_TOP_K        = 5
MULTI_SRC_ETA_TOL      = 2
MULTI_SRC_PENALTY      = 0.97
HOSTILE_SWARM_ETA_TOL  = 1

THREE_SRC_ENABLED      = True; THREE_SRC_MIN_SHIPS = 25; THREE_SRC_ETA_TOL = 1
THREE_SRC_PENALTY      = 0.93

FOUR_SRC_ENABLED       = True; FOUR_SRC_MIN_SHIPS  = 55; FOUR_SRC_ETA_TOL  = 1
FOUR_SRC_PENALTY       = 0.88

# ─── Crash exploit ────────────────────────────────────────────────────────────
CRASH_ENABLED       = True
CRASH_MIN_SHIPS     = 10
CRASH_ETA_WINDOW    = 2
CRASH_POST_DELAY    = 1

# ─── Doomed evacuation ────────────────────────────────────────────────────────
DOOMED_EVAC_TURNS  = 24
DOOMED_MIN_SHIPS   = 8

# ─── Follow-up pass ───────────────────────────────────────────────────────────
FOLLOWUP_MIN       = 10
LOW_COMET_PROD     = 1
LATE_CAPTURE_BUF   = 5
VERY_LATE_BUF      = 3

# ─── Rear logistics ───────────────────────────────────────────────────────────
REAR_SRC_MIN       = 20; REAR_DIST_RATIO = 1.25; REAR_STAGE_PROG = 0.78
REAR_RATIO_2P      = 0.62; REAR_RATIO_4P = 0.70
REAR_SEND_MIN      = 10; REAR_MAX_TRAVEL = 40

# ─── Multi-enemy proactive defense ────────────────────────────────────────────
MULTI_ENEMY_HORIZON    = 14
MULTI_ENEMY_STACK_WIN  = 5
MULTI_ENEMY_RATIO      = 0.22
PROACTIVE_DEF_HORIZON  = 12
PROACTIVE_DEF_RATIO    = 0.18


# =============================================================================
# NAMED TYPES
# =============================================================================

Planet = namedtuple("Planet", ["id", "owner", "x", "y", "radius", "ships", "production"])
Fleet  = namedtuple("Fleet",  ["id", "owner", "x", "y", "angle", "from_planet_id", "ships"])


@dataclass(frozen=True)
class ShotOption:
    score: float; src_id: int; target_id: int
    angle: float; turns: int; needed: int; send_cap: int; mission: str = "capture"


@dataclass
class Mission:
    kind: str; score: float; target_id: int; turns: int
    options: list = field(default_factory=list)


# =============================================================================
# PHYSICS — FLEET SPEED
# =============================================================================

def fleet_speed(ships: float) -> float:
    """
    Engine-exact logarithmic speed curve.
    1 ship  → 1.0 u/turn
    ~1000   → 6.0 u/turn  (MAX_SPEED)
    """
    if ships <= 1:
        return 1.0
    ratio = math.log(ships) / math.log(1000.0)
    ratio = max(0.0, min(1.0, ratio))
    return 1.0 + (MAX_SPEED - 1.0) * (ratio ** 1.5)


# =============================================================================
# PHYSICS — GEOMETRY
# =============================================================================

def pdist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def point_to_segment_dist(px, py, x1, y1, x2, y2) -> float:
    """Perpendicular (or endpoint) distance from point to segment."""
    dx, dy = x2 - x1, y2 - y1
    seg_sq = dx * dx + dy * dy
    if seg_sq <= 1e-12:
        return pdist(px, py, x1, y1)
    t = ((px - x1) * dx + (py - y1) * dy) / seg_sq
    t = max(0.0, min(1.0, t))
    return pdist(px, py, x1 + t * dx, y1 + t * dy)


def segment_hits_sun(x1, y1, x2, y2, safety: float = SUN_SAFETY) -> bool:
    """True when fleet path [from→to] passes through sun (with margin)."""
    return point_to_segment_dist(CENTER_X, CENTER_Y, x1, y1, x2, y2) < SUN_R + safety


def ray_board_exit_dist(sx: float, sy: float, angle: float) -> float:
    """
    Maximum distance a fleet launched at (sx,sy) with `angle` can travel
    while remaining inside the [0, BOARD]² play field.
    Returns the distance to the first boundary hit.
    """
    ca, sa = math.cos(angle), math.sin(angle)
    t_max = 1e18
    if ca > 1e-9:
        t_max = min(t_max, (BOARD - sx) / ca)
    elif ca < -1e-9:
        t_max = min(t_max, -sx / ca)
    if sa > 1e-9:
        t_max = min(t_max, (BOARD - sy) / sa)
    elif sa < -1e-9:
        t_max = min(t_max, -sy / sa)
    return t_max


# =============================================================================
# PHYSICS — LAUNCH GEOMETRY
# =============================================================================

def launch_origin(sx, sy, sr, angle):
    """Fleet spawns just outside source planet radius."""
    c = sr + LAUNCH_CLEARANCE
    return sx + math.cos(angle) * c, sy + math.sin(angle) * c


def fleet_path(sx, sy, sr, tx, ty, tr):
    """
    Returns (angle, lx, ly, ex, ey, hit_dist) where:
      angle    = direction from source centre to target centre
      (lx, ly) = launch point (just outside source surface)
      (ex, ey) = point where fleet touches target surface
      hit_dist = actual travel distance
    """
    angle   = math.atan2(ty - sy, tx - sx)
    lx, ly  = launch_origin(sx, sy, sr, angle)
    # travel until fleet reaches target surface
    hit_dist = max(0.0, pdist(sx, sy, tx, ty) - (sr + LAUNCH_CLEARANCE) - tr)
    ex = lx + math.cos(angle) * hit_dist
    ey = ly + math.sin(angle) * hit_dist
    return angle, lx, ly, ex, ey, hit_dist


def safe_aim(sx, sy, sr, tx, ty, tr):
    """
    Compute (angle, travel_dist) for direct path src→tgt.
    Returns None if path hits sun OR fleet would exit board before reaching target.
    """
    angle, lx, ly, ex, ey, hit_dist = fleet_path(sx, sy, sr, tx, ty, tr)
    if segment_hits_sun(lx, ly, ex, ey):
        return None
    board_d = ray_board_exit_dist(lx, ly, angle)
    if board_d < hit_dist - 0.5:        # fleet exits board first
        return None
    return angle, hit_dist


def travel_turns(sx, sy, sr, tx, ty, tr, ships) -> int:
    """Integer turns needed to travel from src to tgt surface."""
    s = safe_aim(sx, sy, sr, tx, ty, tr)
    if s is None:
        return 10 ** 9
    _, d = s
    return max(1, int(math.ceil(d / fleet_speed(max(1, ships)))))


# =============================================================================
# PHYSICS — PLANET POSITION PREDICTION
# =============================================================================

def is_static(planet, initial_by_id) -> bool:
    """
    Use initial orbital radius (from game start) for classification.
    A planet's current position may be mid-orbit, giving wrong r.
    """
    init = initial_by_id.get(planet.id)
    ref  = init if init is not None else planet
    r    = pdist(ref.x, ref.y, CENTER_X, CENTER_Y)
    return r + ref.radius >= ROTATION_LIMIT


def orbit_pos(planet, initial_by_id, ang_vel: float, t: float):
    """
    Position of an orbiting planet after t turns.
    Uses initial_by_id for orbital radius (invariant through rotation).
    Returns (x, y) — same as current pos if planet is static.
    """
    init = initial_by_id.get(planet.id)
    if init is None:
        return planet.x, planet.y
    r = pdist(init.x, init.y, CENTER_X, CENTER_Y)
    if r + init.radius >= ROTATION_LIMIT:
        return planet.x, planet.y          # static — no movement
    phi = math.atan2(planet.y - CENTER_Y, planet.x - CENTER_X)
    phi += ang_vel * t
    return CENTER_X + r * math.cos(phi), CENTER_Y + r * math.sin(phi)


def comet_pos(planet_id: int, comets, t: int):
    """Position along precomputed elliptical comet path. None if expired."""
    for g in comets:
        pids = g.get("planet_ids", [])
        if planet_id not in pids:
            continue
        idx   = pids.index(planet_id)
        paths = g.get("paths", [])
        pi    = g.get("path_index", 0)
        if idx >= len(paths):
            return None
        path = paths[idx]
        fi   = pi + int(t)
        return (path[fi][0], path[fi][1]) if 0 <= fi < len(path) else None
    return None


def comet_life(planet_id: int, comets) -> int:
    for g in comets:
        pids = g.get("planet_ids", [])
        if planet_id not in pids:
            continue
        idx   = pids.index(planet_id)
        paths = g.get("paths", [])
        pi    = g.get("path_index", 0)
        return max(0, len(paths[idx]) - pi) if idx < len(paths) else 0
    return 0


def comet_chase_limit(planet, comets) -> int:
    """Dynamic comet chase cap: longer for high-production comets."""
    life  = comet_life(planet.id, comets)
    limit = max(COMET_CHASE_BASE, planet.production * COMET_CHASE_PROD_MULT)
    return min(limit, life - COMET_LIFE_MARGIN) if life > COMET_LIFE_MARGIN else 0


def target_pos(planet, t, initial_by_id, ang_vel, comets, comet_ids):
    """Universal position prediction dispatching on planet type."""
    if planet.id in comet_ids:
        return comet_pos(planet.id, comets, t)       # may be None
    return orbit_pos(planet, initial_by_id, ang_vel, t)


# =============================================================================
# PHYSICS — INTERCEPT SOLVER
# =============================================================================

def _intercept_scan(src, target, ships, initial_by_id, ang_vel, comets, comet_ids,
                     t_start: int):
    """
    Exhaustive scan from t_start to HORIZON for valid intercept windows.
    Uses t_min_physics lower bound to skip impossible early turns.
    Returns (angle, turns, px, py) of the earliest valid intercept, or None.
    """
    v       = fleet_speed(max(1, ships))
    best    = None
    best_sc = None
    max_t   = HORIZON
    if target.id in comet_ids:
        max_t = min(max_t, max(0, comet_life(target.id, comets) - 1))

    for t in range(t_start, max_t + 1):
        pos = target_pos(target, t, initial_by_id, ang_vel, comets, comet_ids)
        if pos is None:
            break
        px, py = pos
        sa = safe_aim(src.x, src.y, src.radius, px, py, target.radius)
        if sa is None:
            continue
        angle, d = sa
        actual_t = max(1, int(math.ceil(d / v)))
        if abs(actual_t - t) > INTERCEPT_TOLERANCE:
            continue
        # Confirm with final planet position
        real_t = max(actual_t, t)
        pos2   = target_pos(target, real_t, initial_by_id, ang_vel, comets, comet_ids)
        if pos2 is None:
            continue
        sa2 = safe_aim(src.x, src.y, src.radius, pos2[0], pos2[1], target.radius)
        if sa2 is None:
            continue
        angle2, d2 = sa2
        actual_t2  = max(1, int(math.ceil(d2 / v)))
        delta      = abs(actual_t2 - real_t)
        if delta > INTERCEPT_TOLERANCE:
            continue
        sc = (delta, actual_t2, t)
        if best is None or sc < best_sc:
            best_sc = sc
            best = (angle2, actual_t2, pos2[0], pos2[1])
        if delta == 0:  # perfect intercept found early
            return best
    return best


def intercept(src, target, ships, initial_by_id, ang_vel, comets, comet_ids):
    """
    Minimum-ETA intercept solver.

    Algorithm:
    1. Compute physics lower bound on arrival: t_min = ⌈d_min / v_max⌉
    2. Iterative refinement (fast path): aim at current pos, advance ETA,
       re-aim at predicted pos, repeat until convergence.
    3. If direct path fails sun/bounds OR iterative doesn't converge,
       fall through to exhaustive scan starting from t_min.
    Returns (angle, turns, pred_x, pred_y) or None.
    """
    v       = fleet_speed(max(1, ships))
    d_floor = max(0.0, pdist(src.x, src.y, target.x, target.y)
                  - src.radius - target.radius)
    t_min   = max(1, int(math.ceil(d_floor / MAX_SPEED)))

    # --- Iterative refinement ---
    direct = safe_aim(src.x, src.y, src.radius, target.x, target.y, target.radius)
    if direct is not None:
        angle, d0 = direct
        t_cur = max(1, int(math.ceil(d0 / v)))
        tx, ty = target.x, target.y
        for _ in range(ITERATIVE_REFINE_ITERS):
            pos = target_pos(target, t_cur, initial_by_id, ang_vel, comets, comet_ids)
            if pos is None:
                return None
            ntx, nty = pos
            sa = safe_aim(src.x, src.y, src.radius, ntx, nty, target.radius)
            if sa is None:
                break                       # path blocked — fall to scan
            angle, nd = sa
            t_next = max(1, int(math.ceil(nd / v)))
            if (abs(ntx - tx) < POSITION_CONVERGE_EPS
                    and abs(nty - ty) < POSITION_CONVERGE_EPS
                    and abs(t_next - t_cur) <= INTERCEPT_TOLERANCE):
                return angle, t_next, ntx, nty
            tx, ty  = ntx, nty
            t_cur   = t_next
        # Confirm last iterate
        pos = target_pos(target, t_cur, initial_by_id, ang_vel, comets, comet_ids)
        if pos is not None:
            sa = safe_aim(src.x, src.y, src.radius, pos[0], pos[1], target.radius)
            if sa is not None:
                angle, d_f = sa
                t_f = max(1, int(math.ceil(d_f / v)))
                if abs(t_f - t_cur) <= INTERCEPT_TOLERANCE:
                    return angle, t_f, pos[0], pos[1]

    # --- Exhaustive scan (fallback) ---
    return _intercept_scan(src, target, ships, initial_by_id, ang_vel, comets, comet_ids,
                           t_min)


# =============================================================================
# WORLD MODEL — FLEET TRACKING
# =============================================================================

def fleet_to_planet(fleet, planets):
    """
    Find which planet a fleet is heading toward via geometric projection.
    Returns (planet, eta_turns) or (None, None).
    Perpendicular distance from planet centre to fleet direction must be
    less than planet radius.
    """
    dx_dir = math.cos(fleet.angle)
    dy_dir = math.sin(fleet.angle)
    speed  = fleet_speed(fleet.ships)
    best_p, best_t = None, 1e18
    for p in planets:
        dx, dy = p.x - fleet.x, p.y - fleet.y
        proj   = dx * dx_dir + dy * dy_dir
        if proj < 0:
            continue
        perp_sq = dx * dx + dy * dy - proj * proj
        if perp_sq >= p.radius * p.radius:
            continue
        hit_d = max(0.0, proj - math.sqrt(max(0.0, p.radius * p.radius - perp_sq)))
        t = hit_d / speed
        if t <= HORIZON and t < best_t:
            best_t, best_p = t, p
    return (best_p, int(math.ceil(best_t))) if best_p else (None, None)


def build_arrival_ledger(fleets, planets):
    ledger = {p.id: [] for p in planets}
    for f in fleets:
        tgt, eta = fleet_to_planet(f, planets)
        if tgt is None:
            continue
        ledger[tgt.id].append((eta, f.owner, int(f.ships)))
    return ledger


# =============================================================================
# WORLD MODEL — COMBAT SIMULATION
# =============================================================================

def resolve_combat(owner, garrison, arrivals):
    """
    Multi-attacker combat at a planet.
    Largest force beats second-largest; survivor fights garrison.
    """
    by_owner = {}
    for _, o, s in arrivals:
        by_owner[o] = by_owner.get(o, 0) + s
    if not by_owner:
        return owner, max(0.0, garrison)
    ranked = sorted(by_owner.items(), key=lambda x: x[1], reverse=True)
    top_o, top_s = ranked[0]
    if len(ranked) > 1:
        sec_s = ranked[1][1]
        if top_s == sec_s:
            return owner, max(0.0, garrison)  # tie → both die
        survivor_o, survivor_s = top_o, top_s - sec_s
    else:
        survivor_o, survivor_s = top_o, top_s
    if survivor_s <= 0:
        return owner, max(0.0, garrison)
    if owner == survivor_o:
        return owner, garrison + survivor_s
    garrison -= survivor_s
    return (survivor_o, -garrison) if garrison < 0 else (owner, garrison)


def simulate_timeline(planet, arrivals, player, horizon):
    """
    Event-driven simulation of planet garrison over [0, horizon].
    Returns dict with owner_at, ships_at, keep_needed, fall_turn, holds_full.
    keep_needed is found via binary search: minimum garrison that survives all waves.
    """
    horizon = max(0, int(math.ceil(horizon)))
    events  = sorted(
        [(max(1, int(math.ceil(t))), o, int(s))
         for t, o, s in arrivals if s > 0 and t <= horizon],
        key=lambda x: x[0])
    by_turn = defaultdict(list)
    for item in events:
        by_turn[item[0]].append(item)

    owner    = planet.owner
    garrison = float(planet.ships)
    owner_at = {0: owner}
    ships_at = {0: max(0.0, garrison)}
    fall_turn = None

    for turn in range(1, horizon + 1):
        if owner != -1:
            garrison += planet.production
        grp = by_turn.get(turn, [])
        prev = owner
        if grp:
            owner, garrison = resolve_combat(owner, garrison, grp)
            if prev == player and owner != player and fall_turn is None:
                fall_turn = turn
        owner_at[turn] = owner
        ships_at[turn] = max(0.0, garrison)

    # Binary search: min garrison that survives
    keep_needed = 0
    holds_full  = True

    if planet.owner == player:
        def survives(keep):
            so, sg = planet.owner, float(keep)
            for t in range(1, horizon + 1):
                if so != -1:
                    sg += planet.production
                grp = by_turn.get(t, [])
                if grp:
                    so, sg = resolve_combat(so, sg, grp)
                    if so != player:
                        return False
            return so == player

        if survives(int(planet.ships)):
            lo, hi = 0, int(planet.ships)
            while lo < hi:
                mid = (lo + hi) // 2
                (hi if survives(mid) else lo).__class__  # trick to assign
                if survives(mid):
                    hi = mid
                else:
                    lo = mid + 1
            keep_needed = lo
        else:
            holds_full  = False
            keep_needed = int(planet.ships)

    return {"owner_at": owner_at, "ships_at": ships_at, "keep_needed": keep_needed,
            "fall_turn": fall_turn, "holds_full": holds_full, "horizon": horizon}


def timeline_state(tl, t):
    t = max(0, min(int(math.ceil(t)), tl["horizon"]))
    return tl["owner_at"].get(t, tl["owner_at"][tl["horizon"]]), \
           max(0.0, tl["ships_at"].get(t, tl["ships_at"][tl["horizon"]]))


# =============================================================================
# WORLD MODEL — STRATEGIC CONTEXT
# =============================================================================

def count_active_players(planets, fleets) -> int:
    owners = {p.owner for p in planets if p.owner != -1}
    owners |= {f.owner for f in fleets}
    return max(2, len(owners))


def indirect_wealth(planet, planets, player) -> float:
    """
    Neighbourhood production value: weighted sum of nearby planets'
    production, adjusted by ownership. Proxy for strategic position.
    """
    w = 0.0
    for other in planets:
        if other.id == planet.id:
            continue
        d = pdist(planet.x, planet.y, other.x, other.y)
        if d < 1:
            continue
        factor = other.production / (d + 12.0)
        if other.owner == player:
            w += factor * INDIRECT_FRIENDLY_W
        elif other.owner == -1:
            w += factor * INDIRECT_NEUTRAL_W
        else:
            w += factor * INDIRECT_ENEMY_W
    return w


def detect_crashes(ledger, player, eta_window):
    """
    Find planets where two different enemy factions will arrive within
    eta_window turns of each other — crash-exploit opportunities.
    """
    crashes = []
    for tid, arrivals in ledger.items():
        enemy_ev = [(e, o, s) for e, o, s in arrivals
                    if o not in (-1, player) and s > 0]
        if len(enemy_ev) < 2:
            continue
        by_owner = defaultdict(list)
        for e, o, s in enemy_ev:
            by_owner[o].append((e, s))
        if len(by_owner) < 2:
            continue
        enemy_ev.sort()
        matched = False
        for i in range(len(enemy_ev)):
            if matched:
                break
            for j in range(i + 1, len(enemy_ev)):
                ea, oa, sa = enemy_ev[i]
                eb, ob, sb = enemy_ev[j]
                if oa == ob or abs(ea - eb) > eta_window:
                    continue
                if sa + sb < CRASH_MIN_SHIPS:
                    continue
                crashes.append({"target_id": tid, "crash_turn": max(ea, eb),
                                "total": sa + sb})
                matched = True
                break
    return crashes


# =============================================================================
# WORLD MODEL — CLASS
# =============================================================================

class WorldModel:
    def __init__(self, player, step, planets, fleets,
                 initial_by_id, ang_vel, comets, comet_ids):
        self.player        = player
        self.step          = step
        self.planets       = planets
        self.fleets        = fleets
        self.initial_by_id = initial_by_id
        self.ang_vel       = ang_vel
        self.comets        = comets
        self.comet_ids     = set(comet_ids)

        self.planet_by_id  = {p.id: p for p in planets}
        self.remaining     = max(1, TOTAL_STEPS - step)

        # Planet classification
        self.my_planets    = [p for p in planets if p.owner == player]
        self.enemy_planets = [p for p in planets if p.owner not in (-1, player)]
        self.neutral_planets = [p for p in planets if p.owner == -1]
        self.static_neutral_planets = [
            p for p in self.neutral_planets
            if is_static(p, initial_by_id) and p.id not in self.comet_ids
        ]

        # Player count & phase flags
        n                  = count_active_players(planets, fleets)
        self.num_players   = n
        self.is_four_player = (n >= 4)           # engine: 4-player means 4 owners
        self.is_early      = step < EARLY_TURN_LIMIT
        self.is_opening    = step < OPENING_TURN_LIMIT
        self.is_late       = self.remaining < LATE_REMAINING_TURNS
        self.is_very_late  = self.remaining < VERY_LATE_REMAINING_TURNS

        # Ship & production counts
        owner_ships = defaultdict(int)
        owner_prod  = defaultdict(int)
        for p in planets:
            if p.owner != -1:
                owner_ships[p.owner] += int(p.ships)
                owner_prod[p.owner]  += int(p.production)
        for f in fleets:
            if f.owner >= 0:
                owner_ships[f.owner] += int(f.ships)
        self.owner_strength   = owner_ships
        self.owner_production = owner_prod
        self.my_total    = owner_ships.get(player, 0)
        self.enemy_total = sum(v for k, v in owner_ships.items() if k != player)
        self.max_enemy   = max((v for k, v in owner_ships.items() if k != player), default=0)
        self.my_prod     = owner_prod.get(player, 0)
        self.enemy_prod  = sum(v for k, v in owner_prod.items() if k != player)

        # Counter-attack: track ships recently launched from enemy planets
        self.enemy_sent_from = defaultdict(float)
        for f in fleets:
            if f.owner != player and f.from_planet_id >= 0:
                self.enemy_sent_from[f.from_planet_id] += f.ships

        # Arrival ledger + timelines
        self.ledger       = build_arrival_ledger(fleets, planets)
        self.timelines    = {
            p.id: simulate_timeline(p, self.ledger.get(p.id, []), player, HORIZON)
            for p in planets
        }
        self.wealth_map   = {p.id: indirect_wealth(p, planets, player) for p in planets}

        # Enemy crashes (4-player only — the scenario where enemies crash into each other)
        self.crashes = (detect_crashes(self.ledger, player, CRASH_ETA_WINDOW)
                        if CRASH_ENABLED and self.is_four_player else [])

        # Defense reserves (binary search + proactive multi-enemy keep)
        self.reserve   = {}
        self.available = {}
        self.doomed    = set()
        self.threatened = {}
        self._build_defense()

        # Caches
        self.reaction_cache  = {}
        self.need_cache      = {}

    # ── Defense buffer computation ───────────────────────────────────────────

    def _proactive_keep(self, planet) -> int:
        """
        Proactive garrison: keep ships to counter the worst foreseeable
        enemy stack arriving within MULTI_ENEMY_HORIZON turns.
        Uses a sliding-window maximum over threat stacks.
        """
        threats = []
        for ep in self.enemy_planets:
            t = travel_turns(ep.x, ep.y, ep.radius,
                             planet.x, planet.y, planet.radius,
                             max(1, ep.ships))
            if t <= MULTI_ENEMY_HORIZON:
                threats.append((t, int(ep.ships)))
        if not threats:
            return 0
        threats.sort()
        best_stack = 0
        lo, running = 0, 0
        for hi in range(len(threats)):
            running += threats[hi][1]
            while threats[hi][0] - threats[lo][0] > MULTI_ENEMY_STACK_WIN:
                running -= threats[lo][1]; lo += 1
            best_stack = max(best_stack, running)
        proactive = int(best_stack * MULTI_ENEMY_RATIO)
        # Legacy single-threat check
        legacy = max(
            (int(s * PROACTIVE_DEF_RATIO) for t, s in threats
             if t <= PROACTIVE_DEF_HORIZON), default=0)
        return max(proactive, legacy)

    def _build_defense(self):
        for p in self.my_planets:
            tl   = self.timelines[p.id]
            keep = max(tl["keep_needed"], self._proactive_keep(p))
            self.reserve[p.id]   = min(int(p.ships), keep)
            self.available[p.id] = max(0, int(p.ships) - self.reserve[p.id])

            fall = tl["fall_turn"]
            if not tl["holds_full"] and fall is not None:
                if fall <= DOOMED_EVAC_TURNS and p.ships >= DOOMED_MIN_SHIPS:
                    self.doomed.add(p.id)
                if (REINFORCE_ENABLED and p.production >= REINFORCE_MIN_PROD
                        and self.remaining >= REINFORCE_MIN_FUTURE_TURNS):
                    oa, sa  = tl["owner_at"], tl["ships_at"]
                    deficit = 0
                    for t in range(1, fall + 1):
                        if oa.get(t) != self.player:
                            deficit = max(deficit, int(math.ceil(sa.get(t, 0))) + 1)
                            break
                    self.threatened[p.id] = {"fall_turn": fall,
                                             "deficit": max(1, deficit)}

    # ── Accessors ────────────────────────────────────────────────────────────

    def is_static_planet(self, pid):
        return is_static(self.planet_by_id[pid], self.initial_by_id)

    def comet_life(self, pid):
        return comet_life(pid, self.comets)

    def chase_limit(self, pid):
        return comet_chase_limit(self.planet_by_id[pid], self.comets)

    def src_inventory(self, sid, spent):
        return max(0, int(self.planet_by_id[sid].ships) - spent[sid])

    def src_attack(self, sid, spent):
        return max(0, self.available.get(sid, 0) - spent[sid])

    def plan_shot(self, src_id, tgt_id, ships):
        return intercept(
            self.planet_by_id[src_id], self.planet_by_id[tgt_id], ships,
            self.initial_by_id, self.ang_vel, self.comets, self.comet_ids)

    def reaction_times(self, tid):
        """Cached (my_min_eta, enemy_min_eta) to planet tid."""
        if tid in self.reaction_cache:
            return self.reaction_cache[tid]
        tgt  = self.planet_by_id[tid]
        my_t = min((travel_turns(p.x, p.y, p.radius, tgt.x, tgt.y, tgt.radius,
                                 max(1, p.ships))
                    for p in self.my_planets), default=10 ** 9)
        en_t = min((travel_turns(p.x, p.y, p.radius, tgt.x, tgt.y, tgt.radius,
                                 max(1, p.ships))
                    for p in self.enemy_planets), default=10 ** 9)
        self.reaction_cache[tid] = (my_t, en_t)
        return my_t, en_t

    def projected_state(self, tid, arrival_turn, planned=None, extra=()):
        planned = planned or {}
        cutoff  = max(1, int(math.ceil(arrival_turn)))
        if not planned.get(tid) and not extra:
            return timeline_state(self.timelines[tid], cutoff)
        arrivals = [x for x in self.ledger.get(tid, []) if x[0] <= cutoff]
        arrivals += [x for x in planned.get(tid, []) if x[0] <= cutoff]
        arrivals += [x for x in extra if x[0] <= cutoff]
        tgt = self.planet_by_id[tid]
        dyn = simulate_timeline(tgt, arrivals, self.player, cutoff)
        return timeline_state(dyn, cutoff)

    def ships_to_capture(self, tid, arrival_turn, planned=None, extra=()):
        planned = planned or {}
        cutoff  = max(1, int(math.ceil(arrival_turn)))
        ck = None
        if not planned.get(tid) and not extra:
            ck = (tid, cutoff)
            if ck in self.need_cache:
                return self.need_cache[ck]
        owner_t, ships_t = self.projected_state(tid, cutoff, planned, extra)
        need = 0 if owner_t == self.player else int(math.ceil(ships_t)) + 1
        if ck is not None:
            self.need_cache[ck] = need
        return need

    def reinforce_needed(self, pid, arrival_turn, planned=None):
        planned = planned or {}
        at = max(1, int(math.ceil(arrival_turn)))
        planet = self.planet_by_id[pid]
        if planet.owner != self.player:
            return self.ships_to_capture(pid, at, planned)
        arrivals = list(self.ledger.get(pid, []))
        arrivals += planned.get(pid, [])
        horizon = max(at + 5, self.timelines[pid]["horizon"])
        tl = simulate_timeline(planet, arrivals, self.player, horizon)
        worst = 0
        for t in range(at, min(horizon, at + 20) + 1):
            o = tl["owner_at"].get(t)
            s = tl["ships_at"].get(t, 0)
            if o != self.player:
                worst = max(worst, int(math.ceil(s)) + 1)
        return worst


# =============================================================================
# STRATEGY — GAME MODES
# =============================================================================

def build_modes(world):
    total = max(1, world.my_total + world.enemy_total)
    dom   = (world.my_total - world.enemy_total) / total
    behind   = dom < BEHIND_DOM
    ahead    = dom > AHEAD_DOM
    dominant = ahead or (world.max_enemy > 0 and world.my_total > world.max_enemy * 1.25)
    finishing = (dom > FINISHING_DOM
                 and world.my_prod > world.enemy_prod * FINISHING_PROD_RATIO
                 and world.step > 100)
    amm = 1.0
    if ahead:      amm += AHEAD_MARGIN_BONUS
    if behind:     amm -= BEHIND_MARGIN_PENALTY
    if finishing:  amm += FINISHING_MARGIN_BONUS
    kill_tgt = next(
        (o for o, s in world.owner_strength.items()
         if o != world.player and s <= KILL_THRESHOLD), None)
    return {"dom": dom, "behind": behind, "ahead": ahead,
            "dominant": dominant, "finishing": finishing,
            "amm": amm, "kill_tgt": kill_tgt}


def is_safe_neutral(tgt, world):
    if tgt.owner != -1:
        return False
    my_t, en_t = world.reaction_times(tgt.id)
    return my_t <= en_t - SAFE_NEUTRAL_MARGIN


def is_contested(tgt, world):
    if tgt.owner != -1:
        return False
    my_t, en_t = world.reaction_times(tgt.id)
    return abs(my_t - en_t) <= CONTESTED_NEUTRAL_MARGIN


# =============================================================================
# STRATEGY — OPENING FILTER
# =============================================================================

def opening_filter(target, arrival, needed, src_avail, world) -> bool:
    """
    Returns True if this move should be suppressed during the opening phase.
    2-player: more aggressive than 4-player.
    """
    if not world.is_opening or target.owner != -1 or target.id in world.comet_ids:
        return False
    if world.is_static_planet(target.id):
        return False
    my_t, en_t = world.reaction_times(target.id)
    gap = en_t - my_t

    if not world.is_four_player:
        # 2-player: grab high-prod if clearly first, OR medium-prod very fast
        if (target.production >= SAFE_OPENING_PROD_THRESHOLD
                and arrival <= SAFE_OPENING_TURN_LIMIT and gap >= SAFE_NEUTRAL_MARGIN):
            return False
        if target.production >= 3 and arrival <= 8 and gap >= 2:
            return False
        return (arrival > ROTATING_OPENING_MAX_TURNS
                or target.production <= ROTATING_OPENING_LOW_PROD)

    # 4-player
    if (target.production >= SAFE_OPENING_PROD_THRESHOLD
            and arrival <= SAFE_OPENING_TURN_LIMIT and gap >= SAFE_NEUTRAL_MARGIN):
        return False
    afford = needed <= max(PARTIAL_SRC_MIN,
                           int(src_avail * FOUR_PLAYER_ROTATING_SEND_RATIO))
    if afford and arrival <= FOUR_PLAYER_ROTATING_TURN_LIMIT and gap >= FOUR_PLAYER_ROTATING_REACTION_GAP:
        return False
    return True


# =============================================================================
# STRATEGY — SCORING
# =============================================================================

def target_value(tgt, arrival, mission, world, modes) -> float:
    """
    Production-compounding ROI value of capturing `tgt` with fleet arriving at `arrival`.
    Incorporates: planet type, ownership, race condition, game phase,
                  player count, counter-attack bonus, elimination acceleration.
    """
    turns_prod = max(1, world.remaining - arrival)
    if tgt.id in world.comet_ids:
        life = world.comet_life(tgt.id)
        turns_prod = max(0, min(turns_prod, life - arrival))
        if turns_prod <= 0:
            return -1.0

    v = tgt.production * turns_prod
    v += world.wealth_map[tgt.id] * turns_prod * INDIRECT_VALUE_SCALE

    if world.is_static_planet(tgt.id):
        v *= STATIC_NEUTRAL_MULT if tgt.owner == -1 else STATIC_HOSTILE_MULT
    elif world.is_opening:
        v *= ROTATING_OPENING_MULT

    if tgt.owner not in (-1, world.player):
        v *= OPENING_HOSTILE_MULT if world.is_opening else HOSTILE_MULT

    if tgt.owner == -1:
        if is_safe_neutral(tgt, world):
            v *= SAFE_NEUTRAL_MULT
        elif is_contested(tgt, world):
            v *= CONTESTED_NEUTRAL_MULT
        if world.is_early:
            v *= EARLY_NEUTRAL_MULT

    if tgt.id in world.comet_ids:
        v *= COMET_MULT

    mission_mult = {"snipe": SNIPE_MULT, "swarm": SWARM_MULT,
                    "reinforce": REINFORCE_MULT, "crash_exploit": CRASH_EXPLOIT_MULT}
    v *= mission_mult.get(mission, 1.0)

    if world.is_late:
        v += max(0, tgt.ships) * LATE_IMMEDIATE_SHIP_VALUE
        if tgt.owner not in (-1, world.player):
            if world.owner_strength.get(tgt.owner, 0) <= WEAK_ENEMY_THRESHOLD:
                v += ELIMINATION_BONUS

    if modes["finishing"] and tgt.owner not in (-1, world.player):
        v *= FINISHING_HOSTILE_MULT
    if modes["behind"] and tgt.owner == -1 and not world.is_static_planet(tgt.id):
        v *= BEHIND_ROTATING_NEUTRAL_MULT
    if modes["behind"] and tgt.owner == -1 and is_safe_neutral(tgt, world):
        v *= 1.08
    if modes["dominant"] and tgt.owner == -1 and is_contested(tgt, world):
        v *= 0.92

    # Counter-attack bonus: enemy planet recently sent ships → weakened garrison
    if tgt.owner not in (-1, world.player):
        sent  = world.enemy_sent_from.get(tgt.id, 0)
        total = tgt.ships + sent
        if total > 0 and sent / total >= WEAKENED_RATIO:
            v *= WEAKENED_MULT

    # Elimination acceleration
    if modes["kill_tgt"] is not None and tgt.owner == modes["kill_tgt"]:
        v *= KILL_MULT

    return v


def preferred_send(tgt, needed, arrival, cap, world, modes) -> int:
    send = max(needed, int(math.ceil(needed * modes["amm"])))
    m = 0
    if tgt.owner == -1:
        m += min(NEUTRAL_MARGIN_CAP, NEUTRAL_MARGIN_BASE + tgt.production * NEUTRAL_MARGIN_PROD_W)
    else:
        m += min(HOSTILE_MARGIN_CAP, HOSTILE_MARGIN_BASE + tgt.production * HOSTILE_MARGIN_PROD_W)
    if world.is_static_planet(tgt.id):           m += STATIC_TARGET_MARGIN
    if is_contested(tgt, world):                  m += CONTESTED_TARGET_MARGIN
    if world.is_four_player:                       m += FOUR_P_TARGET_MARGIN
    if arrival > LONG_TRAVEL_MARGIN_START:
        m += min(LONG_TRAVEL_MARGIN_CAP, arrival // LONG_TRAVEL_MARGIN_DIV)
    if tgt.id in world.comet_ids:                 m = max(0, m - COMET_MARGIN_RELIEF)
    if modes["finishing"] and tgt.owner not in (-1, world.player):
        m += FINISHING_HOSTILE_SEND_BONUS
    return min(cap, send + m)


def score_modifiers(base, tgt, mission, world) -> float:
    s = base
    if world.is_static_planet(tgt.id):
        s *= STATIC_SCORE_MULT
    if world.is_early and tgt.owner == -1 and world.is_static_planet(tgt.id):
        s *= EARLY_STATIC_NEUTRAL_S_MULT
    if world.is_four_player and tgt.owner == -1 and not world.is_static_planet(tgt.id):
        s *= FOUR_P_ROTATING_NEUTRAL_S_MULT
    if (len(world.static_neutral_planets) >= DENSE_STATIC_NEUTRAL_COUNT
            and tgt.owner == -1 and not world.is_static_planet(tgt.id)):
        s *= DENSE_ROTATING_NEUTRAL_S_MULT
    if mission == "snipe":  s *= SNIPE_SCORE_MULT
    elif mission == "swarm": s *= SWARM_SCORE_MULT
    return s


# =============================================================================
# STRATEGY — MISSION BUILDERS
# =============================================================================

def build_snipe(src, target, src_avail, world, planned, modes):
    if target.owner != -1:
        return None
    enemy_etas = sorted({
        int(math.ceil(e)) for e, o, s in world.ledger.get(target.id, [])
        if o not in (-1, world.player) and s > 0})
    if not enemy_etas:
        return None
    probe = min(src_avail, max(PARTIAL_SRC_MIN, int(target.ships) + 8))
    rough = world.plan_shot(src.id, target.id, probe)
    if rough is None:
        return None
    _, rt, _, _ = rough
    best = None
    for ee in enemy_etas:
        if abs(rt - ee) > 3:
            continue
        desired = ee - 1
        if desired < 1:
            continue
        need = world.ships_to_capture(target.id, desired, planned)
        if need <= 0 or need > int(src.ships):
            continue
        aim = world.plan_shot(src.id, target.id, need)
        if aim is None:
            continue
        angle, turns, _, _ = aim
        if abs(turns - desired) > 2:
            continue
        need = world.ships_to_capture(target.id, turns, planned)
        if need <= 0:
            continue
        v = target_value(target, turns, "snipe", world, modes)
        if v <= 0:
            continue
        sc = score_modifiers(v / (need + turns * SNIPE_COST_TURN_WEIGHT + 1.0),
                             target, "snipe", world)
        opt = ShotOption(sc, src.id, target.id, angle, turns, need, need, "snipe")
        m   = Mission("snipe", sc, target.id, turns, [opt])
        if best is None or sc > best.score:
            best = m
    return best


def build_reinforcements(world, planned, modes, src_left_fn):
    missions = []
    if not REINFORCE_ENABLED:
        return missions
    for pid, info in world.threatened.items():
        planet = world.planet_by_id[pid]
        fall   = info["fall_turn"]
        deficit = info["deficit"]
        sources = sorted(
            [p for p in world.my_planets if p.id != pid
             and src_left_fn(p.id) >= deficit],
            key=lambda p: travel_turns(p.x, p.y, p.radius,
                                       planet.x, planet.y, planet.radius,
                                       max(1, p.ships)))
        for src in sources[:3]:
            cap = min(src_left_fn(src.id), int(src.ships * REINFORCE_MAX_SRC_FRACTION))
            if cap < deficit:
                continue
            aim = world.plan_shot(src.id, pid, deficit)
            if aim is None:
                continue
            angle, turns, _, _ = aim
            if turns > REINFORCE_MAX_TRAVEL or turns >= fall:
                continue
            need = world.reinforce_needed(pid, turns, planned)
            if need <= 0:
                break
            v = target_value(planet, turns, "reinforce", world, modes)
            if v <= 0:
                continue
            sc  = score_modifiers(v / (need + turns * ATTACK_COST_TURN_WEIGHT + 1.0),
                                  planet, "reinforce", world)
            opt = ShotOption(sc, src.id, pid, angle, turns, need, cap, "reinforce")
            missions.append(Mission("reinforce", sc, pid, turns, [opt]))
            break
    return missions


def build_crash_exploits(world, planned, modes):
    missions = []
    for crash in world.crashes:
        tid    = crash["target_id"]
        target = world.planet_by_id[tid]
        desired = crash["crash_turn"] + CRASH_POST_DELAY
        best = None
        for src in world.my_planets:
            need = world.ships_to_capture(tid, desired, planned)
            if need <= 0 or need > int(src.ships):
                continue
            aim = world.plan_shot(src.id, tid, need)
            if aim is None:
                continue
            angle, turns, _, _ = aim
            if abs(turns - desired) > 2:
                continue
            need = world.ships_to_capture(tid, turns, planned)
            if need <= 0:
                continue
            v = target_value(target, turns, "crash_exploit", world, modes)
            if v <= 0:
                continue
            sc = score_modifiers(v / (need + turns * SNIPE_COST_TURN_WEIGHT + 1.0),
                                 target, "crash_exploit", world)
            opt = ShotOption(sc, src.id, tid, angle, turns, need, need, "crash_exploit")
            m   = Mission("crash_exploit", sc, tid, turns, [opt])
            if best is None or sc > best.score:
                best = m
        if best:
            missions.append(best)
    return missions


# =============================================================================
# STRATEGY — MOVE PLANNER
# =============================================================================

def plan_moves(world):
    modes   = build_modes(world)
    planned = defaultdict(list)   # planned_commitments[tid] = [(turn, player, ships)]
    src_opts_by_tgt = defaultdict(list)
    missions = []
    moves    = []
    spent    = defaultdict(int)

    def inv(sid):  return world.src_inventory(sid, spent)
    def atk(sid):  return world.src_attack(sid, spent)

    def emit(src_id, angle, ships):
        send = min(int(ships), inv(src_id))
        if send < 1:
            return 0
        moves.append([src_id, float(angle), int(send)])
        spent[src_id] += send
        return send

    # ── Reinforcement missions (compete on score with captures) ───────────────
    missions.extend(build_reinforcements(world, planned, modes, inv))

    # ── Build all single-source capture / snipe options ───────────────────────
    for src in world.my_planets:
        sa = atk(src.id)
        if sa <= 0:
            continue
        for tgt in world.planets:
            if tgt.id == src.id or tgt.owner == world.player:
                continue

            # Quick rough aim to get ETA
            rough_n   = max(1, min(sa, max(PARTIAL_SRC_MIN, int(tgt.ships) + 1)))
            rough_aim = world.plan_shot(src.id, tgt.id, rough_n)
            if rough_aim is None:
                continue
            _, rt, _, _ = rough_aim

            if world.is_very_late and rt > world.remaining - VERY_LATE_BUF:
                continue
            if tgt.id in world.comet_ids:
                cl = world.chase_limit(tgt.id)
                if rt >= world.comet_life(tgt.id) or rt > cl:
                    continue

            r_need = world.ships_to_capture(tgt.id, rt, planned)
            if r_need <= 0:
                continue
            if opening_filter(tgt, rt, r_need, sa, world):
                continue

            sg     = preferred_send(tgt, r_need, rt, sa, world, modes)
            aim    = world.plan_shot(src.id, tgt.id, max(1, sg))
            if aim is None:
                continue
            angle, turns, _, _ = aim

            if world.is_very_late and turns > world.remaining - VERY_LATE_BUF:
                continue
            if tgt.id in world.comet_ids:
                cl = world.chase_limit(tgt.id)
                if turns >= world.comet_life(tgt.id) or turns > cl:
                    continue

            need = world.ships_to_capture(tgt.id, turns, planned)
            if need <= 0:
                continue
            if opening_filter(tgt, turns, need, sa, world):
                continue

            cap = min(sa, preferred_send(tgt, need, turns, sa, world, modes))
            if cap < 1 or (cap < need and cap < PARTIAL_SRC_MIN):
                continue

            v = target_value(tgt, turns, "capture", world, modes)
            if v <= 0:
                continue
            exp_send = max(need, min(cap, preferred_send(tgt, need, turns, cap, world, modes)))
            sc = score_modifiers(
                v / (exp_send + turns * ATTACK_COST_TURN_WEIGHT + 1.0),
                tgt, "capture", world)

            opt = ShotOption(sc, src.id, tgt.id, angle, turns, need, cap, "capture")
            src_opts_by_tgt[tgt.id].append(opt)

            if cap >= need:
                missions.append(Mission("single", sc, tgt.id, turns, [opt]))

            snipe = build_snipe(src, tgt, sa, world, planned, modes)
            if snipe:
                missions.append(snipe)

    # ── Multi-source swarm assembly ───────────────────────────────────────────
    for tid, opts in src_opts_by_tgt.items():
        if len(opts) < 2:
            continue
        tgt   = world.planet_by_id[tid]
        top   = sorted(opts, key=lambda o: -o.score)[:MULTI_SRC_TOP_K]
        hostile = tgt.owner not in (-1, world.player)
        tol2  = HOSTILE_SWARM_ETA_TOL if hostile else MULTI_SRC_ETA_TOL

        # 2-source
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                a, b = top[i], top[j]
                if a.src_id == b.src_id or abs(a.turns - b.turns) > tol2:
                    continue
                jt   = max(a.turns, b.turns)
                need = world.ships_to_capture(tid, jt, planned)
                if need <= 0 or a.send_cap >= need or b.send_cap >= need:
                    continue
                if a.send_cap + b.send_cap < need:
                    continue
                v = target_value(tgt, jt, "swarm", world, modes)
                if v <= 0:
                    continue
                sc = score_modifiers(
                    v / (need + jt * ATTACK_COST_TURN_WEIGHT + 1.0),
                    tgt, "swarm", world) * MULTI_SRC_PENALTY
                missions.append(Mission("swarm", sc, tid, jt, [a, b]))

        # 3-source
        if THREE_SRC_ENABLED and hostile and int(tgt.ships) >= THREE_SRC_MIN_SHIPS and len(top) >= 3:
            for i in range(len(top)):
                for j in range(i+1, len(top)):
                    for k in range(j+1, len(top)):
                        trio = [top[i], top[j], top[k]]
                        if len({o.src_id for o in trio}) < 3:
                            continue
                        if max(o.turns for o in trio) - min(o.turns for o in trio) > THREE_SRC_ETA_TOL:
                            continue
                        jt   = max(o.turns for o in trio)
                        need = world.ships_to_capture(tid, jt, planned)
                        if need <= 0 or sum(o.send_cap for o in trio) < need:
                            continue
                        if any(trio[a].send_cap + trio[b].send_cap >= need
                               for a in range(3) for b in range(a+1, 3)):
                            continue
                        v = target_value(tgt, jt, "swarm", world, modes)
                        if v <= 0:
                            continue
                        sc = score_modifiers(
                            v / (need + jt * ATTACK_COST_TURN_WEIGHT + 1.0),
                            tgt, "swarm", world) * THREE_SRC_PENALTY
                        missions.append(Mission("swarm", sc, tid, jt, trio))

        # 4-source (very heavy targets)
        if FOUR_SRC_ENABLED and hostile and int(tgt.ships) >= FOUR_SRC_MIN_SHIPS and len(top) >= 4:
            for i in range(len(top)):
                for j in range(i+1, len(top)):
                    for k in range(j+1, len(top)):
                        for l in range(k+1, len(top)):
                            quad = [top[i], top[j], top[k], top[l]]
                            if len({o.src_id for o in quad}) < 4:
                                continue
                            sp = max(o.turns for o in quad) - min(o.turns for o in quad)
                            if sp > FOUR_SRC_ETA_TOL:
                                continue
                            jt   = max(o.turns for o in quad)
                            need = world.ships_to_capture(tid, jt, planned)
                            if need <= 0 or sum(o.send_cap for o in quad) < need:
                                continue
                            idxs = [(0,1,2),(0,1,3),(0,2,3),(1,2,3)]
                            if any(sum(quad[a].send_cap for a in idx) >= need for idx in idxs):
                                continue
                            v = target_value(tgt, jt, "swarm", world, modes)
                            if v <= 0:
                                continue
                            sc = score_modifiers(
                                v / (need + jt * ATTACK_COST_TURN_WEIGHT + 1.0),
                                tgt, "swarm", world) * FOUR_SRC_PENALTY
                            missions.append(Mission("swarm", sc, tid, jt, quad))

    # Crash-exploit missions
    missions.extend(build_crash_exploits(world, planned, modes))
    missions.sort(key=lambda m: -m.score)

    # ── Dispatch missions ─────────────────────────────────────────────────────
    for m in missions:
        tgt = world.planet_by_id[m.target_id]

        if m.kind in ("single", "snipe", "reinforce", "crash_exploit"):
            opt  = m.options[0]
            left = inv(opt.src_id) if m.kind == "reinforce" else atk(opt.src_id)
            if left <= 0:
                continue
            at      = opt.turns
            missing = (world.reinforce_needed(opt.target_id, at, planned)
                       if m.kind == "reinforce"
                       else world.ships_to_capture(tgt.id, at, planned))
            if missing <= 0:
                continue
            slimit = min(left, opt.send_cap)
            if slimit < missing:
                continue
            if m.kind in ("snipe", "crash_exploit"):
                send = missing
            elif m.kind == "reinforce":
                send = min(slimit, missing + REINFORCE_SAFETY_MARGIN)
            else:
                send = min(slimit, max(missing,
                                       preferred_send(tgt, missing, at, slimit, world, modes)))
            if send < missing:
                continue
            sent = emit(opt.src_id, opt.angle, send)
            if sent < missing:
                continue
            planned[tgt.id].append((at, world.player, int(sent)))
            continue

        # Swarm mission
        limits = [min(atk(o.src_id), o.send_cap) for o in m.options]
        if min(limits) <= 0:
            continue
        missing = world.ships_to_capture(tgt.id, m.turns, planned)
        if missing <= 0 or sum(limits) < missing:
            continue
        ordered   = sorted(zip(m.options, limits),
                           key=lambda x: (x[0].turns, -x[1], x[0].src_id))
        remaining = missing
        sends     = {}
        for idx, (opt, lim) in enumerate(ordered):
            other  = sum(l for _, l in ordered[idx+1:])
            s      = min(lim, max(0, remaining - other))
            sends[opt.src_id] = s
            remaining -= s
        if remaining > 0:
            continue
        committed = []
        for opt, _ in ordered:
            s = sends.get(opt.src_id, 0)
            if s <= 0:
                continue
            actual = emit(opt.src_id, opt.angle, s)
            if actual > 0:
                committed.append((opt.turns, world.player, int(actual)))
        if sum(c[2] for c in committed) < missing:
            continue
        planned[tgt.id].extend(committed)

    # ── Follow-up pass ────────────────────────────────────────────────────────
    if not world.is_very_late:
        for src in world.my_planets:
            left = atk(src.id)
            if left < FOLLOWUP_MIN:
                continue
            best = None
            for tgt in world.planets:
                if tgt.id == src.id or tgt.owner == world.player:
                    continue
                if tgt.id in world.comet_ids and tgt.production <= LOW_COMET_PROD:
                    continue
                rn = max(1, min(left, max(PARTIAL_SRC_MIN, int(tgt.ships) + 1)))
                ra = world.plan_shot(src.id, tgt.id, rn)
                if ra is None:
                    continue
                _, rt, _, _ = ra
                if world.is_late and rt > world.remaining - LATE_CAPTURE_BUF:
                    continue
                if tgt.id in world.comet_ids:
                    cl = world.chase_limit(tgt.id)
                    if rt >= world.comet_life(tgt.id) or rt > cl:
                        continue
                need = world.ships_to_capture(tgt.id, rt, planned)
                if need <= 0 or need > left:
                    continue
                if opening_filter(tgt, rt, need, left, world):
                    continue
                cap = min(left, preferred_send(tgt, need, rt, left, world, modes))
                if cap < need and cap < PARTIAL_SRC_MIN:
                    continue
                v = target_value(tgt, rt, "capture", world, modes)
                if v <= 0:
                    continue
                exp = max(need, min(cap, preferred_send(tgt, need, rt, cap, world, modes)))
                sc  = score_modifiers(v / (exp + rt * ATTACK_COST_TURN_WEIGHT + 1.0),
                                      tgt, "capture", world)
                if best is None or sc > best[0]:
                    best = (sc, tgt, need, cap, rt)
            if best is None:
                continue
            _, tgt, need, cap, rt = best
            aim = world.plan_shot(src.id, tgt.id, max(1, cap))
            if aim is None:
                continue
            angle, turns, _, _ = aim
            need2 = world.ships_to_capture(tgt.id, turns, planned)
            if need2 <= 0:
                continue
            sc2 = min(atk(src.id), preferred_send(tgt, need2, turns,
                                                   min(atk(src.id), cap),
                                                   world, modes))
            if sc2 < need2:
                continue
            sent = emit(src.id, angle, sc2)
            if sent >= need2:
                planned[tgt.id].append((turns, world.player, int(sent)))

    # ── Doomed-planet evacuation (toward frontier) ────────────────────────────
    for planet in world.my_planets:
        if planet.id not in world.doomed:
            continue
        avail = inv(planet.id)
        if avail < DOOMED_MIN_SHIPS:
            continue
        frontier = (world.enemy_planets or world.neutral_planets or world.my_planets)
        allies   = [p for p in world.my_planets
                    if p.id != planet.id and p.id not in world.doomed]
        if not allies:
            continue
        # Pick ally closest to frontier (pushes ships toward action)
        def ally_score(a):
            fd = min(pdist(a.x, a.y, f.x, f.y) for f in frontier)
            pd = pdist(planet.x, planet.y, a.x, a.y)
            return fd + 0.5 * pd
        dest = min(allies, key=ally_score)
        aim  = world.plan_shot(planet.id, dest.id, avail)
        if aim is None:
            continue
        emit(planet.id, aim[0], avail)

    # ── Rear-to-front logistics ───────────────────────────────────────────────
    if (world.enemy_planets or world.neutral_planets) and len(world.my_planets) > 1 and not world.is_late:
        ftargets = (world.enemy_planets
                    if world.enemy_planets
                    else (world.static_neutral_planets or world.neutral_planets))
        fdist = {p.id: min(pdist(p.x, p.y, f.x, f.y) for f in ftargets)
                 for p in world.my_planets}
        safe  = [p for p in world.my_planets if p.id not in world.doomed]
        if safe:
            anchor = min(safe, key=lambda p: fdist[p.id])
            ratio  = REAR_RATIO_4P if world.is_four_player else REAR_RATIO_2P
            if modes["finishing"]:
                ratio = max(ratio, REAR_RATIO_4P)
            for rear in sorted(world.my_planets, key=lambda p: -fdist[p.id]):
                if rear.id == anchor.id or rear.id in world.doomed:
                    continue
                if atk(rear.id) < REAR_SRC_MIN:
                    continue
                if fdist[rear.id] < fdist[anchor.id] * REAR_DIST_RATIO:
                    continue
                stage = [p for p in safe if p.id != rear.id
                         and fdist[p.id] < fdist[rear.id] * REAR_STAGE_PROG]
                if stage:
                    front = min(stage, key=lambda p: pdist(rear.x, rear.y, p.x, p.y))
                else:
                    obj   = min(ftargets, key=lambda t: pdist(rear.x, rear.y, t.x, t.y))
                    rems  = [p for p in safe if p.id != rear.id]
                    if not rems:
                        continue
                    front = min(rems, key=lambda p: pdist(p.x, p.y, obj.x, obj.y))
                if front.id == rear.id:
                    continue
                send = int(atk(rear.id) * ratio)
                if send < REAR_SEND_MIN:
                    continue
                aim = world.plan_shot(rear.id, front.id, send)
                if aim is None:
                    continue
                angle, turns, _, _ = aim
                if turns > REAR_MAX_TRAVEL:
                    continue
                emit(rear.id, angle, send)

    # ── Final inventory clamp (no merging — merging changes fleet speed) ──────
    final = []
    used  = defaultdict(int)
    for sid, angle, ships in moves:
        cap  = int(world.planet_by_id[sid].ships) - used[sid]
        send = min(int(ships), cap)
        if send >= 1:
            final.append([sid, float(angle), int(send)])
            used[sid] += send
    return final


# =============================================================================
# ENTRY POINT
# =============================================================================

def _r(obs, k, d=None):
    return obs.get(k, d) if isinstance(obs, dict) else getattr(obs, k, d)


def agent(obs):
    player   = _r(obs, "player", 0)
    step     = _r(obs, "step", 0) or 0
    ang_vel  = _r(obs, "angular_velocity", 0.0) or 0.0
    raw_pl   = _r(obs, "planets", []) or []
    raw_fl   = _r(obs, "fleets",  []) or []
    raw_init = _r(obs, "initial_planets", []) or []
    comets   = _r(obs, "comets", []) or []
    comet_ids = set(_r(obs, "comet_planet_ids", []) or [])

    planets       = [Planet(*p) for p in raw_pl]
    fleets        = [Fleet(*f)  for f in raw_fl]
    initial_by_id = {Planet(*p).id: Planet(*p) for p in raw_init}

    world = WorldModel(player, step, planets, fleets,
                       initial_by_id, ang_vel, comets, comet_ids)
    if not world.my_planets:
        return []
    return plan_moves(world)