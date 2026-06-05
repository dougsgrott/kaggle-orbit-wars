"""
Orbit Wars — definitive physics & game utilities.

Canonical source: clean_scripts/orbit_wars_physics_helper_module.py (v7).
Inter-script divergences documented inline as `# Variants:` comments.
Phase-1 exploration confirmed that fleet_speed, predict_planet_position,
predict_comet_position, is_static_planet, orbital_radius, dist,
point_to_segment_distance, launch_point, safe_angle_and_distance,
estimate_arrival, travel_time, arc_safe_angle, and probe_ship_candidates
are byte-identical across the high-scoring scripts in clean_scripts/.

Sections:
    A. Constants
    B. Geometry primitives
    C. Orbital geometry
    D. Physics & path safety
    E. Position prediction
    F. Aim solver (VERIFIED — every returned shot is forward-sim checked)
    G. Combat resolution + ships_needed_to_capture (simple & timeline)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Iterable, List

__all__ = [
    # constants
    "BOARD_SIZE", "CENTER_X", "CENTER_Y", "CENTER",
    "SUN_RADIUS", "SUN_SAFETY", "MAX_SHIP_SPEED",
    "ROTATION_LIMIT", "LAUNCH_CLEARANCE",
    "ROUTE_SEARCH_HORIZON", "HORIZON", "EPISODE_STEPS",
    "COMET_MAX_CHASE_TURNS", "ANG_VEL_MIN", "ANG_VEL_MAX",
    # geometry
    "dist", "point_to_segment_distance", "segment_intersects_circle",
    # orbital
    "orbital_radius", "is_static_planet",
    # physics
    "fleet_speed", "segment_hits_sun", "is_path_clear",
    "launch_point", "safe_angle_and_distance",
    # prediction
    "predict_planet_position", "predict_comet_position",
    "comet_remaining_life", "predict_target_position", "target_can_move",
    # aim solver
    "estimate_arrival", "estimate_arrival_frac", "travel_time",
    "arc_safe_angle", "search_safe_intercept",
    "aim_with_prediction", "intercept_aim", "probe_ship_candidates",
    # combat
    "resolve_combat",
    "ships_needed_to_capture_simple",
    "ships_needed_to_capture_timeline",
]


# ── A. Constants ──────────────────────────────────────────────────────────────
# From the official spec; values verified consistent across every clean_scripts/
# file that defines them.
BOARD_SIZE            = 100.0
CENTER_X              = 50.0
CENTER_Y              = 50.0
CENTER                = 50.0
SUN_RADIUS            = 10.0
SUN_SAFETY            = 1.5        # conservative buffer ON TOP of SUN_RADIUS
MAX_SHIP_SPEED        = 6.0
ROTATION_LIMIT        = 50.0       # orbital_radius + planet_radius >= 50 → static
LAUNCH_CLEARANCE      = 0.1
ROUTE_SEARCH_HORIZON  = 150        # v7: raised from 90 (covers speed=1 over dist=150)
HORIZON               = 110
EPISODE_STEPS         = 500
COMET_MAX_CHASE_TURNS = 10

ANG_VEL_MIN = 0.025
ANG_VEL_MAX = 0.050

_FWD_ITER_MAX   = 16
_EDGE_AIM_FRACS = (0.25, 0.50, 0.75, 0.95)


# ── B. Geometry primitives ────────────────────────────────────────────────────

def dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def point_to_segment_distance(px: float, py: float,
                              x1: float, y1: float,
                              x2: float, y2: float) -> float:
    dx, dy = x2 - x1, y2 - y1
    seg_sq = dx * dx + dy * dy
    if seg_sq < 1e-9:
        return dist(px, py, x1, y1)
    t = ((px - x1) * dx + (py - y1) * dy) / seg_sq
    t = max(0.0, min(1.0, t))
    return dist(px, py, x1 + t * dx, y1 + t * dy)


def segment_intersects_circle(ax: float, ay: float,
                              bx: float, by: float,
                              cx: float, cy: float,
                              r: float) -> bool:
    return point_to_segment_distance(cx, cy, ax, ay, bx, by) <= r


# ── C. Orbital geometry ───────────────────────────────────────────────────────

def orbital_radius(px: float, py: float) -> float:
    return dist(px, py, CENTER_X, CENTER_Y)


def is_static_planet(px: float, py: float, radius: float) -> bool:
    return orbital_radius(px, py) + radius >= ROTATION_LIMIT


# ── D. Physics & path safety ──────────────────────────────────────────────────

def fleet_speed(ships: int) -> float:
    """Spec formula: speed = 1 + (max-1) * (log(ships)/log(1000))^1.5."""
    if ships <= 1:
        return 1.0
    ratio = math.log(ships) / math.log(1000.0)
    ratio = max(0.0, min(1.0, ratio))
    return min(1.0 + (MAX_SHIP_SPEED - 1.0) * ratio ** 1.5, MAX_SHIP_SPEED)


# Variants: orbit_wars_2026_starter.py:25 uses a quadratic ray-parameterisation
# instead of the perpendicular-projection algebra below; the two are
# mathematically equivalent. Buffer (SUN_RADIUS + 1.5 = 11.5) is universal
# across all scripts — do not tighten it; smaller values lose fleets to grazing.
def segment_hits_sun(x1: float, y1: float,
                     x2: float, y2: float,
                     safety: float = SUN_SAFETY) -> bool:
    return point_to_segment_distance(
        CENTER_X, CENTER_Y, x1, y1, x2, y2
    ) < SUN_RADIUS + safety


def is_path_clear(sx: float, sy: float, tx: float, ty: float) -> bool:
    return not segment_hits_sun(sx, sy, tx, ty)


def launch_point(sx: float, sy: float,
                 sr: float, angle: float) -> Tuple[float, float]:
    c = sr + LAUNCH_CLEARANCE
    return sx + math.cos(angle) * c, sy + math.sin(angle) * c


def safe_angle_and_distance(sx: float, sy: float, sr: float,
                            tx: float, ty: float, tr: float
                            ) -> Optional[Tuple[float, float]]:
    angle = math.atan2(ty - sy, tx - sx)
    lx, ly = launch_point(sx, sy, sr, angle)
    hit_dist = max(0.0, dist(sx, sy, tx, ty) - sr - LAUNCH_CLEARANCE - tr)
    ex = lx + math.cos(angle) * hit_dist
    ey = ly + math.sin(angle) * hit_dist
    if segment_hits_sun(lx, ly, ex, ey):
        return None
    return angle, hit_dist


# ── E. Position prediction ────────────────────────────────────────────────────

def predict_planet_position(planet_id: int,
                            cur_x: float, cur_y: float, radius: float,
                            initial_by_id: dict,
                            angular_velocity: float,
                            turns_ahead: int) -> Tuple[float, float]:
    init = initial_by_id.get(planet_id)
    if init is None:
        return cur_x, cur_y
    ix, iy = init["x"], init["y"]
    r = dist(ix, iy, CENTER_X, CENTER_Y)
    if r + radius >= ROTATION_LIMIT:
        return cur_x, cur_y
    cur_ang = math.atan2(cur_y - CENTER_Y, cur_x - CENTER_X)
    new_ang = cur_ang + angular_velocity * turns_ahead
    return CENTER_X + r * math.cos(new_ang), CENTER_Y + r * math.sin(new_ang)


def predict_comet_position(planet_id: int, comets: list,
                           turns: int) -> Optional[Tuple[float, float]]:
    for group in comets:
        pids = group.get("planet_ids", [])
        if planet_id not in pids:
            continue
        idx        = pids.index(planet_id)
        paths      = group.get("paths", [])
        path_index = group.get("path_index", 0)
        if idx >= len(paths):
            return None
        future = path_index + int(turns)
        path   = paths[idx]
        if 0 <= future < len(path):
            return path[future][0], path[future][1]
        return None
    return None


def comet_remaining_life(planet_id: int, comets: list) -> int:
    for group in comets:
        pids = group.get("planet_ids", [])
        if planet_id not in pids:
            continue
        idx        = pids.index(planet_id)
        paths      = group.get("paths", [])
        path_index = group.get("path_index", 0)
        if idx < len(paths):
            return max(0, len(paths[idx]) - path_index)
    return 0


def predict_target_position(planet_id: int,
                            cur_x: float, cur_y: float, radius: float,
                            initial_by_id: dict,
                            angular_velocity: float,
                            comets: list, comet_ids: set,
                            turns: int) -> Optional[Tuple[float, float]]:
    if planet_id in comet_ids:
        return predict_comet_position(planet_id, comets, turns)
    return predict_planet_position(
        planet_id, cur_x, cur_y, radius,
        initial_by_id, angular_velocity, turns)


def target_can_move(planet_id: int,
                    cur_x: float, cur_y: float, radius: float,
                    initial_by_id: dict, comet_ids: set) -> bool:
    if planet_id in comet_ids:
        return True
    init = initial_by_id.get(planet_id)
    if init is None:
        return False
    r = dist(init["x"], init["y"], CENTER_X, CENTER_Y)
    return r + radius < ROTATION_LIMIT


# ── F. Aim solver ─────────────────────────────────────────────────────────────

def _fractional_turns(total_d: float, ships: int) -> float:
    return total_d / fleet_speed(max(1, ships))


def estimate_arrival(sx: float, sy: float, sr: float,
                     tx: float, ty: float, tr: float,
                     ships: int) -> Optional[Tuple[float, int]]:
    result = safe_angle_and_distance(sx, sy, sr, tx, ty, tr)
    if result is None:
        return None
    angle, total_d = result
    turns = max(1, int(math.ceil(_fractional_turns(total_d, ships))))
    return angle, turns


def estimate_arrival_frac(sx: float, sy: float, sr: float,
                          tx: float, ty: float, tr: float,
                          ships: int) -> Optional[Tuple[float, float]]:
    result = safe_angle_and_distance(sx, sy, sr, tx, ty, tr)
    if result is None:
        return None
    angle, total_d = result
    return angle, max(1.0, _fractional_turns(total_d, ships))


def travel_time(sx: float, sy: float, sr: float,
                tx: float, ty: float, tr: float,
                ships: int) -> int:
    est = estimate_arrival(sx, sy, sr, tx, ty, tr, ships)
    return est[1] if est is not None else 10 ** 9


def _fwd_window(turns: int) -> int:
    return max(8, turns // 2)


def arc_safe_angle(sx: float, sy: float, sr: float,
                   tx: float, ty: float, tr: float,
                   ships: int) -> Optional[Tuple[float, int]]:
    dx, dy = tx - sx, ty - sy
    d = math.hypot(dx, dy)
    if d < 1e-9:
        return None
    ux, uy = dx / d, dy / d
    nx, ny = -uy, ux

    aim_points = [(tx, ty)]
    for frac in _EDGE_AIM_FRACS:
        off = tr * frac
        aim_points.append((tx + nx * off, ty + ny * off))
        aim_points.append((tx - nx * off, ty - ny * off))

    best = None
    for ax, ay in aim_points:
        angle      = math.atan2(ay - sy, ax - sx)
        lx, ly     = launch_point(sx, sy, sr, angle)
        rvx, rvy   = math.cos(angle), math.sin(angle)
        cx, cy     = tx - lx, ty - ly
        proj       = cx * rvx + cy * rvy
        closest_sq = cx * cx + cy * cy - proj * proj
        if proj <= 0.0 or closest_sq > tr * tr:
            continue
        entry_dist = max(0.0, proj - math.sqrt(max(0.0, tr * tr - closest_sq)))
        ex = lx + rvx * entry_dist
        ey = ly + rvy * entry_dist
        if segment_hits_sun(lx, ly, ex, ey):
            continue
        turns = max(1, int(math.ceil(entry_dist / fleet_speed(max(1, ships)))))
        score = (turns, entry_dist)
        if best is None or score < best[0]:
            best = (score, angle, turns)

    return (best[1], best[2]) if best else None


def _verify_shot_hits(sx: float, sy: float, sr: float,
                      angle: float, turns: int, ships: int,
                      target_id: int,
                      tx: float, ty: float, tr: float,
                      initial_by_id: dict,
                      angular_velocity: float,
                      comets: list, comet_ids: set) -> bool:
    """Ground-truth forward-sim: True only if the fleet physically hits the
    target within the scan window. Used to gate EVERY result before it leaves
    aim_with_prediction()."""
    speed  = fleet_speed(max(1, ships))
    fx, fy = launch_point(sx, sy, sr, angle)
    vx, vy = math.cos(angle) * speed, math.sin(angle) * speed
    window = _fwd_window(turns)

    for t in range(1, turns + window + 1):
        pfx, pfy = fx, fy
        fx += vx
        fy += vy
        if segment_hits_sun(pfx, pfy, fx, fy):
            return False
        pos = predict_target_position(
            target_id, tx, ty, tr,
            initial_by_id, angular_velocity,
            comets, comet_ids, t)
        if pos is None:
            continue
        if segment_intersects_circle(pfx, pfy, fx, fy, pos[0], pos[1], tr):
            return True
    return False


def _dynamic_tolerance(target_id: int,
                       initial_by_id: dict,
                       angular_velocity: float,
                       comet_ids: set) -> int:
    if target_id in comet_ids:
        return 2
    init = initial_by_id.get(target_id)
    if init is None:
        return 1
    orb_r     = dist(init["x"], init["y"], CENTER_X, CENTER_Y)
    orb_speed = orb_r * abs(angular_velocity)
    return 2 if orb_speed >= 1.0 else 1


# Variants: an earlier version of this function used tolerance=3 and
# ROUTE_SEARCH_HORIZON=90. Both were superseded: tolerance>2 caused
# search_safe_intercept to pick the wrong orbital position; horizon<150
# missed valid slow-fleet shots across the board. Do not lower either.
def search_safe_intercept(sx: float, sy: float, sr: float,
                          target_id: int,
                          tx: float, ty: float, tr: float,
                          ships: int,
                          initial_by_id: dict,
                          angular_velocity: float,
                          comets: list, comet_ids: set,
                          tolerance: int = None,
                          ) -> Optional[Tuple[float, int, float, float]]:
    if tolerance is None:
        tolerance = _dynamic_tolerance(target_id, initial_by_id,
                                       angular_velocity, comet_ids)
    max_turns = ROUTE_SEARCH_HORIZON
    if target_id in comet_ids:
        max_turns = min(max_turns,
                        max(0, comet_remaining_life(target_id, comets) - 1))

    for candidate_turns in range(1, max_turns + 1):
        pos = predict_target_position(
            target_id, tx, ty, tr,
            initial_by_id, angular_velocity,
            comets, comet_ids, candidate_turns)
        if pos is None:
            continue

        est = estimate_arrival(sx, sy, sr, pos[0], pos[1], tr, ships)
        if est is None:
            est = arc_safe_angle(sx, sy, sr, pos[0], pos[1], tr, ships)
            if est is None:
                continue

        _, turns = est
        if abs(turns - candidate_turns) > tolerance:
            continue

        actual_turns = max(turns, candidate_turns)
        actual_pos   = predict_target_position(
            target_id, tx, ty, tr,
            initial_by_id, angular_velocity,
            comets, comet_ids, actual_turns)
        if actual_pos is None:
            continue

        confirm = estimate_arrival(sx, sy, sr, actual_pos[0], actual_pos[1], tr, ships)
        if confirm is None:
            confirm = arc_safe_angle(sx, sy, sr, actual_pos[0], actual_pos[1], tr, ships)
            if confirm is None:
                continue

        delta = abs(confirm[1] - actual_turns)
        if delta > tolerance:
            continue

        angle_out, turns_out = confirm[0], confirm[1]
        if _verify_shot_hits(sx, sy, sr, angle_out, turns_out, ships,
                             target_id, tx, ty, tr,
                             initial_by_id, angular_velocity,
                             comets, comet_ids):
            return (angle_out, turns_out, actual_pos[0], actual_pos[1])

    return None


def _aim_raw(sx: float, sy: float, sr: float,
             target_id: int,
             tx: float, ty: float, tr: float,
             ships: int,
             initial_by_id: dict,
             angular_velocity: float,
             comets: list, comet_ids: set,
             ) -> Optional[Tuple[float, int, float, float]]:
    """Iterative convergence solver. Results are UNVERIFIED — the public
    aim_with_prediction() runs _verify_shot_hits() on this output before
    returning. Do not call directly unless you also verify."""
    tol = _dynamic_tolerance(target_id, initial_by_id, angular_velocity, comet_ids)

    est = estimate_arrival_frac(sx, sy, sr, tx, ty, tr, ships)
    if est is None:
        est_arc = arc_safe_angle(sx, sy, sr, tx, ty, tr, ships)
        if est_arc is None:
            if not target_can_move(target_id, tx, ty, tr, initial_by_id, comet_ids):
                angle   = math.atan2(ty - sy, tx - sx)
                total_d = max(0.0, dist(sx, sy, tx, ty) - sr - LAUNCH_CLEARANCE - tr)
                turns   = max(1, int(math.ceil(total_d / fleet_speed(max(1, ships)))))
                return angle, turns, tx, ty
            return None
        return est_arc[0], est_arc[1], tx, ty

    for _ in range(_FWD_ITER_MAX):
        _, turns_f = est
        turns_i    = int(math.ceil(turns_f))
        pos        = predict_target_position(
            target_id, tx, ty, tr,
            initial_by_id, angular_velocity,
            comets, comet_ids, turns_i)
        if pos is None:
            return None
        ntx, nty = pos
        next_est = estimate_arrival_frac(sx, sy, sr, ntx, nty, tr, ships)
        if next_est is None:
            arc = arc_safe_angle(sx, sy, sr, ntx, nty, tr, ships)
            if arc:
                return arc[0], arc[1], ntx, nty
            return None
        _, next_turns_f = next_est
        if abs(next_turns_f - turns_f) <= tol:
            angle_int = estimate_arrival(sx, sy, sr, ntx, nty, tr, ships)
            if angle_int is None:
                arc = arc_safe_angle(sx, sy, sr, ntx, nty, tr, ships)
                return (arc[0], arc[1], ntx, nty) if arc else None
            return angle_int[0], angle_int[1], ntx, nty
        est = next_est

    final_pos = predict_target_position(
        target_id, tx, ty, tr,
        initial_by_id, angular_velocity,
        comets, comet_ids, int(math.ceil(est[1])))
    if final_pos is None:
        return None
    refined = estimate_arrival(sx, sy, sr, final_pos[0], final_pos[1], tr, ships)
    if refined is None:
        arc = arc_safe_angle(sx, sy, sr, final_pos[0], final_pos[1], tr, ships)
        return (arc[0], arc[1], final_pos[0], final_pos[1]) if arc else None
    return refined[0], refined[1], final_pos[0], final_pos[1]


# Variants: orbit_wars_2026_starter.py:119 returns _aim_raw output WITHOUT the
# _verify_shot_hits gate — that variant will emit shots that miss on rotating
# targets. orbit_wars_robust_agent.py:488 (named intercept) uses a pure
# exhaustive scan with no fast-path; correct but slower. Two-stage (fast raw
# + verify + exhaustive fallback) below is the v7 canonical and only one
# whose non-None results are guaranteed verified.
def aim_with_prediction(sx: float, sy: float, sr: float,
                        target_id: int,
                        tx: float, ty: float, tr: float,
                        ships: int,
                        initial_by_id: dict,
                        angular_velocity: float,
                        comets: list, comet_ids: set,
                        ) -> Optional[Tuple[float, int, float, float]]:
    """Public solver. Returns (angle, turns, target_x, target_y) or None.
    Every non-None result has been forward-sim verified."""
    res = _aim_raw(sx, sy, sr, target_id, tx, ty, tr, ships,
                   initial_by_id, angular_velocity, comets, comet_ids)
    if res is not None:
        angle, turns, _, _ = res
        if _verify_shot_hits(sx, sy, sr, angle, turns, ships,
                             target_id, tx, ty, tr,
                             initial_by_id, angular_velocity,
                             comets, comet_ids):
            return res

    fallback = search_safe_intercept(
        sx, sy, sr, target_id, tx, ty, tr,
        ships, initial_by_id, angular_velocity, comets, comet_ids)
    if fallback is not None:
        return fallback

    return None


# ── F2. Continuous-intercept aim (AG14) ───────────────────────────────────────
# The Producer's aim, ported: a sub-turn continuous-fixed-point lead + a
# byte-exact **swept-pair** first-contact verify. The difference from
# `aim_with_prediction` is the verify: `_verify_shot_hits` checks the target's
# *endpoint* position each integer turn, which approves near-misses on fast-orbiting
# targets (measured ~4.8% of flow_value's fleets sail OOB/into the sun). The
# swept-pair tests the fleet segment against the target's *moving* segment over the
# step — the engine's actual rule — so it rejects those near-misses (we keep the
# ships home) and the continuous lead finds better angles. Same signature as
# `aim_with_prediction`, so it is a drop-in aimer (see `agents/flow_value_ia`).


def _swept_point_circle_hit(ax: float, ay: float, bx: float, by: float,
                            p0x: float, p0y: float, p1x: float, p1y: float,
                            r: float) -> bool:
    """True iff a point moving ``A→B`` contacts a circle of radius ``r`` whose
    centre moves ``P0→P1`` over the same unit step (the engine swept-pair test)."""
    d0x, d0y = ax - p0x, ay - p0y
    dvx, dvy = (bx - ax) - (p1x - p0x), (by - ay) - (p1y - p0y)
    a = dvx * dvx + dvy * dvy
    c = d0x * d0x + d0y * d0y - r * r
    if a < 1e-12:
        return c <= 0.0
    b = 2.0 * (d0x * dvx + d0y * dvy)
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)
    return t2 >= 0.0 and t1 <= 1.0


def _verify_swept(sx: float, sy: float, sr: float,
                  angle: float, turns: int, ships: int,
                  target_id: int, tx: float, ty: float, tr: float,
                  initial_by_id: dict, angular_velocity: float,
                  comets: list, comet_ids: set) -> bool:
    """Swept-pair forward verify: True iff the fleet contacts the *moving* target
    before going out of bounds or grazing the sun. Planet contact resolves before
    env removal in a step, so target contact is checked first (engine rule)."""
    speed = fleet_speed(max(1, ships))
    fx, fy = launch_point(sx, sy, sr, angle)
    vx, vy = math.cos(angle) * speed, math.sin(angle) * speed
    window = _fwd_window(turns)
    p_prev = predict_target_position(
        target_id, tx, ty, tr, initial_by_id, angular_velocity, comets, comet_ids, 0)
    if p_prev is None:
        p_prev = (tx, ty)
    for t in range(1, turns + window + 1):
        pfx, pfy = fx, fy
        fx += vx
        fy += vy
        p_now = predict_target_position(
            target_id, tx, ty, tr, initial_by_id, angular_velocity, comets, comet_ids, t)
        if p_now is not None:
            if _swept_point_circle_hit(pfx, pfy, fx, fy,
                                       p_prev[0], p_prev[1], p_now[0], p_now[1], tr):
                return True  # contact wins the step
            p_prev = p_now
        if segment_hits_sun(pfx, pfy, fx, fy):
            return False
        if fx < 0.0 or fx > BOARD_SIZE or fy < 0.0 or fy > BOARD_SIZE:
            return False
    return False


def _intercept_lead(sx: float, sy: float, sr: float,
                    target_id: int, tx: float, ty: float, tr: float, ships: int,
                    initial_by_id: dict, angular_velocity: float,
                    comets: list, comet_ids: set,
                    horizon: float = float(ROUTE_SEARCH_HORIZON),
                    iters: int = 6):
    """Continuous fixed-point intercept: solves ``t = (dist(target(t), src) − gap)
    / speed`` (no grid scan) and aims at the target's position at ``t``. Returns
    ``(angle, turns, px, py)``."""
    speed = max(1e-6, fleet_speed(max(1, ships)))
    gap = sr + LAUNCH_CLEARANCE + tr
    t = max(0.0, (dist(sx, sy, tx, ty) - gap) / speed)
    px, py = tx, ty
    for _ in range(iters):
        pos = predict_target_position(
            target_id, tx, ty, tr, initial_by_id, angular_velocity, comets, comet_ids, t)
        if pos is not None:
            px, py = pos
        t = max(0.0, min(float(horizon), (dist(sx, sy, px, py) - gap) / speed))
    angle = math.atan2(py - sy, px - sx)
    turns = max(1, int(math.ceil(t)))
    return angle, turns, px, py


def intercept_aim(sx: float, sy: float, sr: float,
                  target_id: int, tx: float, ty: float, tr: float, ships: int,
                  initial_by_id: dict, angular_velocity: float,
                  comets: list, comet_ids: set,
                  ) -> Optional[Tuple[float, int, float, float]]:
    """Continuous-intercept solver (AG14). Drop-in for ``aim_with_prediction``:
    same args, returns ``(angle, turns, px, py)`` or ``None``, every result
    swept-pair verified."""
    angle, turns, px, py = _intercept_lead(
        sx, sy, sr, target_id, tx, ty, tr, ships,
        initial_by_id, angular_velocity, comets, comet_ids)
    if _verify_swept(sx, sy, sr, angle, turns, ships, target_id, tx, ty, tr,
                     initial_by_id, angular_velocity, comets, comet_ids):
        return angle, turns, px, py
    # fall back to the iterative solver, but hold it to the same swept-pair verify.
    res = _aim_raw(sx, sy, sr, target_id, tx, ty, tr, ships,
                   initial_by_id, angular_velocity, comets, comet_ids)
    if res is not None:
        a, k, rpx, rpy = res
        if _verify_swept(sx, sy, sr, a, k, ships, target_id, tx, ty, tr,
                         initial_by_id, angular_velocity, comets, comet_ids):
            return a, k, rpx, rpy
    return None


def probe_ship_candidates(need: int = None, avail: int = 0,
                          ships: int = None) -> List[int]:
    if need is None:
        need = ships
    if need is None:
        return []
    candidates = sorted(set([
        max(1, int(0.25 * need)),
        max(1, int(0.50 * need)),
        max(1, int(0.75 * need)),
        max(1, need - 5),
        need,
        min(avail, need + 5),
        min(avail, need + 10),
    ]))
    return [c for c in candidates if 1 <= c <= avail]


# ── G. Combat resolution & capture-cost ───────────────────────────────────────

# Variants: physics_helper_module.py expresses combat implicitly inside its
# search loops. robust_agent.py:586 has an explicit resolve_combat. The form
# below is from complete_game_mechanics_deep_dive.py:248 — clearest, with
# explicit handling of the three rule cases (multi-attacker tie / single
# attacker / reinforcement of own).
def resolve_combat(planet_owner: int,
                   planet_ships: int,
                   arriving_fleets: Iterable[Tuple[int, int]]
                   ) -> Tuple[int, int]:
    """Exact engine combat logic.

    arriving_fleets: iterable of (owner, ships) tuples landing this turn.
    Returns (new_owner, new_ships)."""
    player_ships: dict = {}
    for owner, ships in arriving_fleets:
        player_ships[owner] = player_ships.get(owner, 0) + ships

    if not player_ships:
        return planet_owner, planet_ships

    sorted_players = sorted(player_ships.items(), key=lambda x: x[1], reverse=True)
    top_player, top_ships = sorted_players[0]

    if len(sorted_players) > 1:
        second_ships = sorted_players[1][1]
        if top_ships == second_ships:
            survivor_ships = 0
            survivor_owner = -1
        else:
            survivor_ships = top_ships - second_ships
            survivor_owner = top_player
    else:
        survivor_ships = top_ships
        survivor_owner = top_player

    if survivor_ships == 0:
        return planet_owner, planet_ships

    if planet_owner == survivor_owner:
        return planet_owner, planet_ships + survivor_ships

    remaining = planet_ships - survivor_ships
    if remaining < 0:
        return survivor_owner, abs(remaining)
    return planet_owner, remaining


def ships_needed_to_capture_simple(defender_ships: int,
                                   defender_owner: int,
                                   attacker_owner: int,
                                   other_arrivals: Iterable[Tuple[int, int]] = ()
                                   ) -> int:
    """Minimum attacker ships landing in the SAME turn to flip the planet.

    Use when you don't care about production or multi-turn dynamics — just a
    one-shot capture check. `other_arrivals` are co-arriving fleets from
    other players that turn (NOT the attacker's own fleet).

    Returns 0 if the planet is already attacker-owned and unthreatened.
    """
    if defender_owner == attacker_owner and not list(other_arrivals):
        return 0

    other = list(other_arrivals)

    # Lower bound: at minimum, must beat second-place after combat.
    # Search upward by doubling, then binary-search.
    def captures(n: int) -> bool:
        new_owner, _ = resolve_combat(
            defender_owner, defender_ships,
            other + [(attacker_owner, n)])
        return new_owner == attacker_owner

    if captures(1):
        # Even 1 ship suffices (rare: e.g. attacker already owns + reinforcement).
        return 1

    hi = max(1, defender_ships + sum(s for _, s in other) + 1)
    while not captures(hi):
        hi *= 2
        if hi > 10 ** 7:
            return hi  # pathological — caller should treat as infeasible
    lo = 1
    while lo < hi:
        mid = (lo + hi) // 2
        if captures(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


# Variants: lb_1200_orbit_wars_ppo_strategy.py:1002 (min_ships_to_own_by) and
# lb_max_1224's WorldModel use this same algorithm but cache results and lean
# on a pre-built per-planet timeline. The standalone version below takes the
# defender's state + production explicitly so it can be called without
# building the full WorldModel infrastructure.
def ships_needed_to_capture_timeline(defender_ships: int,
                                     defender_owner: int,
                                     defender_production: int,
                                     attacker_owner: int,
                                     arrival_turn: int,
                                     scheduled_arrivals: Iterable[Tuple[int, int, int]] = (),
                                     ) -> int:
    """Production-aware multi-wave capture cost.

    Simulates planet state from turn 1 up to `arrival_turn`, applying
    each scheduled fleet at its arrival turn AND adding production each
    turn the planet has an owner != -1. Returns the minimum attacker ships
    arriving at `arrival_turn` such that the planet is attacker-owned
    immediately after combat at that turn.

    scheduled_arrivals: iterable of (turn, owner, ships) for fleets ALREADY
    in flight (excluding the attacker's hypothetical fleet under search).
    Use when you need to account for production accrual or pre-existing
    co-arriving waves — i.e. the typical mission-planner question.
    """
    arrivals = [
        (max(1, int(turn)), int(owner), int(ships))
        for turn, owner, ships in scheduled_arrivals
        if int(ships) > 0 and int(turn) <= arrival_turn
    ]
    arrivals_by_turn: dict = {}
    for t, o, s in arrivals:
        arrivals_by_turn.setdefault(t, []).append((o, s))

    arrival_turn = max(1, int(arrival_turn))

    def captures(attacker_ships: int) -> bool:
        owner = defender_owner
        ships = int(defender_ships)
        for t in range(1, arrival_turn + 1):
            if owner != -1 and t > 1:
                ships += defender_production
            fleets = list(arrivals_by_turn.get(t, []))
            if t == arrival_turn:
                fleets.append((attacker_owner, attacker_ships))
            if fleets:
                owner, ships = resolve_combat(owner, ships, fleets)
        return owner == attacker_owner

    if captures(0):
        return 0

    # Doubling search for hi, binary search to refine.
    hi = max(1, defender_ships + defender_production * arrival_turn + 1)
    while not captures(hi):
        hi *= 2
        if hi > 10 ** 7:
            return hi
    lo = 1
    while lo < hi:
        mid = (lo + hi) // 2
        if captures(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo
