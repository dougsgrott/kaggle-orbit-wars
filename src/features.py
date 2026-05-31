"""
ML feature extraction for shot validation.

Lifted from clean_scripts/train_submit_v4_ml_validator_topk2_tutorial.py
(catalogue score 899.7 — highest measured score in the corpus).

- encode_shot: 24-dim hand-crafted feature vector per shot
- find_target_via_ray: recover implicit target_id from (src, angle)
- label_outcome: windowed success label (10-turn ownership window)
"""

from __future__ import annotations

import math
import numpy as np

from .utils import fleet_speed, BOARD_SIZE, MAX_SHIP_SPEED

FEATURE_DIM = 24


def encode_shot(obs: dict, src_id: int, target_id: int, ships_sent: int):
    """Return a 24-dim float32 feature vector for a single shot.

    Returns None if src_id or target_id are missing from obs["planets"].
    All features are normalized so values cluster around [-1, 1].
    """
    pdict = {int(p[0]): p for p in obs["planets"]}
    if src_id not in pdict or target_id not in pdict:
        return None
    src = pdict[src_id]
    tgt = pdict[target_id]
    me = int(obs.get("player", 0))
    fleets = obs.get("fleets", [])
    planets = obs["planets"]

    my_ships_total    = sum(int(p[5]) for p in planets if int(p[1]) == me)
    enemy_ships_total = sum(int(p[5]) for p in planets if int(p[1]) >= 0 and int(p[1]) != me)
    my_planets    = sum(1 for p in planets if int(p[1]) == me)
    enemy_planets = sum(1 for p in planets if int(p[1]) >= 0 and int(p[1]) != me)

    sx, sy, sr, sships = float(src[2]), float(src[3]), float(src[4]), int(src[5])
    tx, ty, tr, tships = float(tgt[2]), float(tgt[3]), float(tgt[4]), int(tgt[5])
    sprod, tprod = float(src[6]), float(tgt[6])
    dx, dy = tx - sx, ty - sy
    distance = max(math.hypot(dx, dy) - sr - tr, 0.0)
    speed = fleet_speed(ships_sent)
    eta = distance / max(speed, 0.5)

    own_self    = 1.0 if int(tgt[1]) == me else 0.0
    own_neutral = 1.0 if int(tgt[1]) < 0 else 0.0
    own_enemy   = 1.0 if (int(tgt[1]) >= 0 and int(tgt[1]) != me) else 0.0
    ship_frac   = ships_sent / max(sships, 1)

    ally_n  = sum(1 for f in fleets if int(f[1]) == me)
    ally_s  = sum(int(f[6]) for f in fleets if int(f[1]) == me)
    enemy_n = sum(1 for f in fleets if int(f[1]) != me)
    enemy_s = sum(int(f[6]) for f in fleets if int(f[1]) != me)
    turn = int(obs.get("step", 0))

    return np.array([
        sships / 100.0, sprod / 5.0, sr / 4.0,
        tships / 100.0, tprod / 5.0, tr / 4.0,
        own_self, own_neutral, own_enemy,
        ships_sent / 100.0, ship_frac,
        distance / BOARD_SIZE, eta / 60.0, speed / MAX_SHIP_SPEED,
        ally_n / 10.0, ally_s / 100.0, enemy_n / 10.0, enemy_s / 100.0,
        turn / 500.0, my_ships_total / 200.0, enemy_ships_total / 200.0,
        (my_ships_total - enemy_ships_total) / 200.0,
        my_planets / 20.0, enemy_planets / 20.0,
    ], dtype=np.float32)


def find_target_via_ray(src_xy, send_angle: float, planets,
                        ray_horizon: float = 200.0,
                        perp_margin: float = 1.0) -> int:
    """Recover the (likely) target planet of a shot from (src, angle).

    v4 emits actions as (src_id, angle, ships) — the target is implicit.
    We project a ray from src along `angle` and return the closest planet
    whose bounding circle the ray crosses. Returns -1 if nothing in range.
    """
    sx, sy = src_xy
    fx, fy = math.cos(send_angle), math.sin(send_angle)
    best_pid, best_perp = -1, 1e9
    for p in planets:
        pid, _, px, py, pr, _, _ = p
        pid = int(pid)
        px = float(px); py = float(py); pr = float(pr)
        dx = px - sx
        dy = py - sy
        t = dx * fx + dy * fy
        if t <= 0 or t > ray_horizon:
            continue
        perp = abs(dx * fy - dy * fx)
        if perp <= pr + perp_margin and perp < best_perp:
            best_perp = perp
            best_pid = pid
    return best_pid


def label_outcome(env_steps, target_id: int, side: int,
                  arrival_turn: int, window: int = 10) -> int:
    """Label = 1 iff `side` owns `target_id` at any turn in [arrival, arrival+window]."""
    end_t = min(arrival_turn + window, len(env_steps) - 1)
    start_t = min(arrival_turn, end_t)
    for t in range(start_t, end_t + 1):
        s = env_steps[t][side].observation
        if s is None:
            continue
        for p in s["planets"]:
            if int(p[0]) == target_id and int(p[1]) == side:
                return 1
    return 0
