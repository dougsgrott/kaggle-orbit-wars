import math
import random


def agent(obs, config=None):
    me = obs["player"] if isinstance(obs, dict) else obs.player
    planets = obs["planets"] if isinstance(obs, dict) else obs.planets
    moves = []
    for src in planets:
        sid, owner, sx, sy, sr, ships, _ = src
        if int(owner) != int(me) or int(ships) < 5:
            continue
        if random.random() < 0.5:
            continue
        others = [p for p in planets if int(p[0]) != int(sid)]
        if not others:
            continue
        tgt = random.choice(others)
        tx, ty = float(tgt[2]), float(tgt[3])
        ang = math.atan2(ty - float(sy), tx - float(sx))
        send = random.randint(1, max(1, int(ships) - 1))
        moves.append([int(sid), float(ang), int(send)])
    return moves
