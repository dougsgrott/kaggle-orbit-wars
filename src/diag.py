"""Head-to-head loss diagnosis: *why* a brain loses to an opponent (not the rate).

Every prior Producer measurement was an aggregate win-rate; this module diagnoses
the **mechanism** of individual lost games. It captures matched head-to-head games
(per-turn board + both brains' actual launches), turns them into features at three
tiers, and provides a Tier-4 causal move-graft. See wiki/producer_loss_diagnosis.md.

  - Tier 1 (`trajectories`)   : per-turn per-player resource curves -> when & which
                                resource diverges (the "bleed onset").
  - Tier 2 (`ownership_events`, `fleet_waste`) : territory transactions + wasted fleets.
  - Tier 3 (`fingerprint`, `counterfactual_diff`) : how the two policies differ, and
                                what the opponent would do on *our* board.
  - Tier 4 (`play_grafted`)   : substitute the opponent's moves into our line over a
                                window and see if the outcome flips (causal).

The pure feature extractors take plain board lists and import only numpy, so they
unit-test without kaggle_environments / torch. Game capture (`play_logged`,
`play_grafted`) imports the env + the Producer lazily.

Board row layout (from the obs): planet = [id, owner, x, y, radius, ships, prod];
fleet = [id, owner, x, y, angle, from_planet_id, ships]. Owners are absolute player
ids (full observability), so one seat's board carries every player's state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --- per-turn record -------------------------------------------------------
@dataclass
class TurnLog:
    step: int
    planets: list            # board this turn (absolute owners)
    fleets: list
    moves: list              # per-seat move issued this turn ([[from,angle,ships],...])
    obs: object = None       # our-seat observation (for the counterfactual re-feed)


@dataclass
class GameLog:
    seed: Optional[int]
    num_players: int
    our_seat: int
    placement: int           # 1 = best
    scores: List[int]
    turns: List[TurnLog] = field(default_factory=list)

    @property
    def won(self) -> bool:
        return self.placement == 1


# ---------------------------------------------------------------------------
# Game capture (lazy env/Producer import)
# ---------------------------------------------------------------------------
def _opp_seats(num_players: int, our_seat: int) -> List[int]:
    return [s for s in range(num_players) if s != our_seat]


def _plain(x):
    """Coerce a kaggle Struct/obs into plain dict/list/scalars (picklable, and
    re-feedable to any brain via `_field`)."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    try:
        import numpy as _np
        if isinstance(x, _np.generic):
            return x.item()
    except Exception:
        pass
    return x


def _obs_snapshot(ob) -> dict:
    """Picklable plain-dict snapshot carrying every field the brains/Producer read."""
    keys = ("player", "step", "planets", "fleets", "initial_planets",
            "angular_velocity", "comets", "comet_planet_ids")
    out = {}
    for k in keys:
        v = ob[k] if (isinstance(ob, dict) and k in ob) else getattr(ob, k, None)
        out[k] = _plain(v)
    return out


def play_logged(our_fn: Callable, opp_fn: Callable, *, seed: int,
                num_players: int = 2, our_seat: int = 0) -> GameLog:
    """Play one head-to-head game, logging every turn's board + per-seat move.

    ``our_fn`` plays ``our_seat``; ``opp_fn`` fills the rest. Returns a GameLog.
    """
    from kaggle_environments import make
    from .arena import score_board, compute_placements

    env = make("orbit_wars",
               configuration=({"seed": seed} if seed is not None else {}),
               debug=False)
    env.reset(num_players)
    turns: List[TurnLog] = []
    t = 0
    while not env.done:
        obss = [s.observation for s in env.steps[-1]]
        # The raw per-seat obs from env.steps omits the shared `step` for non-zero
        # seats (env.run merges it, manual stepping doesn't). Inject it — the
        # Producer resets its rolling fleet cache when step==0, so a missing step
        # silently cripples it at any seat but 0.
        for ob_s in obss:
            try:
                ob_s["step"] = t
            except Exception:
                pass
        moves: list = []
        for seat in range(num_players):
            fn = our_fn if seat == our_seat else opp_fn
            try:
                mv = fn(obss[seat])
            except Exception:
                mv = []
            moves.append(list(mv) if mv else [])
        ob = obss[our_seat]
        snap = _obs_snapshot(ob)
        snap["step"] = t  # non-zero seats' obs omit 'step'; inject the turn index
        turns.append(TurnLog(step=t, planets=snap["planets"],
                             fleets=snap["fleets"], moves=_plain(moves), obs=snap))
        env.step(moves)
        t += 1

    fob = env.steps[-1][our_seat].observation
    planets = [list(p) for p in fob["planets"]]
    fleets = [list(f) for f in fob["fleets"]]
    scores = score_board(planets, fleets, num_players)
    placement = compute_placements(scores)[our_seat]
    return GameLog(seed=seed, num_players=num_players, our_seat=our_seat,
                   placement=placement, scores=scores, turns=turns)


def play_grafted(our_fn: Callable, opp_fn: Callable, *, seed: int,
                 num_players: int, our_seat: int, graft_steps) -> int:
    """Tier 4 causal graft: at ``our_seat``, play ``opp_fn``'s move on steps in
    ``graft_steps`` and ``our_fn`` otherwise; opponents always play ``opp_fn``.
    Returns our placement (1 = best). If grafting a window flips losses to wins,
    that window *causes* the loss."""
    from kaggle_environments import make
    from .arena import score_board, compute_placements

    gset = set(int(s) for s in graft_steps)
    env = make("orbit_wars",
               configuration=({"seed": seed} if seed is not None else {}),
               debug=False)
    env.reset(num_players)
    step = 0
    while not env.done:
        obss = [s.observation for s in env.steps[-1]]
        for ob_s in obss:                       # inject shared step (see play_logged)
            try:
                ob_s["step"] = step
            except Exception:
                pass
        moves = []
        for seat in range(num_players):
            if seat == our_seat:
                fn = opp_fn if step in gset else our_fn
            else:
                fn = opp_fn
            try:
                mv = fn(obss[seat])
            except Exception:
                mv = []
            moves.append(list(mv) if mv else [])
        env.step(moves)
        step += 1
    fob = env.steps[-1][our_seat].observation
    scores = score_board([list(p) for p in fob["planets"]],
                         [list(f) for f in fob["fleets"]], num_players)
    return compute_placements(scores)[our_seat]


# ---------------------------------------------------------------------------
# Tier 1 — per-turn per-player resource metrics + divergence
# ---------------------------------------------------------------------------
def metrics_per_player(planets: list, fleets: list, n: int) -> Dict[str, np.ndarray]:
    """Per-player [n] arrays: ships(on planets)+fleet, production, planet_count,
    fleet_ships(in transit), idle_ships(on planets)."""
    on_planet = np.zeros(n); prod = np.zeros(n); pcount = np.zeros(n)
    in_fleet = np.zeros(n)
    for p in planets:
        o = int(p[1])
        if 0 <= o < n:
            on_planet[o] += float(p[5]); prod[o] += float(p[6]); pcount[o] += 1
    for f in fleets:
        o = int(f[1])
        if 0 <= o < n:
            in_fleet[o] += float(f[6])
    return {"ships": on_planet + in_fleet, "production": prod,
            "planets": pcount, "fleet_ships": in_fleet, "idle_ships": on_planet}


def game_series(game: GameLog) -> Dict[str, np.ndarray]:
    """Per-turn differential (our − best-opponent) series for each metric. [T]."""
    n = game.num_players
    opp = _opp_seats(n, game.our_seat)
    keys = ("ships", "production", "planets", "fleet_ships", "idle_ships")
    out = {k: [] for k in keys}
    for tl in game.turns:
        m = metrics_per_player(tl.planets, tl.fleets, n)
        for k in keys:
            ours = m[k][game.our_seat]
            best_opp = max(m[k][s] for s in opp)
            out[k].append(ours - best_opp)
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def _align(series_list: List[np.ndarray], length: int) -> np.ndarray:
    """Stack [G] variable-length series into [G, length], padding short games with
    their final value (the game ended -> advantage frozen)."""
    out = np.full((len(series_list), length), np.nan)
    for i, s in enumerate(series_list):
        L = min(len(s), length)
        out[i, :L] = s[:L]
        if L < length and L > 0:
            out[i, L:] = s[L - 1]
    return out


def aggregate_trajectories(games: List[GameLog], length: int = 500) -> Dict[str, dict]:
    """Mean differential trajectory + 90% bootstrap band per metric, over a set of
    games (typically the lost ones). Plus the ships divergence point."""
    keys = ("ships", "production", "planets", "fleet_ships", "idle_ships")
    series = {k: [] for k in keys}
    for g in games:
        gs = game_series(g)
        for k in keys:
            series[k].append(gs[k])
    res = {}
    rng = np.random.default_rng(0)
    for k in keys:
        M = _align(series[k], length)                       # [G, length]
        mean = np.nanmean(M, axis=0)
        # bootstrap CI on the mean at each turn
        G = M.shape[0]
        boot = np.empty((400, length))
        for b in range(400):
            idx = rng.integers(0, G, G)
            boot[b] = np.nanmean(M[idx], axis=0)
        lo = np.nanpercentile(boot, 5, axis=0)
        hi = np.nanpercentile(boot, 95, axis=0)
        res[k] = {"mean": mean, "lo": lo, "hi": hi}
    res["divergence_point"] = _divergence_point(res["ships"]["mean"])
    return res


def _divergence_point(mean_ships: np.ndarray) -> int:
    """First turn after which the mean ships-advantage stays <= 0 (the bleed onset).
    -1 if it never goes/stays negative."""
    n = len(mean_ships)
    for t in range(n):
        if mean_ships[t] <= 0.0 and np.all(mean_ships[t:] <= 0.0):
            return t
    return -1


# ---------------------------------------------------------------------------
# Tier 2 — territory event ledger + fleet waste
# ---------------------------------------------------------------------------
def ownership_events(game: GameLog) -> List[dict]:
    """Planet ownership changes across turns: (step, planet, frm, to, by_seat)."""
    events = []
    prev = {int(p[0]): int(p[1]) for p in game.turns[0].planets} if game.turns else {}
    for tl in game.turns[1:]:
        cur = {int(p[0]): int(p[1]) for p in tl.planets}
        for pid, o in cur.items():
            po = prev.get(pid, o)
            if o != po:
                events.append({"step": tl.step, "planet": pid, "frm": po, "to": o})
        prev = cur
    return events


def territory_rates(game: GameLog) -> Dict[str, float]:
    """Per-player capture/loss counts + our expansion speed (turns to own N)."""
    n = game.num_players; me = game.our_seat
    cap = np.zeros(n); lost = np.zeros(n)
    for e in ownership_events(game):
        if 0 <= e["to"] < n:
            cap[e["to"]] += 1
        if 0 <= e["frm"] < n:
            lost[e["frm"]] += 1
    # expansion: first turn we own >= 5 / 8 planets
    def turns_to(k):
        for tl in game.turns:
            if sum(1 for p in tl.planets if int(p[1]) == me) >= k:
                return tl.step
        return -1
    opp = _opp_seats(n, me)
    return {"our_captures": float(cap[me]), "our_losses": float(lost[me]),
            "opp_captures": float(max(cap[s] for s in opp)),
            "opp_losses": float(max(lost[s] for s in opp)),
            "turns_to_5": float(turns_to(5)), "turns_to_8": float(turns_to(8))}


def fleet_waste(game: GameLog) -> Dict[str, float]:
    """Reuse replay's death-cause inference to count wasted (sun/OOB) fleets per
    launcher seat, normalised by launches issued."""
    from .replay import fleet_deaths
    from .arena import Frame
    frames = [Frame(step=tl.step, planets=tl.planets, fleets=tl.fleets, actions=tl.moves)
              for tl in game.turns]
    n = game.num_players; me = game.our_seat
    waste = np.zeros(n)
    for t in range(len(frames) - 1):
        for d in fleet_deaths(frames[t], frames[t + 1]):
            o = int(getattr(d, "owner", -1))
            if d.cause in ("out_of_bounds", "sun") and 0 <= o < n:
                waste[o] += 1
    launches = np.zeros(n)
    for tl in game.turns:
        for seat in range(n):
            launches[seat] += len(tl.moves[seat]) if seat < len(tl.moves) and tl.moves[seat] else 0
    opp = _opp_seats(n, me)
    return {"our_waste": float(waste[me]), "our_launches": float(launches[me]),
            "our_waste_rate": float(waste[me] / max(1.0, launches[me])),
            "opp_waste": float(max(waste[s] for s in opp)),
            "opp_launches": float(max(launches[s] for s in opp)),
            "opp_waste_rate": float(max(waste[s] for s in opp) / max(1.0, max(launches[s] for s in opp)))}


# ---------------------------------------------------------------------------
# Tier 3 — policy fingerprint + counterfactual diff
# ---------------------------------------------------------------------------
def _classify_move(m, planets_by_id: dict, me: int) -> str:
    """attack(enemy) / grab(neutral) / reinforce(own) by the move's source owner ...
    target is implicit (angle), so classify by *source* spend context only: we tag
    by what the launcher owns nearby is hard; use ship-count buckets instead below."""
    return "launch"


def fingerprint(games: List[GameLog], seat_of: str = "our",
                lo_step: int = 0, hi_step: Optional[int] = None) -> Dict[str, float]:
    """Aggregate behavioural stats over a brain's *actual* launches in a set of
    games, restricted to ``lo_step <= step < hi_step`` (default all turns).
    seat_of='our' -> our_seat; 'opp' -> a representative opponent seat."""
    launches = 0; ships = 0.0; turns = 0; active_turns = 0
    dists = []
    for g in games:
        n = g.num_players
        seat = g.our_seat if seat_of == "our" else _opp_seats(n, g.our_seat)[0]
        for tl in g.turns:
            if tl.step < lo_step or (hi_step is not None and tl.step >= hi_step):
                continue
            turns += 1
            mv = tl.moves[seat] if seat < len(tl.moves) else []
            if mv:
                active_turns += 1
            launches += len(mv)
            pos = {int(p[0]): (float(p[2]), float(p[3])) for p in tl.planets}
            for shot in mv:
                ships += float(shot[2])
                src = pos.get(int(shot[0]))
                if src is not None:
                    # nearest planet distance as a coarse launch-distance proxy
                    ds = [math.hypot(src[0]-x, src[1]-y)
                          for pid,(x,y) in pos.items() if pid != int(shot[0])]
                    if ds:
                        dists.append(min(ds))
    return {"launches_per_turn": launches / max(1, turns),
            "ships_per_turn": ships / max(1, turns),
            "ships_per_launch": ships / max(1, launches),
            "active_turn_frac": active_turns / max(1, turns),
            "median_launch_dist": float(np.median(dists)) if dists else float("nan")}


def _move_key(shot) -> Tuple[int, int]:
    """A coarse, comparable key for a launch: (source planet, rounded angle bucket)."""
    return (int(shot[0]), int(round(float(shot[1]) / (math.pi / 8))))


def counterfactual_diff(game: GameLog, opp_fn: Callable,
                        up_to_step: Optional[int] = None) -> Dict[str, float]:
    """Re-feed each turn's *our* observation to ``opp_fn`` and diff its move-set
    against what we actually played. Returns agreement (Jaccard) + disagreement
    typology over turns with step < up_to_step (default all)."""
    agree = 0; total = 0
    we_pass_they_act = 0; we_act_they_pass = 0
    diff_target = 0; jac_sum = 0.0
    n = game.num_players
    for tl in game.turns:
        if up_to_step is not None and tl.step >= up_to_step:
            break
        if tl.obs is None:
            continue
        ours = tl.moves[game.our_seat] if game.our_seat < len(tl.moves) else []
        try:
            theirs = opp_fn(tl.obs) or []
        except Exception:
            theirs = []
        ok = {_move_key(s) for s in ours}
        tk = {_move_key(s) for s in theirs}
        total += 1
        if not ours and not theirs:
            agree += 1; jac_sum += 1.0; continue
        if not ours and theirs:
            we_pass_they_act += 1
        elif ours and not theirs:
            we_act_they_pass += 1
        elif ok != tk:
            diff_target += 1
        inter = len(ok & tk); union = len(ok | tk)
        jac = inter / union if union else 1.0
        jac_sum += jac
        if ok == tk:
            agree += 1
    return {"turns": float(total), "agreement": agree / max(1, total),
            "mean_jaccard": jac_sum / max(1, total),
            "we_pass_they_act": float(we_pass_they_act),
            "we_act_they_pass": float(we_act_they_pass),
            "diff_target": float(diff_target)}
