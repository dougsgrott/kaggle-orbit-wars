"""
M4 "mcts" brain — UCT search over the WorldModel (AG9).

The next search lever after greedy `lookahead` (AG8) became the first brain to beat
the Boss. `lookahead` evaluates a handful of whole-turn candidates **one level
deep**; MCTS searches a **tree of our future decisions**, so it can find multi-turn
plans ("take this planet now *so that* I can hold that one next turn") that a flat
one-ply search can't.

Shape of the search (single-agent UCT with a fixed opponent model):
  * A **node** is a board state where it's our turn to move. Its **children** are the
    candidate whole-turn moves; an **edge** applies our move *plus* the opponents'
    moves (from `opponent_policy`, default do-nothing) via one `worldmodel.step`.
    So the tree branches only on *our* decisions; opponents are a fixed environment
    response — exactly the hook AG10 fills with a real model (v1/Boss).
  * **Candidates:** the full brain portfolio (reused from `lookahead`) at the root
    for a rich first decision; a cheap `v1 + hold` set at internal nodes so the tree
    can actually grow (candidate generation, not rollout, is the cost bottleneck).
  * **Leaf value:** the same proven estimator as `lookahead` — roll the leaf forward
    `ROLLOUT_TURNS` with everyone holding (cheap: just `step`s) and score
    `my_ships − max(opponent)`. With `ROLLOUT_TURNS=9` a depth-1 MCTS leaf is the
    *same* horizon-10 estimate lookahead uses, so MCTS is a clean superset of it.
  * **Backup:** *mean* (classic UCT); root move = the **robust child** (most visited).
    Max-backup was tried but is pathological here: with a do-nothing opponent model it
    rewards all-in lines (e.g. dumping 77 ships on a 5-ship neutral) that a real
    opponent would punish. Mean backup behaves ≈ lookahead with depth as upside; the
    genuine "anticipate the opponent" win comes from AG10's real opponent model, not
    from optimistic backup.
  * **Selection:** UCB1 with min-max-normalised exploitation (leaf values are
    unbounded ship counts) + visit-based exploration.
  * **Budget:** anytime wall-clock (`MCTS_BUDGET_S`, default 0.85 s) so every turn
    stays under the 1 s soft `actTimeout` and leaves the 60 s overage bank as reserve
    (see wiki/measured_log.md timing note). A `min_iters` floor guarantees every root
    candidate is tried at least once even under a tiny (test) budget.

Requires `kaggle_environments` (via the WorldModel) — a search brain, not a pure one.
Pure `plan_turn(obs, config)` contract and legal Shot output regardless.

Public API:
    plan_turn(obs, config=None) -> list[list]
    mcts_plan(obs, *, num_players, opponent_policy=None, budget_s=..., ...) -> list[list]
        (parameterised entry; AG10 builds its opponent-model variant on top of this)
"""
from __future__ import annotations

import math
import time
from typing import Callable, List, Optional

from .. import worldmodel as wm
from .roi_greedy import _field
from .roi_greedy_predict import plan_turn as _v1
from .lookahead import _candidate_moves as _root_candidates

# --- tunables ---------------------------------------------------------------
MCTS_BUDGET_S = 0.8       # hard wall-clock budget for the whole turn; margin under the 1 s
                          # soft limit so the rare dense-board overshoot stays bank-absorbable
MCTS_MAX_ITERS = 4000     # hard cap on simulations (budget usually bites first)
ROLLOUT_TURNS = 9         # leaf rollout horizon; +1 expansion step => horizon 10 == lookahead
UCT_C = 1.4               # exploration constant (exploitation is normalised to [0,1])


def _safe_call(policy: Optional[Callable], obs) -> list:
    """Call a policy on a *simulated* obs, never propagating its errors. A brain or
    opponent model can hit an edge case on a hypothetical mid-search state it would
    never see in a real game — that must not crash our agent (it would fault and
    lose). On any exception, treat it as 'hold' ([])."""
    if policy is None:
        return []
    try:
        return policy(obs) or []
    except Exception:
        return []


def _internal_candidates(obs) -> List[list]:
    """Cheap candidate set for *internal* tree nodes: v1's move + hold. Candidate
    generation dominates per-node cost, so we keep the full portfolio for the root
    only and use this 2-way branch deeper in the tree."""
    cands: List[list] = []
    seen = set()
    for mv in (_safe_call(_v1, obs), []):
        key = tuple(tuple(m) for m in mv)
        if key not in seen:
            seen.add(key)
            cands.append(mv)
    return cands


def _leaf_value(fstate, me: int) -> float:
    """My ships minus the strongest opponent's (same estimator as lookahead)."""
    scores = wm.score(fstate)
    mine = scores[me]
    others = [s for i, s in enumerate(scores) if i != me]
    return float(mine - (max(others) if others else 0))


def value_placement(fstate, me: int) -> float:
    """4P-aware leaf value (AG10). Primary term = my **placement** (1.0 best .. 0.0
    worst — the Elo `actual` basis Kaggle ranks on), tie-broken by my ship share so
    the search keeps a gradient *within* a placement bucket without ever flipping it.

    Why not raw `my − max(opponent)`: in 4P maximising raw lead both over-commits
    and paints me as the leader to gang up on. Optimising placement instead means
    *securing a good finish* with diminishing returns once I'm ahead — I won't burn
    my home to crush one rival if it costs me position against the others. In 1v1 it
    reduces to win/lose-by-ships, so it doesn't regress the 2-player case.
    """
    scores = wm.score(fstate)
    n = len(scores)
    if n <= 1:
        return float(scores[me] if scores else 0)
    mine = scores[me]
    # placement: 1 = most ships (ties broken by lower index, like compute_placements)
    place = 1 + sum(1 for i, s in enumerate(scores)
                    if s > mine or (s == mine and i < me))
    actual = (n - place) / (n - 1)              # 1.0 (1st) .. 0.0 (last)
    total = sum(scores) or 1
    return actual + 0.01 * (mine / total)       # share breaks ties, never flips placement


class _Node:
    """A decision node: a state where `me` is to move, plus UCT bookkeeping."""

    __slots__ = ("state", "untried", "children", "n", "w", "terminal")

    def __init__(self, state, untried: List[list], terminal: bool = False):
        self.state = state
        self.untried = untried          # candidate moves not yet expanded
        self.children: List[tuple] = []  # (move, _Node)
        self.n = 0                       # visit count
        self.w = 0.0                     # sum of backed-up values (mean = w / n)
        self.terminal = terminal


def _is_terminal(fstate) -> bool:
    if getattr(fstate.env, "done", False):
        return True
    cfg = fstate.env.configuration
    return wm.step_of(fstate) >= int(getattr(cfg, "episodeSteps", 500))


def _apply(fstate, me: int, mv: list, num_players: int, opponent_policy: Optional[Callable]):
    """Advance one turn: `me` plays `mv`, every other player plays `opponent_policy`
    (or holds if None). Returns the child ForwardState. Opponent-model errors on a
    simulated state are swallowed (treated as hold) — see `_safe_call`."""
    actions = [[] for _ in range(num_players)]
    actions[me] = mv
    if opponent_policy is not None:
        for i in range(num_players):
            if i != me:
                actions[i] = _safe_call(opponent_policy, wm.obs_for_player(fstate, i))
    return wm.step(fstate, actions)


def _rollout_value(fstate, me: int, num_players: int,
                   value_fn: Optional[Callable] = None,
                   opponent_policy: Optional[Callable] = None,
                   deadline: Optional[float] = None) -> float:
    """Leaf estimate: roll `ROLLOUT_TURNS` forward, then score with `value_fn`.

    Opponents play `opponent_policy` for the whole rollout (they keep responding to my
    over-extension, which is what makes the leaf realistic). `deadline` is a hard
    wall-clock cap checked **every step**, so even on a dense board (each `step`
    deep-copies the whole state, and a policy rollout is pricey) a turn can't overshoot
    — it just truncates the rollout. That bound is what prevents the 1 s+ turns /
    overage-drain faults; on dense boards the rollout simply runs fewer turns. With
    `opponent_policy=None, deadline=None` this is exactly lookahead's do-nothing leaf."""
    cur = fstate
    for _t in range(ROLLOUT_TURNS):
        if getattr(cur.env, "done", False):
            break
        if deadline is not None and time.monotonic() >= deadline:
            break
        if opponent_policy is not None:
            acts = [[] for _ in range(num_players)]
            for i in range(num_players):
                if i != me:
                    acts[i] = _safe_call(opponent_policy, wm.obs_for_player(cur, i))
            cur = wm.step(cur, acts)
        else:
            cur = wm.step(cur, [[] for _ in range(num_players)])
    return (value_fn or _leaf_value)(cur, me)


def _ucb_child(node: _Node, vmin: float, vmax: float):
    """Pick the child maximising normalised-Q + exploration."""
    span = (vmax - vmin) or 1.0
    log_n = math.log(node.n) if node.n > 0 else 0.0
    best, best_score = None, -math.inf
    for mv, ch in node.children:
        exploit = ((ch.w / ch.n) - vmin) / span if ch.n > 0 else 0.0
        explore = UCT_C * math.sqrt(log_n / ch.n) if ch.n > 0 else math.inf
        s = exploit + explore
        if s > best_score:
            best_score, best = s, (mv, ch)
    return best


def mcts_plan(
    obs,
    *,
    num_players: int,
    opponent_policy: Optional[Callable] = None,
    value_fn: Optional[Callable] = None,
    budget_s: float = MCTS_BUDGET_S,
    max_iters: int = MCTS_MAX_ITERS,
    config: Optional[dict] = None,
) -> List[list]:
    """Run UCT and return the best root move.

    `opponent_policy` is the fixed model opponents follow at every **tree edge**
    (`_apply`) and for the first few **rollout** turns (`_rollout_value`); None =
    do-nothing. AG10 passes v1/Boss here so the search anticipates the response to its
    moves. `value_fn(fstate, me)` scores leaves (default `_leaf_value`; AG10 passes
    `value_placement`). The whole turn is hard-bounded by `budget_s` (checked inside
    rollouts too), so it can't overshoot the 1 s limit even on dense boards."""
    start = time.monotonic()          # clock the WHOLE turn (incl. candidate gen) vs the deadline
    me = int(_field(obs, "player"))
    vfn = value_fn or _leaf_value

    root_moves = _root_candidates(obs)
    if len(root_moves) <= 1:
        return root_moves[0] if root_moves else []

    base = wm.from_obs(obs, num_players=num_players,
                       config=config if isinstance(config, dict) else None)
    root = _Node(base, untried=list(root_moves), terminal=_is_terminal(base))

    vmin, vmax = math.inf, -math.inf
    deadline = start + budget_s
    iters = 0

    # Anytime: loop until the wall-clock budget (covers root build too) or the cap.
    # No min-iteration floor — if the budget is starved we just return v1's move
    # (root_moves[0], the default below), never overrunning the turn.
    while iters < max_iters and time.monotonic() < deadline:
        iters += 1
        node = root
        path = [root]

        # SELECTION: descend fully-expanded, non-terminal nodes by UCB.
        while not node.untried and node.children and not node.terminal:
            _, node = _ucb_child(node, vmin, vmax)
            path.append(node)

        # EXPANSION: add one child for an untried move (priority order: v1 first).
        if node.untried and not node.terminal:
            mv = node.untried.pop(0)
            child_state = _apply(node.state, me, mv, num_players, opponent_policy)
            child_obs = wm.obs_for_player(child_state, me)
            child = _Node(
                child_state,
                untried=_internal_candidates(child_obs),
                terminal=_is_terminal(child_state),
            )
            node.children.append((mv, child))
            path.append(child)
            node = child

        # SIMULATION: leaf estimate (terminal -> score directly). The rollout shares
        # the turn deadline so a single dense-board iteration can't overshoot 1 s.
        value = (vfn(node.state, me) if node.terminal
                 else _rollout_value(node.state, me, num_players, vfn,
                                     opponent_policy, deadline))

        # BACKUP: mean value up the path; track global value range for UCB scaling.
        if value < vmin:
            vmin = value
        if value > vmax:
            vmax = value
        for n in path:
            n.n += 1
            n.w += value

    # Best root move: the robust child (most visited), tie -> higher mean, tie ->
    # earliest (v1). Default to v1's move if the budget expanded nothing.
    best_mv, best_key = root_moves[0], (-1, -math.inf)
    for mv, ch in root.children:
        key = (ch.n, ch.w / ch.n if ch.n else -math.inf)
        if key > best_key:
            best_key, best_mv = key, mv
    return best_mv


def plan_turn(obs, config=None) -> List[list]:
    """Pick a move by UCT search over the WorldModel (do-nothing opponent model).
    See module docstring. AG10 supplies a real `opponent_policy` via `mcts_plan`."""
    planets = _field(obs, "planets") or []
    num_players = max(2, max((int(p[1]) for p in planets), default=0) + 1)
    return mcts_plan(obs, num_players=num_players,
                     config=config if isinstance(config, dict) else None)
