"""``producer_solver`` brain — AG23 (Track B): exact-solver turn allocation on top
of the vendored Producer.

The champion ``producer_port`` selects fleet launches **greedily**
(``orbit_lite.planner_core._greedy_select`` — "pick the best wave each iter") over a
candidate set whose **scores are fixed** (computed once in
``producer_runtime.plan_lite_waves`` and never re-scored per pick), subject to three
coupling constraints — per-source ship capacity, one-wave-per-target, and a role
mutex. That makes the selection a **static weighted ILP** the greedy only
approximates. This brain swaps the *selector* for an exact ``scipy.optimize.milp``
solve, reusing 100% of the Producer's physics / aim / flow-diff scoring. It is the
most faithful possible "contribution on top of ``vendor/``" — see
``docs/issues/agent/AG23-solver-turn-allocation.md`` and the forum hint in
``kaggle_discussions/`` ("reframe part of the game as an optimization problem").

Isolation: we load our OWN copy of ``vendor/producer_runtime.py`` (via ``importlib``)
and monkeypatch *that copy's* ``_greedy_select``. ``producer_port`` imports the
canonical module; each runtime has its own module-level ``_greedy_select`` binding,
so our patch can never leak into ``producer_port`` even when the two are seats in one
A/B game (the same trick the ``sota`` opponent uses, see ``src/opponents/producer.py``).

Modes (env ``PRODUCER_SOLVER_MODE``):
  ``off``     — pure Producer (isolated copy; behaviour-identical to ``producer_port``).
  ``measure`` — pure Producer PLUS per-turn greedy-vs-ILP logging (behaviour-neutral;
                Stage 0 premise measurement, writes ``analysis/solver_premise_<pid>.csv``).
  ``live``    — ILP selection replaces greedy (Stage 1; gated on the premise — see issue).

Public API:
    plan_turn(obs, config=None) -> list[list]   # [from_planet_id, angle, num_ships]
"""
from __future__ import annotations

import csv
import importlib.util
import os
import sys
import threading
from typing import List, Optional, Tuple

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_VENDOR = os.path.join(_REPO, "vendor")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)


def _load_isolated_runtime():
    """Load a PRIVATE copy of ``producer_runtime`` under a distinct module name so our
    monkeypatch of ``_greedy_select`` cannot leak into ``producer_port`` (which imports
    the canonical ``producer_runtime``). The fresh module re-runs the import that binds
    ``_greedy_select`` into its own namespace — patching it rebinds only our copy."""
    path = os.path.join(_VENDOR, "producer_runtime.py")
    spec = importlib.util.spec_from_file_location("producer_runtime_solver", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["producer_runtime_solver"] = mod
    spec.loader.exec_module(mod)
    return mod


_pr = _load_isolated_runtime()
_ORIG_GREEDY = _pr._greedy_select

_MODE = os.environ.get("PRODUCER_SOLVER_MODE", "off").lower()
_ILP_TIME_LIMIT = float(os.environ.get("PRODUCER_SOLVER_TIME_LIMIT", "0.1"))  # seconds


# ---------------------------------------------------------------------------
# The exact ILP: the Producer's OWN selection objective, solved to optimum.
# ---------------------------------------------------------------------------


def _solve_ilp(
    *,
    score,
    cand_src,
    cand_send,
    cand_tgt_short,
    cand_tgt_slot,
    source_budget,
    roi_threshold: float,
    W: int,
    time_limit: float = _ILP_TIME_LIMIT,
    cap_waves: bool = True,
    tie_break: bool = True,
) -> Optional[Tuple[float, List[int], int]]:
    """Exact optimum of ``maximize Σ score[c]·x[c]`` over candidates ``c`` (``x∈{0,1}``)
    subject to the SAME constraints the greedy respects:

    - cardinality:        ``Σ x ≤ W``                     (greedy fires ≤ W waves)
    - one-wave-per-target ``Σ_{tgt_short==t} x ≤ 1``
    - per-source capacity ``Σ_{src==p} send·x ≤ budget[p]``
    - role mutex:         no planet is both a source and a target of selected waves

    Eligible candidates only: ``finite(score) & score > roi_threshold`` (exactly the
    greedy's fire gate). Returns ``(objective, chosen_global_idx, n_waves)`` with the
    objective measured on the UNPERTURBED scores, or ``None`` on solver failure/timeout
    (caller falls back to the real greedy). ``ilp_value ≥ greedy_value`` always — the
    greedy's own selection is feasible here.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp

    sc = score.detach().cpu().numpy().astype(np.float64)            # [C]
    src = cand_src[:, 0].detach().cpu().numpy().astype(np.int64)    # [C] (L==1)
    send = cand_send[:, 0].detach().cpu().numpy().astype(np.float64)
    tsh = cand_tgt_short.detach().cpu().numpy().astype(np.int64)    # [C]
    tsl = cand_tgt_slot.detach().cpu().numpy().astype(np.int64)     # [C]
    budget = source_budget.detach().cpu().numpy().astype(np.float64)  # [P]

    eligible = np.isfinite(sc) & (sc > float(roi_threshold))
    idx = np.where(eligible)[0]
    n = int(idx.shape[0])
    if n == 0:
        return (0.0, [], 0)

    s = sc[idx]
    src_e = src[idx]
    send_e = send[idx]
    tsh_e = tsh[idx]
    tsl_e = tsl[idx]

    # Objective (minimise -score). A tiny lexicographic perturbation by ascending
    # candidate index makes the chosen SET deterministic on ties (matching the
    # Producer's lowest-index `_stable_argmax`); the reported value uses raw scores.
    if tie_break:
        eps = 1e-7 * (1.0 + float(np.abs(s).max()))
        pert = eps * (n - np.arange(n)) / max(n, 1)
        c_obj = -(s + pert)
    else:
        c_obj = -s

    cons = []
    # cardinality
    if cap_waves:
        cons.append(LinearConstraint(np.ones((1, n)), 0, int(W)))
    # one wave per target
    u_t = np.unique(tsh_e)
    A_t = (tsh_e[None, :] == u_t[:, None]).astype(np.float64)
    cons.append(LinearConstraint(A_t, 0, 1))
    # per-source capacity
    u_s = np.unique(src_e)
    A_s = np.where(src_e[None, :] == u_s[:, None], send_e[None, :], 0.0)
    ub_s = budget[u_s.clip(0, budget.shape[0] - 1)]
    cons.append(LinearConstraint(A_s, 0, ub_s))
    # role mutex: planets that appear as BOTH a source and a target slot
    both = np.intersect1d(np.unique(src_e), np.unique(tsl_e))
    mutex_rows = []
    for p in both:
        i_src = np.where(src_e == p)[0]
        j_tgt = np.where(tsl_e == p)[0]
        for i in i_src:
            for j in j_tgt:
                if i == j:
                    continue
                row = np.zeros(n)
                row[i] = 1.0
                row[j] = 1.0
                mutex_rows.append(row)
        if len(mutex_rows) > 50000:  # defensive guard; never hit at these sizes
            break
    if mutex_rows:
        cons.append(LinearConstraint(np.vstack(mutex_rows), 0, 1))

    res = milp(
        c=c_obj,
        constraints=cons,
        integrality=np.ones(n),
        bounds=Bounds(0, 1),
        options={"time_limit": float(time_limit)},
    )
    if not getattr(res, "success", False) or res.x is None:
        return None
    chosen_local = np.where(np.asarray(res.x) > 0.5)[0]
    objective = float(s[chosen_local].sum())
    chosen_global = idx[chosen_local].tolist()
    return (objective, chosen_global, int(chosen_local.shape[0]))


def _greedy_value_from_entries(entries, score, cand_src, cand_tgt_slot) -> Tuple[float, int]:
    """The greedy's achieved objective = Σ score over the candidates it fired, recovered
    from the returned `LaunchEntries` by matching (source planet, target planet) → the
    unique candidate (L==1, one wave per target ⇒ keys are unique)."""
    sc = score.detach().cpu().numpy()
    src0 = cand_src[:, 0].detach().cpu().numpy().astype(np.int64)
    tsl = cand_tgt_slot.detach().cpu().numpy().astype(np.int64)
    # Map (src planet, tgt planet) -> the REAL candidate. `_candidate_indices` pads
    # non-existent shortlist slots with a repeated (clamped) index, so invalid
    # (score=-inf) candidates can collide with a real fired wave's key; ignore those
    # and keep the highest finite score per key so fired waves match correctly.
    key2c: dict = {}
    for c in range(sc.shape[0]):
        if not np.isfinite(sc[c]):
            continue
        k = (int(src0[c]), int(tsl[c]))
        if k not in key2c or sc[c] > sc[key2c[k]]:
            key2c[k] = c
    es = entries.source_slots.detach().cpu().numpy().astype(np.int64)
    et = entries.target_slots.detach().cpu().numpy().astype(np.int64)
    ev = entries.valid.detach().cpu().numpy().astype(bool)
    total = 0.0
    nw = 0
    for i in range(ev.shape[0]):
        if not ev[i]:
            continue
        c = key2c.get((int(es[i]), int(et[i])))
        if c is not None and np.isfinite(sc[c]):
            total += float(sc[c])
            nw += 1
    return total, nw


# ---------------------------------------------------------------------------
# Per-turn context (turn/game/player-count) for the premise log — captured by
# wrapping the runtime's tensor_action, the one place that sees obs["step"].
# ---------------------------------------------------------------------------

_CTX = {"game": 0, "turn": -1, "pc": 2}
_ORIG_TA = _pr.ProducerLiteRuntime.tensor_action


def _tracked_tensor_action(self, obs_tensors):
    try:
        step = int(obs_tensors["step"].flatten()[0].item())
    except Exception:
        step = _CTX["turn"] + 1
    if step == 0:
        _CTX["game"] += 1
    _CTX["turn"] = step
    out = _ORIG_TA(self, obs_tensors)
    try:
        _CTX["pc"] = int(self.memory.cached_player_count or 2)
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# Premise-measurement CSV (one file per process — A/B games run in mp workers).
# ---------------------------------------------------------------------------

_LOG_BASE = os.environ.get(
    "SOLVER_PREMISE_CSV", os.path.join(_REPO, "analysis", "solver_premise")
)
_LOG_FIELDS = [
    "pid", "game", "turn", "pc", "n_cand", "n_elig",
    "greedy_val", "greedy_w", "ilp_val", "ilp_w", "gap", "rel_gap", "feasible_ok",
]
_log_lock = threading.Lock()
_log_state = {"fh": None, "writer": None, "pid": None}


def _log_row(*, n_cand, n_elig, greedy_val, greedy_w, ilp_val, ilp_w):
    with _log_lock:
        pid = os.getpid()
        if _log_state["pid"] != pid:
            path = f"{_LOG_BASE}_{pid}.csv"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            newfile = not os.path.exists(path)
            fh = open(path, "a", newline="")
            w = csv.DictWriter(fh, fieldnames=_LOG_FIELDS)
            if newfile:
                w.writeheader()
            _log_state.update(fh=fh, writer=w, pid=pid)
        gap = ilp_val - greedy_val
        rel = gap / greedy_val if greedy_val > 1e-9 else (0.0 if gap <= 1e-9 else float("inf"))
        _log_state["writer"].writerow({
            "pid": pid, "game": _CTX["game"], "turn": _CTX["turn"], "pc": _CTX["pc"],
            "n_cand": n_cand, "n_elig": n_elig,
            "greedy_val": round(greedy_val, 6), "greedy_w": greedy_w,
            "ilp_val": round(ilp_val, 6), "ilp_w": ilp_w,
            "gap": round(gap, 6), "rel_gap": round(rel, 6),
            "feasible_ok": int(ilp_val >= greedy_val - 1e-6),
        })
        _log_state["fh"].flush()


# ---------------------------------------------------------------------------
# The two selector variants installed over the isolated runtime's _greedy_select.
# ---------------------------------------------------------------------------


def _measure_greedy(*, P, W, device, dtype, score, cand_src, cand_send, cand_angle,
                    cand_eta, cand_active, cand_tgt_slot, cand_tgt_short, cand_is_def,
                    source_budget, target_exists, roi_threshold):
    """Behaviour-neutral: play the REAL greedy unchanged, but also solve the ILP on the
    same tensors and log the greedy-vs-optimum gap (Stage 0 premise)."""
    entries, leftover = _ORIG_GREEDY(
        P=P, W=W, device=device, dtype=dtype, score=score,
        cand_src=cand_src, cand_send=cand_send, cand_angle=cand_angle, cand_eta=cand_eta,
        cand_active=cand_active, cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
        cand_is_def=cand_is_def, source_budget=source_budget,
        target_exists=target_exists, roi_threshold=roi_threshold,
    )
    try:
        import torch

        gval, gw = _greedy_value_from_entries(entries, score, cand_src, cand_tgt_slot)
        res = _solve_ilp(
            score=score, cand_src=cand_src, cand_send=cand_send,
            cand_tgt_short=cand_tgt_short, cand_tgt_slot=cand_tgt_slot,
            source_budget=source_budget, roi_threshold=float(roi_threshold), W=int(W),
        )
        if res is not None:
            ival, _chosen, iw = res
            n_elig = int((torch.isfinite(score) & (score > float(roi_threshold))).sum().item())
            _log_row(n_cand=int(score.shape[0]), n_elig=n_elig,
                     greedy_val=gval, greedy_w=gw, ilp_val=ival, ilp_w=iw)
    except Exception:
        pass  # measurement must never break play
    return entries, leftover


def _ilp_select(*, P, W, device, dtype, score, cand_src, cand_send, cand_angle,
                cand_eta, cand_active, cand_tgt_slot, cand_tgt_short, cand_is_def,
                source_budget, target_exists, roi_threshold):
    """Stage 1 (live) — ILP selection replaces the greedy. Gated on the AG23 premise;
    implemented only after Stage 0 shows real selection headroom."""
    raise NotImplementedError(
        "producer_solver live mode (_ilp_select) is Stage 1 — gated on the AG23 "
        "premise measurement (docs/issues/agent/AG23-solver-turn-allocation.md)."
    )


if _MODE == "measure":
    _pr.ProducerLiteRuntime.tensor_action = _tracked_tensor_action
    _pr._greedy_select = _measure_greedy
elif _MODE == "live":
    _pr._greedy_select = _ilp_select
# "off" (default): no patch — pure Producer, behaviour-identical to producer_port.


def plan_turn(obs, config=None) -> List[list]:
    """This turn's Shots from the (isolated) Producer runtime, with the selector chosen
    by ``PRODUCER_SOLVER_MODE``. ``config`` is accepted and ignored (the vendored entry
    is single-arg)."""
    return _pr.agent(obs)
