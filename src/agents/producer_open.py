"""``producer_open`` brain — AG20: early-game position-building layer on top of the Producer.

The [loss diagnosis](../../wiki/producer_loss_diagnosis.md) localised the *beatable* gap to
**early position-building (turns 0–90)**: the Producer is a pure single-turn greedy with no
notion of multi-turn board position, and early commitment was *identical* to ours (so the
lever is *where/what* we expand, not *how much*). This brain, inside an opening window
``0..T`` (``T≈90``), **re-ranks the Producer's OWN candidates** by adding a position-value
term the single-turn flow-diff under-weights, then hands the chosen Shots back through the
normal pipeline. Outside the window it is a **strict no-op** (byte-identical to
``producer_port``), so it can only move play where the diagnosis flagged.

Same isolation as ``producer_solver``: we patch an **isolated copy** of the runtime
(``_greedy_select`` re-ranks the score; ``run_turn`` is wrapped to stash the board so the
position term can read planet owners/production), so ``producer_port`` is never disturbed.
The term is a single **pluggable hook** so terms are A/B'd one at a time through the AG19 gate.

Tunables (env, read at import):
  PRODUCER_OPEN_T       window end turn (inclusive), default 90
  PRODUCER_OPEN_TERM    'neutral_expand' (default) | 'centrality' | 'none'
  PRODUCER_OPEN_WEIGHT  term weight in flow-diff (ship) units, default 2.0

Public API:
    plan_turn(obs, config=None) -> list[list]
"""
from __future__ import annotations

import importlib.util
import os
import sys
from typing import List

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_VENDOR = os.path.join(_REPO, "vendor")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)


def _load_isolated_runtime():
    """A PRIVATE copy of ``producer_runtime`` (its own ``_greedy_select`` / ``run_turn``
    bindings) so our monkeypatch can't leak into ``producer_port`` or ``producer_solver``."""
    path = os.path.join(_VENDOR, "producer_runtime.py")
    spec = importlib.util.spec_from_file_location("producer_runtime_open", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["producer_runtime_open"] = mod
    spec.loader.exec_module(mod)
    return mod


_pr = _load_isolated_runtime()
_ORIG_GREEDY = _pr._greedy_select
_ORIG_RUN_TURN = _pr.run_turn

_T = int(os.environ.get("PRODUCER_OPEN_T", "90"))
_TERM = os.environ.get("PRODUCER_OPEN_TERM", "neutral_expand").lower()
_WEIGHT = float(os.environ.get("PRODUCER_OPEN_WEIGHT", "2.0"))

# Per-turn board context stashed by the run_turn wrapper (single game, single thread, so it
# is valid when _greedy_select reads it the same turn). Isolated to THIS runtime copy.
_CTX: dict = {"turn": 0, "planets": None}


def _position_term(cand_tgt_slot, planets, *, ref):
    """Position-value bonus per candidate, in flow-diff (ship) units, added to the score
    inside the opening window. ``ref`` is the score tensor (for dtype/device). ``planets``
    is ``[*,7] = [id, owner, x, y, radius, ships, production]``.

    - ``neutral_expand`` — bonus for capturing a **neutral** (owner < 0) target, scaled by
      its production: claim productive territory early (a neutral is worth more *next* turn
      than the single-turn flow-diff prices it). Enemy/own targets get 0 (unchanged).
    - ``centrality`` — bonus for targets near the board centre (sun at 50,50), which dominate
      the orbit geometry and threaten more neutrals; decays with distance to centre.
    """
    import torch

    pl = planets.reshape(-1, planets.shape[-1]).to(ref.device)
    P = pl.shape[0]
    tgt = cand_tgt_slot.to(torch.long).clamp(0, max(P - 1, 0))
    owner = pl[:, 1]
    prod = pl[:, 6].to(ref.dtype)
    if _TERM == "neutral_expand":
        is_neutral = (owner[tgt] < 0).to(ref.dtype)
        return _WEIGHT * is_neutral * prod[tgt]
    if _TERM == "centrality":
        x = pl[:, 2].to(ref.dtype)
        y = pl[:, 3].to(ref.dtype)
        d = torch.sqrt((x[tgt] - 50.0) ** 2 + (y[tgt] - 50.0) ** 2).clamp(min=0.0)
        # 1 at centre → 0 at a corner (~64 units), applied to every candidate target
        # (no seat-relative ownership needed → correct for any seat).
        return _WEIGHT * (1.0 - (d / 64.0)).clamp(min=0.0)
    return torch.zeros_like(ref)


def _tracked_run_turn(obs_tensors, *, config, player_count, memory):
    _CTX["planets"] = obs_tensors.get("planets")
    try:
        _CTX["turn"] = int(obs_tensors["step"].flatten()[0].item())
    except Exception:
        _CTX["turn"] = int(_CTX.get("turn", 0)) + 1
    return _ORIG_RUN_TURN(obs_tensors, config=config, player_count=player_count, memory=memory)


def _reranked_greedy(*, P, W, device, dtype, score, cand_src, cand_send, cand_angle,
                     cand_eta, cand_active, cand_tgt_slot, cand_tgt_short, cand_is_def,
                     source_budget, target_exists, roi_threshold):
    """Inside the opening window, add the position term to the candidate scores, then run the
    UNMODIFIED greedy selection on the re-ranked scores. Outside the window → strict no-op.
    Adding a finite bonus to an invalid (-inf) candidate leaves it -inf, so invalids stay out.
    """
    if _TERM != "none" and int(_CTX.get("turn", 0)) <= _T and _CTX.get("planets") is not None:
        try:
            score = score + _position_term(cand_tgt_slot, _CTX["planets"], ref=score)
        except Exception:
            pass  # never break play — fall back to the pure score
    return _ORIG_GREEDY(
        P=P, W=W, device=device, dtype=dtype, score=score,
        cand_src=cand_src, cand_send=cand_send, cand_angle=cand_angle, cand_eta=cand_eta,
        cand_active=cand_active, cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
        cand_is_def=cand_is_def, source_budget=source_budget,
        target_exists=target_exists, roi_threshold=roi_threshold,
    )


# Install the layer (the brain IS the contribution candidate; PRODUCER_OPEN_TERM=none makes
# it a pure-Producer no-op for control checks).
_pr.run_turn = _tracked_run_turn
_pr._greedy_select = _reranked_greedy


def plan_turn(obs, config=None) -> List[list]:
    """This turn's Shots from the (isolated) Producer runtime, re-ranked toward position in
    the opening window. ``config`` accepted and ignored (the vendored entry is single-arg)."""
    return _pr.agent(obs)
