"""Our candidate agent brains — distinct strategies to experiment with and compare.

This project is experimental: a single strategy may turn out weak, so brains are
kept side by side rather than mutated in place. Each module here exposes the same
pure contract — `plan_turn(obs, config=None) -> list[Shot]` (a Shot is
`[from_planet_id, angle, num_ships]`) — built on `src.utils`. That uniform
interface lets the eval harness sweep `REGISTRY` to rank every brain against each
other and the ladder, and lets the single Kaggle entry point (`src/agent.py`)
wrap whichever brain is current.

This is the counterpart to `src/opponents/`: opponents are *fixed yardsticks*
that never import the solution package; the brains here are *ours* and reuse it.

To add a strategy: drop a module exposing `plan_turn`, then register it below.
"""
from __future__ import annotations

from typing import Callable, Dict

from .roi_greedy import plan_turn as roi_greedy
from .roi_greedy_predict import plan_turn as roi_greedy_predict
from .missions import plan_turn as missions
from .roi_defense import plan_turn as roi_defense
from .roi_ledger import plan_turn as roi_ledger
from .lookahead import plan_turn as lookahead
from .mcts import plan_turn as mcts
from .mcts_om import plan_turn as mcts_om
from .producer_lite import plan_turn as roi_projected
from .flow_value import plan_turn as flow_value
from .flow_value_ia import plan_turn as flow_value_ia
from .flow_value_def import plan_turn as flow_value_def
from .flow_value_dr import plan_turn as flow_value_dr
from .flow_value_cfg import plan_turn as flow_value_cfg  # AG16: Producer per-format config
# Ablation brains (experiment-only; never DEFAULT/submitted) — single steps from
# flow_value_def toward the Producer's tuned config, for the gap attribution in
# wiki/producer_diff.md.
from .flow_value_abl import (
    fv_abl_h, fv_abl_wide, fv_abl_thresh, fv_abl_minship,
    fv_abl_aim, fv_abl_regroup, fv_abl_format, fv_abl_all,
)

# name -> plan_turn callable. The Kaggle entry point and the eval sweep both
# select brains by these names.
REGISTRY: Dict[str, Callable] = {
    "roi_greedy": roi_greedy,
    "roi_greedy_predict": roi_greedy_predict,
    "missions": missions,
    "roi_defense": roi_defense,
    "roi_ledger": roi_ledger,
    "lookahead": lookahead,
    "mcts": mcts,
    "mcts_om": mcts_om,
    "roi_projected": roi_projected,
    "flow_value": flow_value,
    "flow_value_ia": flow_value_ia,
    "flow_value_def": flow_value_def,
    "flow_value_dr": flow_value_dr,
    "flow_value_cfg": flow_value_cfg,
    # experiment-only ablations (see wiki/producer_diff.md) — not for promotion.
    "fv_abl_h": fv_abl_h,
    "fv_abl_wide": fv_abl_wide,
    "fv_abl_thresh": fv_abl_thresh,
    "fv_abl_minship": fv_abl_minship,
    "fv_abl_aim": fv_abl_aim,
    "fv_abl_regroup": fv_abl_regroup,
    "fv_abl_format": fv_abl_format,
    "fv_abl_all": fv_abl_all,
}

# The brain `src/agent.py` submits unless told otherwise (our current best).
# Point this at a new entry once a better brain wins the comparison.
# Promoted to roi_greedy_predict after AG4: motion-aware aim beat v0 on every
# ladder opponent with non-overlapping Wilson CIs (overall 99.1% vs 78.6%;
# vs `starter` 93.8% vs 18.8%). See wiki/measured_log.md.
# Promoted to lookahead after AG8/M3b: greedy K-turn lookahead over the
# WorldModel beat roi_greedy_predict vs the Boss in a paired n=60 A/B — boss 1st
# 41.7% vs 20.0%, paired sign-test p=0.002 (the pre-registered promotion bar).
# Promoted to flow_value after AG13: the Producer's competitive flow-diff value
# (Δnet_me − Σ_opp Δnet_opp over a do-nothing projection) crushed lookahead — 1v1
# vs Boss 100% vs 25% (n=24, sign_p=0.000) AND 4P mean-place 1.12 vs 2.75 (n=8).
# Submitted as sub_04 -> LB 956.1 (new best). (AG14 intercept aim: honest negative.)
# Promoted to flow_value_def after AG15: flow_value + friendly-flip proactive defense
# (owned planets the projection shows flipping become value-scored targets). Beat
# flow_value 1v1 17-3 (n=20, sign_p=0.003) and 4P 1.80 vs 2.00; took the first games
# off the `sota` Producer (1-5, was 0-6). (Regroup was an honest negative — dropped.)
# Promoted to flow_value_cfg after AG16: the Producer's *tuned per-format config* (the
# gap-attribution found our knobs were super-additive — flat one at a time, big as a
# whole; see wiki/producer_diff.md). 2P H=18/4P H=13 + Producer shortlist widths,
# threshold 1.5, min_ships 4, regroup on, intercept aim. No regression on any tier
# (boss/strong ≥ def, mostly better); paired vs LB1200 cfg 100% vs def 85% (n=40,
# 6-0, sign_p=0.031); vs the `sota` Producer 1v1 4.2%->12.5%, 4P mean-place 2.50->2.08
# (n=12); bank-safe (90 ms late board). Ships via the zip-bootstrap bundle
# (tools/build_submission_bundle.py --brain flow_value_cfg); see submissions/submission_06/.
# See wiki/measured_log.md.
DEFAULT = "flow_value_cfg"
