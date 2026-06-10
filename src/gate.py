"""AG19 — Contribution A/B gate: the high-power local arbiter for genuine contributions
on top of the vendored Producer.

The LB is too noisy to promote a contribution — two engine-identical submissions read 55
points apart and a single build drifts ±50–100/day (see wiki/measured_log.md). So a
challenger brain promotes `DEFAULT` only by beating the standing champion (`producer_port`)
in a **high-power local paired A/B**, never on the LB.

Design — **paired common-opponent vs the `sota` Producer** (engine-identical to the champion,
seated as a *file* so the env isolates each seat; this is the proven AG16–AG18 method, and it
sidesteps the in-process runtime-sharing that biases seating one brain in many seats):

- **1v1:** `ab_compare(challenger, producer_port, opponent=sota)` — both brains play the SAME
  (seed, side) boards vs the Producer; we get each one's win-rate + Wilson CI and the paired
  discordant split + **sign-test p**. The `producer_port`-vs-`sota` arm IS the self-A/B sanity
  (engine-identical mirror → ~50%, the LB-noise-floor calibration).
- **4P:** the challenger and the champion each play a **field of three `sota`** on the same
  boards; we compare placements (paired) + each one's first-rate vs the 0.25 null.
- **power/MDE:** reported on the 1v1 discordant pairs, so an underpowered result is flagged
  **inconclusive** rather than read as "flat."

PRE-REGISTERED PROMOTION BAR (AG20–AG22 reference this one rule):
  A challenger promotes `DEFAULT` only on a paired win over `producer_port` that is
  (a) SIGNIFICANT — the 1v1 paired sign-test p < 0.05 in the challenger's favour at the agreed
      n (the power line must show the run is adequately powered; an underpowered run is
      INCONCLUSIVE, never a pass and never a "flat" negative), with the 4P field corroborating
      (challenger ≥ champion, no regression); and
  (b) BANK-SAFE — max per-turn time under the 1 s soft cap.
  The LB is only a coarse post-hoc sanity check, never the arbiter.

  python -m src.gate <challenger_brain> [n_1v1_seeds] [n_4p_seeds] [workers]
"""
from __future__ import annotations

import math
import multiprocessing as mp
import os
import time
from typing import Optional

from .ladder import _SEED_BASE, _play_job, _sign_test_p, ab_compare
from .opponents import PRODUCER  # the `sota` Producer, as an env-isolated file
from .rating import aggregate_outcomes


def _min_sig_k(d: int) -> Optional[int]:
    """Smallest favourable discordant count k (of d) that is significant at two-sided p<0.05
    under the sign test — the split the design can resolve at this many discordant pairs.
    None if no split is significant (d too small to ever reach p<0.05)."""
    for k in range((d // 2) + 1, d + 1):
        if _sign_test_p(k, d) < 0.05:
            return k
    return None


def _paired_4p(challenger: str, control: str, opponent, n_seeds: int, workers: int):
    """Challenger and control each play a field of three `opponent` on the SAME boards; return
    each one's placement aggregate + the paired first-place split. Reuses `ladder._play_job`
    (one in-process brain per game; the three field seats are the env-isolated `opponent`)."""
    seeds = [_SEED_BASE + 1000 + s for s in range(n_seeds)]
    jobs, keys = [], []
    for s in seeds:
        seat = s % 4
        for tag, brain in (("a", challenger), ("b", control)):
            keys.append((s, tag))
            jobs.append((("brain", brain), ("agent", opponent), "sota", "sota", "4p", s, seat, 4))
    res = {}
    with mp.Pool(workers) as pool:
        for key, r in zip(keys, pool.imap(_play_job, jobs)):
            res[key] = r  # (opp, tier, fmt, placement, num_players)
    a_out, b_out = [], []
    a_only = b_only = both = neither = faults = 0
    for s in seeds:
        pa, pb = res[(s, "a")][3], res[(s, "b")][3]
        if pa is not None:
            a_out.append((pa, 4))
        if pb is not None:
            b_out.append((pb, 4))
        if pa is None or pb is None:
            faults += 1
            continue
        aw, bw = (pa == 1), (pb == 1)
        a_only += aw and not bw
        b_only += bw and not aw
        both += aw and bw
        neither += not aw and not bw
    return {"a": aggregate_outcomes(a_out), "b": aggregate_outcomes(b_out),
            "paired": {"a_only": a_only, "b_only": b_only, "both": both, "neither": neither,
                       "discordant": a_only + b_only,
                       "sign_p": _sign_test_p(a_only, a_only + b_only)},
            "faults": faults}


def contribution_gate(
    challenger: str,
    control: str = "producer_port",
    opponent=PRODUCER,
    n_1v1_seeds: int = 64,
    n_4p_seeds: int = 32,
    workers: int = 8,
    csv_prefix: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Paired A/B of ``challenger`` vs ``control`` (the champion), both fighting the ``sota``
    Producer, in 1v1 (both sides) + a 4P field. Returns a summary with the win-rates, paired
    sign-test, power/MDE, and a verdict against the module's pre-registered bar."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU-only → fork-safe
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    opp_name = "sota"
    if verbose:
        print(f"\n=== CONTRIBUTION GATE — '{challenger}' vs champion '{control}' "
              f"(common opponent: {opp_name}) ===")

    s1 = ab_compare(challenger, control, opponent, n_seeds=n_1v1_seeds, n_workers=workers,
                    csv_path=(f"{csv_prefix}_1v1.csv" if csv_prefix else None), verbose=False)
    s4 = _paired_4p(challenger, control, opponent, n_4p_seeds, workers)

    p1 = s1["paired"]
    d1, a1, b1, sign1 = p1["discordant"], p1["a_only"], p1["b_only"], p1["sign_p"]
    k_sig = _min_sig_k(d1)
    a_rate, b_rate = s1["a"]["first_rate"], s1["b"]["first_rate"]
    # Significant challenger win: paired sign p<0.05 AND challenger won more of the discordant.
    significant = sign1 < 0.05 and a1 > b1
    regression = sign1 < 0.05 and b1 > a1
    underpowered = (not significant and not regression) and (k_sig is None or a1 < k_sig)
    p4 = s4["paired"]
    win4 = s4["a"]["first_rate"] >= s4["b"]["first_rate"]  # challenger not regressing in 4P
    verdict = ("PROMOTE-candidate (significant 1v1 win — confirm 4P ≥ champ + bank)"
               if significant and win4
               else "SIGNIFICANT 1v1 WIN but 4P regresses — investigate" if significant
               else "SIGNIFICANT REGRESSION" if regression
               else "INCONCLUSIVE (underpowered — more discordant pairs needed)" if underpowered
               else "FLAT (no edge at this power)")

    summary = {"challenger": challenger, "control": control, "opponent": opp_name,
               "v1": s1, "v4": s4, "v1_min_sig_k": k_sig,
               "significant": significant, "regression": regression,
               "underpowered": underpowered, "verdict": verdict}

    if verbose:
        print(f"  1v1 vs {opp_name}:  challenger {a_rate*100:5.1f}% "
              f"[{s1['a']['ci_lo']*100:4.1f}–{s1['a']['ci_hi']*100:4.1f}]  "
              f"champion {b_rate*100:5.1f}% [{s1['b']['ci_lo']*100:4.1f}–{s1['b']['ci_hi']*100:4.1f}]  "
              f"(n={s1['a']['n']}/{s1['b']['n']})")
        print(f"      paired: a_only={a1} b_only={b1} both={p1['both']} neither={p1['neither']} "
              f"| discordant={d1}  sign_p={sign1:.3f}")
        print(f"      power: min favourable split for sig at d={d1} is "
              f"{(str(k_sig)+'/'+str(d1)) if k_sig else 'NONE (d too small)'} "
              f"— {'UNDERPOWERED' if underpowered else 'powered'}")
        print(f"  4P field (3×{opp_name}): challenger 1st {s4['a']['first_rate']*100:5.1f}% "
              f"meanPlace {s4['a']['mean_placement']:.2f}  |  champion 1st "
              f"{s4['b']['first_rate']*100:5.1f}% meanPlace {s4['b']['mean_placement']:.2f} "
              f"(n={s4['a']['n']}/{s4['b']['n']}; sign_p={p4['sign_p']:.3f})")
        print(f"  faults: 1v1={s1['faults']} 4P={s4['faults']}")
        print(f"  VERDICT: {verdict}")
    return summary


def bank_probe(brain: str, n_seeds: int = 3, num_players: int = 2,
               episode_steps: int = 200, verbose: bool = True) -> dict:
    """Bank-safety probe: time every ``plan_turn`` call of ``brain`` over a few 2P games vs
    the `sota` Producer (env-isolated file → no runtime sharing) and report max/mean ms per
    turn vs the 1 s soft cap."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from .agents import REGISTRY
    from .arena import EpisodeConfig, run_episode

    fn = REGISTRY[brain]
    times: list = []

    def _timed(obs, config=None):
        t = time.perf_counter()
        out = fn(obs, config)
        times.append((time.perf_counter() - t) * 1000.0)
        return out

    for s in range(n_seeds):
        line = [_timed] + [PRODUCER] * (num_players - 1)
        run_episode(line, EpisodeConfig(num_players=num_players, seed=_SEED_BASE + 2000 + s,
                                        episode_steps=episode_steps))
    max_ms = max(times) if times else 0.0
    mean_ms = sum(times) / len(times) if times else 0.0
    safe = max_ms < 1000.0
    if verbose:
        print(f"  bank: {brain} max {max_ms:.0f} ms / mean {mean_ms:.0f} ms over "
              f"{len(times)} turns — {'BANK-SAFE' if safe else 'OVER 1s CAP'}")
    return {"max_ms": max_ms, "mean_ms": mean_ms, "n_turns": len(times), "bank_safe": safe}


def _main() -> None:
    import sys

    if len(sys.argv) < 2:
        print(__doc__)
        return
    challenger = sys.argv[1]
    n1 = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    n4 = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    contribution_gate(challenger, n_1v1_seeds=n1, n_4p_seeds=n4, workers=workers,
                      csv_prefix=challenger + "_gate")
    bank_probe(challenger)


if __name__ == "__main__":
    _main()
