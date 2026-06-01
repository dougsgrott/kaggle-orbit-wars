# src/

Our Orbit Wars entry. The corpus analysis lives upstream in [../wiki/inventory.md](../wiki/inventory.md), [../wiki/SKILL.md](../wiki/SKILL.md), and [../wiki/strategies.md](../wiki/strategies.md) — read those first.

## Layout

| File | What's inside | Lifted from |
|---|---|---|
| [utils.py](utils.py) | Definitive physics / geometry / aim solver / combat. All 5 inter-script divergences documented inline as `# Variants:` comments. | clean_scripts/orbit_wars_physics_helper_module.py (canonical, v7) + complete_game_mechanics + lb_1200 |
| [features.py](features.py) | `encode_shot` (24-dim), `find_target_via_ray`, `label_outcome` for ML shot validation. | clean_scripts/train_submit_v4_ml_validator_topk2_tutorial.py (LB 899.7) |
| [arena.py](arena.py) | `run_episode(agents, config)` → `EpisodeResult` (per-agent **Placement** + final score); `record_episode` → `EpisodeTrace` (full per-turn record for the replay viewer). Player-count-agnostic (2P + 4P), seeded, fault-tolerant. The stable swap-seam over the Official env ([ADR-0001](../docs/adr/0001-one-engine-transcribed-from-official-source.md)) — the **only** module that imports `kaggle_environments`. | (new — slices A1–A2, T1) |
| [eval.py](eval.py) | `run_match`, `wilson_ci`, `evaluate_agent` — multi-opponent win-rate harness with CIs and CSV logging. *(Still imports `kaggle_environments` directly; migrates onto `arena.run_episode` in slice E2.)* | clean_scripts/orbit_wars_validation_ml_robuste_harnais_eval.py (LB 835.6) |
| [stats.py](stats.py) | `PhysicsStats` per-game instrumentation. | clean_scripts/orbit_wars_physics_helper_module.py |
| [viz.py](viz.py) | Dark-theme matplotlib helpers + `draw_board(obs)`. | clean_scripts/orbit_wars_advanced_agent_target_1608_6.py |
| [replay.py](replay.py) | Game-trace debugger: turn-by-turn replay of an `arena.record_episode` trace with fleet death-cause (combat/out-of-bounds/sun) + missed-**Shot** overlays. `python -m src.replay` → GIF + death summary. | (new — slice T1) |
| [agents/](agents/) | Our candidate **brains** — coexisting decision policies, each a pure `plan_turn(obs, config=None) → list[Shot]`, selected via a `REGISTRY` + `DEFAULT` ([ADR-0002](../docs/adr/0002-coexisting-agent-brains-registry.md)). First brain: `roi_greedy` (the v0 baseline). | (new — slice AG1) |
| [agent.py](agent.py) | Single Kaggle entry point — a thin wrapper that forwards to the `DEFAULT` brain from `agents/` (AG2). | (none — write our own) |
| [opponents/](opponents/) | 5 simple agents (`nearest_sniper`, `weakest_first`, `production_first`, `defender`, `random_play`) for eval. Fixed yardsticks — the counterpart to our `agents/` brains. | clean_scripts/train_submit_v4 OPPONENT_CODES |
| [tests/](tests/) | pytest tests for utils, arena, replay, and the brains. Pure tests need no `kaggle_environments`; the Arena integration tests skip without it. | (new) |

## Quick start

```bash
# From repo root:
cd /home/user/projects/orbit_wars

# 1. Sanity-check utils (no kaggle_environments needed)
python -m pytest src/tests/ -v

# 2. Play one game through the Arena — the only seam over kaggle_environments
python -m src.arena            # prints placements for one demo game
# ...or call the API directly:
python -c "
from src.arena import run_episode, EpisodeConfig
from src import opponents
r = run_episode([opponents.NEAREST_SNIPER, opponents.WEAKEST_FIRST],
                EpisodeConfig(num_players=2, seed=2026))
for o in sorted(r.outcomes, key=lambda o: o.placement):
    print(f'  {o.placement}. agent {o.index}: score={o.score}')
"

# 3. Run a tiny eval of our agent (n_seeds=2 → 4 games per opponent),
#    appending to the build ledger eval_log.csv
python -c "
from src import opponents
from src.eval import evaluate_agent
evaluate_agent('src/agent.py',
               ['random', 'starter', *opponents.ALL],   # official floor + baseline + panel
               n_seeds=2, n_workers=4,
               build_id='dev', csv_path='eval_log.csv')
"

# 4. Visualize a game — writes analysis/replay_demo/episode.gif + a death summary
python -m src.replay
```

## Measured baseline

`eval_log.csv` is the build ledger — one row per opponent per build (Wilson 95% CI), appended by
`evaluate_agent(..., csv_path='eval_log.csv')`. Compare CIs across builds, not point estimates.

First measured build, `v0-roi_greedy` (8 seeds × 2 sides): clears the official `random` floor
(87.5%) and beats every simple panel opponent (81–94%), but **loses to the official `starter`
(18.8%)** — the gap the M1 mission system targets. The full placement-based, both-formats Ladder
sweep (E2/E1/L1) is still owed; this baseline used the 1v1 win-rate harness against the panel +
official builtins.

## How to evolve the agent

1. Read [../wiki/strategies.md](../wiki/strategies.md) for the recommended start path.
2. Add or improve a **brain** in [agents/](agents/): a module exposing `plan_turn(obs, config=None) → list[Shot]`, registered in [agents/__init__.py](agents/__init__.py). Brains coexist so a weak strategy is one row in a comparison, not a dead end ([ADR-0002](../docs/adr/0002-coexisting-agent-brains-registry.md)); point `DEFAULT` at the current best. All physics / aim calls go through [utils.py](utils.py); never re-derive `fleet_speed` or `aim_with_prediction` in brain code. The thin [agent.py](agent.py) entry point wraps `DEFAULT` (wiring is AG2).
3. Each iteration, run `evaluate_agent(...)` against `opponents.ALL` and append to `eval_log.csv` (passed via `csv_path=...`). Compare Wilson CIs across builds, not point estimates.
4. When you start training the ML shot validator, mirror [features.py](features.py) and the gotchas in [../wiki/SKILL.md](../wiki/SKILL.md#3-known-gotchas--data-leakage-risks): game-level splits, `pos_rate` in 50–75%, 3-seed ensemble, AUC for best-epoch selection.

## Conventions

- `utils.py` is read-mostly. Add new physics helpers there sparingly and document divergence from upstream sources inline.
- Opponent agents are Kaggle-environments compatible `.py` files (top-level `def agent(obs, config=None)`). Don't import from `src` inside them — they need to be self-contained for `env.run()`.
- Tests must run without `kaggle_environments` installed (game-running tests live in a separate smoke script).
