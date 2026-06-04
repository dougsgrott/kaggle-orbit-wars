"""Reference opponents — the fixed yardsticks we measure brains against.

The simple panel was lifted from clean_scripts/train_submit_v4 OPPONENT_CODES;
the **Boss** (`boss.py`) is the ported `robust_agent` (catalogue 685.1). Each
file is a standalone Kaggle-environments agent (top-level `agent(obs[, config])`)
that does **not** import the solution package, so it always runs inside the
Arena via its file path: `run_episode([brain, opponents.BOSS], ...)`.

The **Ladder** (`LADDER`) is the tiered, graded yardstick the eval harness sweeps
(see L1 / docs/issues/eval). Tiers go floor -> panel -> official -> boss -> strong
-> snapshots so signal stays non-saturating: when a brain maxes out the panel, the
Boss discriminates; when it beats the Boss, the **strong** tier (L2: higher-rated
ported agents) still does. Builtin names ("random"/"starter") are resolved by the
Official env; the rest are file paths.
"""

from pathlib import Path

OPPONENTS_DIR = Path(__file__).parent

NEAREST_SNIPER   = str(OPPONENTS_DIR / "nearest_sniper.py")
WEAKEST_FIRST    = str(OPPONENTS_DIR / "weakest_first.py")
PRODUCTION_FIRST = str(OPPONENTS_DIR / "production_first.py")
DEFENDER         = str(OPPONENTS_DIR / "defender.py")
RANDOM_PLAY      = str(OPPONENTS_DIR / "random_play.py")

# Boss tier: a strong reference (ported robust_agent, catalogue 685.1).
BOSS = str(OPPONENTS_DIR / "boss.py")

# Strong tier (L2): higher-rated ported agents *above* the Boss, so a lever has to
# beat competent play — the Boss alone is now a soft field (lookahead beats it ~42%,
# the real LB spans 800-2000). See docs/issues/eval/L2-stronger-ladder-tier.md.
LB1200 = str(OPPONENTS_DIR / "lb1200.py")   # catalogue 815.9 (heuristic WorldModel)
LB1224 = str(OPPONENTS_DIR / "lb1224.py")   # catalogue 727.5 (WorldModel + missions)
STRONG = [LB1200, LB1224]

# Official env builtins (resolved by name by kaggle_environments).
RANDOM  = "random"
STARTER = "starter"

# The simple Opponent panel (basic-mechanics regression).
ALL = [NEAREST_SNIPER, WEAKEST_FIRST, PRODUCTION_FIRST, DEFENDER, RANDOM_PLAY]

# Tiered Ladder, ordered weakest -> strongest. `snapshots` holds frozen past
# brain versions registered for self-play (by registry name); empty until a
# snapshot mechanism lands. Each entry is an arena-runnable agent (file path or
# builtin name) the eval harness can pass straight to `run_episode`.
LADDER = {
    "floor": [RANDOM],
    "panel": list(ALL),
    "official": [STARTER],
    "boss": [BOSS],
    "strong": list(STRONG),
    "snapshots": [],
}

# Flat, de-duplicated list of every Ladder opponent, weakest -> strongest.
LADDER_ALL = [opp for tier in LADDER.values() for opp in tier]
