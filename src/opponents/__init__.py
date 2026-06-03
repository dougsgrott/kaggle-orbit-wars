"""Reference opponents — the fixed yardsticks we measure brains against.

The simple panel was lifted from clean_scripts/train_submit_v4 OPPONENT_CODES;
the **Boss** (`boss.py`) is the ported `robust_agent` (catalogue 685.1). Each
file is a standalone Kaggle-environments agent (top-level `agent(obs[, config])`)
that does **not** import the solution package, so it always runs inside the
Arena via its file path: `run_episode([brain, opponents.BOSS], ...)`.

The **Ladder** (`LADDER`) is the tiered, graded yardstick the eval harness sweeps
(see L1 / docs/issues/eval). Tiers go floor -> panel -> official -> boss ->
snapshots so signal stays non-saturating: when a brain maxes out the panel, the
Boss tier still discriminates. Builtin names ("random"/"starter") are resolved by
the Official env; the rest are file paths.
"""

from pathlib import Path

OPPONENTS_DIR = Path(__file__).parent

NEAREST_SNIPER   = str(OPPONENTS_DIR / "nearest_sniper.py")
WEAKEST_FIRST    = str(OPPONENTS_DIR / "weakest_first.py")
PRODUCTION_FIRST = str(OPPONENTS_DIR / "production_first.py")
DEFENDER         = str(OPPONENTS_DIR / "defender.py")
RANDOM_PLAY      = str(OPPONENTS_DIR / "random_play.py")

# Boss tier: the strongest reference (ported robust_agent, catalogue 685.1) — a
# realistic leaderboard stand-in at the top of the Ladder.
BOSS = str(OPPONENTS_DIR / "boss.py")

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
    "snapshots": [],
}

# Flat, de-duplicated list of every Ladder opponent, weakest -> strongest.
LADDER_ALL = [opp for tier in LADDER.values() for opp in tier]
