"""Simple opponent agents lifted from
clean_scripts/train_submit_v4_ml_validator_topk2_tutorial.py OPPONENT_CODES.

Each file is a standalone Kaggle-environments agent. Use the file path with
env.run([my_agent_path, opponents/nearest_sniper.py])."""

from pathlib import Path

OPPONENTS_DIR = Path(__file__).parent

NEAREST_SNIPER   = str(OPPONENTS_DIR / "nearest_sniper.py")
WEAKEST_FIRST    = str(OPPONENTS_DIR / "weakest_first.py")
PRODUCTION_FIRST = str(OPPONENTS_DIR / "production_first.py")
DEFENDER         = str(OPPONENTS_DIR / "defender.py")
RANDOM_PLAY      = str(OPPONENTS_DIR / "random_play.py")

ALL = [NEAREST_SNIPER, WEAKEST_FIRST, PRODUCTION_FIRST, DEFENDER, RANDOM_PLAY]
