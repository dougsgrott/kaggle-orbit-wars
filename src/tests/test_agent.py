"""Pure tests for the Kaggle entry point (no kaggle_environments).

`src/agent.py` must be a thin wrapper that delegates to the DEFAULT brain and
returns a legal move list. The full-game / illegal-launch behaviour is the
brain's responsibility (covered in test_roi_greedy.py); here we only check the
wiring and the entry-point contract.
"""

from src.agent import agent
from src.agents import REGISTRY, DEFAULT


# planet row: [id, owner, x, y, radius, ships, production]   (owner -1 = neutral)
def _obs(player, planets, fleets=()):
    return {
        "player": player,
        "planets": [list(p) for p in planets],
        "fleets": [list(f) for f in fleets],
    }


def test_agent_delegates_to_default_brain():
    # A board where the DEFAULT brain has a clear move: the entry point must
    # return exactly what that brain returns.
    obs = _obs(0, [[0, 0, 10.0, 10.0, 1.0, 80, 1], [1, -1, 40.0, 10.0, 1.0, 5, 1]])
    assert agent(obs) == REGISTRY[DEFAULT](obs)


def test_agent_returns_legal_move_list():
    obs = _obs(0, [[0, 0, 10.0, 10.0, 1.0, 80, 1], [1, -1, 40.0, 10.0, 1.0, 5, 1]])
    moves = agent(obs)
    assert isinstance(moves, list)
    by_id = {int(p[0]): p for p in obs["planets"]}
    for move in moves:
        sid, angle, ships = move
        assert int(by_id[sid][1]) == obs["player"]      # launch from a planet I own
        assert 1 <= int(ships) <= int(by_id[sid][5])    # legal ship count
        assert isinstance(angle, float)


def test_agent_accepts_config_arg_and_handles_no_targets():
    # Entry-point signature parity (obs, config) and the empty-action contract.
    obs = _obs(0, [[0, 0, 10.0, 10.0, 1.0, 50, 1]])  # only my planet, nothing to attack
    assert agent(obs, config=None) == []
