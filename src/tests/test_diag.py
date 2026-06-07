"""Unit tests for the loss-diagnosis extractors (src/diag.py).

Pure: a tiny synthetic 2P game (we lead, then lose a planet) exercises every
Tier 1-3 extractor without kaggle_environments / torch.
"""
from src import diag


def _planet(pid, owner, x, y, ships, prod):
    return [pid, owner, x, y, 1.0, ships, prod]


def _synthetic_game():
    boards = [
        [_planet(0, 0, 10, 10, 10, 2), _planet(1, 1, 90, 90, 10, 2), _planet(2, -1, 50, 20, 3, 1)],
        [_planet(0, 0, 10, 10, 12, 2), _planet(1, 1, 90, 90, 12, 2), _planet(2, 0, 50, 20, 4, 1)],
        [_planet(0, 0, 10, 10, 8, 2),  _planet(1, 1, 90, 90, 16, 2), _planet(2, 1, 50, 20, 2, 1)],
    ]
    moves = [[[[0, 1.5, 5]], []], [[], []], [[], []]]  # we launch turn 0; opp passes
    turns = [diag.TurnLog(step=i, planets=b, fleets=[], moves=moves[i], obs={
        "player": 0, "step": i, "planets": b, "fleets": [],
        "initial_planets": boards[0], "angular_velocity": 0.0,
        "comets": [], "comet_planet_ids": []}) for i, b in enumerate(boards)]
    return diag.GameLog(seed=1, num_players=2, our_seat=0, placement=2, scores=[8, 16], turns=turns)


def test_metrics_per_player():
    g = _synthetic_game()
    m = diag.metrics_per_player(g.turns[2].planets, [], 2)
    assert list(m["ships"]) == [8.0, 18.0]
    assert list(m["planets"]) == [1.0, 2.0]
    assert list(m["production"]) == [2.0, 3.0]


def test_game_series_and_divergence():
    g = _synthetic_game()
    s = diag.game_series(g)["ships"]
    assert list(s) == [0.0, 4.0, -10.0]          # ahead, then collapse
    agg = diag.aggregate_trajectories([g], length=3)
    assert agg["divergence_point"] == 2           # bleed onset at the loss turn


def test_ownership_events():
    g = _synthetic_game()
    ev = diag.ownership_events(g)
    assert {"step": 1, "planet": 2, "frm": -1, "to": 0} in ev   # we grab neutral
    assert {"step": 2, "planet": 2, "frm": 0, "to": 1} in ev    # enemy retakes


def test_territory_rates():
    g = _synthetic_game()
    r = diag.territory_rates(g)
    assert r["our_captures"] == 1.0 and r["our_losses"] == 1.0
    assert r["opp_captures"] == 1.0


def test_fingerprint():
    g = _synthetic_game()
    fp = diag.fingerprint([g], "our")
    assert abs(fp["ships_per_launch"] - 5.0) < 1e-9
    assert 0.0 < fp["active_turn_frac"] <= 1.0


def test_counterfactual_diff():
    g = _synthetic_game()
    # an opponent that always passes: it agrees on turns where we also pass.
    passer = lambda obs, config=None: []
    d = diag.counterfactual_diff(g, passer)
    assert d["turns"] == 3.0
    assert d["we_act_they_pass"] == 1.0          # turn 0 we launched, passer didn't
    assert 0.0 <= d["agreement"] <= 1.0
