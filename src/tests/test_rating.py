"""Pure tests for the rating & aggregation module (E1) — no kaggle_environments."""

import math

from src.rating import (
    aggregate_outcomes,
    aggregate_placements,
    expected_score,
    placement_score,
    update_elo,
)


class _Result:
    """Minimal stand-in for arena.EpisodeResult: enough for aggregation."""

    def __init__(self, placements):
        self._placements = placements
        self.num_players = len(placements)

    def placement_of(self, i):
        return self._placements[i]


# --- placement_score ----------------------------------------------------------


def test_placement_score_endpoints_and_midpoints():
    assert placement_score(1, 2) == 1.0   # 1v1 win
    assert placement_score(2, 2) == 0.0   # 1v1 loss
    assert placement_score(1, 4) == 1.0   # 4P first
    assert placement_score(4, 4) == 0.0   # 4P last
    assert placement_score(2, 4) == 2 / 3
    assert placement_score(3, 4) == 1 / 3


def test_placement_score_degenerate_field():
    assert placement_score(1, 1) == 1.0


# --- aggregate_placements -----------------------------------------------------


def test_aggregate_counts_firsts_and_means():
    # Agent 0 places 1,1,2 across three 1v1s.
    results = [_Result([1, 2]), _Result([1, 2]), _Result([2, 1])]
    agg = aggregate_placements(results, 0)
    assert agg["n"] == 3
    assert agg["firsts"] == 2
    assert math.isclose(agg["first_rate"], 2 / 3)
    assert math.isclose(agg["mean_placement"], (1 + 1 + 2) / 3)
    # scores: 1.0, 1.0, 0.0 -> mean 2/3
    assert math.isclose(agg["mean_score"], 2 / 3)
    assert 0.0 <= agg["ci_lo"] <= agg["first_rate"] <= agg["ci_hi"] <= 1.0


def test_aggregate_4p_uses_normalised_score():
    # Agent 0 finishes 1st, then 3rd, in 4P games.
    results = [_Result([1, 2, 3, 4]), _Result([3, 1, 2, 4])]
    agg = aggregate_placements(results, 0)
    assert agg["firsts"] == 1
    assert math.isclose(agg["mean_placement"], 2.0)
    # scores: 1.0 and (4-3)/3 = 1/3 -> mean 2/3
    assert math.isclose(agg["mean_score"], (1.0 + 1 / 3) / 2)


def test_aggregate_empty_is_zeroed():
    agg = aggregate_placements([], 0)
    assert agg["n"] == 0 and agg["firsts"] == 0
    assert agg["first_rate"] == 0.0 and agg["mean_placement"] == 0.0


def test_aggregate_outcomes_primitive_matches_results():
    # The (placement, num_players) primitive must agree with the result-based API.
    results = [_Result([1, 2, 3, 4]), _Result([3, 1, 2, 4])]
    via_results = aggregate_placements(results, 0)
    via_tuples = aggregate_outcomes([(1, 4), (3, 4)])
    assert via_tuples == via_results


# --- Elo ----------------------------------------------------------------------


def test_expected_score_symmetry_and_direction():
    assert math.isclose(expected_score(1000, 1000), 0.5)
    assert expected_score(1200, 1000) > 0.5   # higher-rated favoured
    assert expected_score(1000, 1200) < 0.5
    # The two perspectives sum to 1.
    assert math.isclose(
        expected_score(1300, 900) + expected_score(900, 1300), 1.0
    )


def test_update_elo_moves_in_right_direction_and_magnitude():
    # Even match (E=0.5). A win nudges up by k/2; a loss down by k/2.
    r = update_elo(1000, expected=0.5, actual=1.0, k=24)
    assert math.isclose(r, 1012.0)
    r = update_elo(1000, expected=0.5, actual=0.0, k=24)
    assert math.isclose(r, 988.0)
    # Beating expectation against a much weaker opponent barely moves it.
    small = update_elo(1000, expected=0.99, actual=1.0, k=24)
    assert 1000 < small < 1000.3


def test_update_elo_zero_sum_on_even_match():
    # In an even 1v1, winner's gain equals loser's loss.
    e = expected_score(1000, 1000)
    winner = update_elo(1000, e, 1.0)
    loser = update_elo(1000, e, 0.0)
    assert math.isclose((winner - 1000) + (loser - 1000), 0.0)
