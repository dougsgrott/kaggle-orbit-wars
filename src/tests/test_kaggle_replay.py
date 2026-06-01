"""Pure tests for the Kaggle replay adapter (no kaggle_environments).

Builds a tiny synthetic replay JSON in the Kaggle `steps` shape and checks the
field mapping into our EpisodeTrace + the final-board placement derivation.
"""

import json

from src.kaggle_replay import load_kaggle_trace, summarize


def _step(planets, fleets, actions, statuses, rewards):
    """One recorded step: a per-agent list of dicts. Slot 0 carries the shared
    board observation; every slot carries its own action/status/reward."""
    n = len(actions)
    return [
        {
            "observation": {"planets": planets, "fleets": fleets} if i == 0 else {},
            "action": actions[i],
            "status": statuses[i],
            "reward": rewards[i],
        }
        for i in range(n)
    ]


def _write_replay(tmp_path, steps, rewards, seed=7):
    data = {
        "configuration": {"episodeSteps": len(steps)},
        "rewards": rewards,
        "statuses": ["DONE"] * len(steps[0]),
        "info": {"seed": seed},
        "steps": steps,
    }
    p = tmp_path / "replay.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_maps_frames_and_actions(tmp_path):
    # 2-player, 2-step replay. Agent 0 owns a planet and launches a fleet on t0.
    planets0 = [[0, 0, 10.0, 10.0, 1.0, 50, 1], [1, 1, 90.0, 90.0, 1.0, 50, 1]]
    fleets0 = []
    planets1 = [[0, 0, 10.0, 10.0, 1.0, 41, 1], [1, 1, 90.0, 90.0, 1.0, 51, 1]]
    fleets1 = [[0, 0, 12.0, 12.0, 0.5, 0, 9]]
    steps = [
        _step(planets0, fleets0, [[[0, 0.5, 9]], []], ["ACTIVE", "ACTIVE"], [0, 0]),
        _step(planets1, fleets1, [[], []], ["DONE", "DONE"], [1, -1]),
    ]
    path = _write_replay(tmp_path, steps, rewards=[1, -1])

    trace = load_kaggle_trace(path)
    assert trace.num_players == 2
    assert len(trace) == 2
    assert trace.seed == 7
    # Frame fields map straight through.
    assert trace.frames[0].planets == planets0
    assert trace.frames[1].fleets == fleets1
    assert trace.frames[0].actions == [[[0, 0.5, 9]], []]


def test_placement_from_final_board(tmp_path):
    # Final board: agent 1 has more ships -> 1st; carries env rewards through.
    final_planets = [[0, 0, 10.0, 10.0, 1.0, 20, 1], [1, 1, 90.0, 90.0, 1.0, 80, 1]]
    steps = [
        _step(final_planets, [], [[], []], ["DONE", "DONE"], [-1, 1]),
        _step(final_planets, [], [[], []], ["DONE", "DONE"], [-1, 1]),
    ]
    path = _write_replay(tmp_path, steps, rewards=[-1, 1])

    trace = load_kaggle_trace(path)
    by_index = {o.index: o for o in trace.result.outcomes}
    assert by_index[1].placement == 1 and by_index[1].score == 80
    assert by_index[0].placement == 2 and by_index[0].score == 20
    assert by_index[1].reward == 1


def test_summary_reports_scores_and_death_tally(tmp_path):
    # A fleet present on t0 that is gone on t1, having crossed nothing and left
    # the board -> counts as a missed (out-of-bounds) death.
    planets = [[0, 0, 5.0, 5.0, 1.0, 30, 1], [1, 1, 95.0, 95.0, 1.0, 30, 1]]
    # Fleet heading off the bottom-left corner at speed (1-ship => speed 1.0).
    fleet_off = [[0, 0, 0.5, 0.5, 3.4, 0, 1]]
    steps = [
        _step(planets, fleet_off, [[], []], ["ACTIVE", "ACTIVE"], [0, 0]),
        _step(planets, [], [[], []], ["DONE", "DONE"], [1, -1]),
    ]
    path = _write_replay(tmp_path, steps, rewards=[1, -1])

    info = summarize(path)
    assert info["num_players"] == 2
    assert info["num_steps"] == 2
    assert info["deaths"]["out_of_bounds"] == 1
    assert info["deaths"]["missed"] == 1
    assert {o["index"] for o in info["outcomes"]} == {0, 1}
