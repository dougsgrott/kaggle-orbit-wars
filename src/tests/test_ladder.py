"""Wiring tests for the ladder harness (E2).

The aggregation, job-building, seat-rotation and ledger logic are exercised
without kaggle_environments by stubbing `run_episode`. (The real env path is
covered by the L1 smoke + the live sweep.)
"""

import csv

import src.ladder as ladder_mod
from src.ladder import _build_jobs, evaluate_ladder


_TINY_LADDER = {"panel": ["weakest_first"], "boss": ["bossX"]}


def test_build_jobs_covers_both_formats_and_seats():
    jobs = _build_jobs(("brain", "x"), _TINY_LADDER, n_seeds=2)
    # Per opponent per seed: 2 one-v-one (both seats) + 1 four-player = 3 games.
    # 2 opponents × 2 seeds × 3 = 12.
    assert len(jobs) == 12
    fmts = [j[4] for j in jobs]
    assert fmts.count("1v1") == 8 and fmts.count("4p") == 4
    # 1v1 covers both seats; 4p num_players is 4.
    one_v_one = [j for j in jobs if j[4] == "1v1"]
    assert {j[6] for j in one_v_one} == {0, 1}
    assert all(j[7] == 4 for j in jobs if j[4] == "4p")
    assert all(j[7] == 2 for j in one_v_one)


class _FakeOutcome:
    def __init__(self, placement, faulted=False):
        self.placement = placement
        self.faulted = faulted


class _FakeResult:
    def __init__(self, outcomes):
        self.outcomes = outcomes


def test_evaluate_ladder_aggregates_and_writes_ledger(tmp_path, monkeypatch):
    # Stub run_episode: the agent always wins (placement 1 in its seat); the
    # other seats get placement 2. No env, no pool surprises (force 1 worker via
    # a serial imap so the monkeypatch is visible to the "worker").
    def fake_run_episode(line, config):
        n = config.num_players
        # find the agent seat = the one callable that isn't the opponent string
        outs = []
        for i in range(n):
            outs_placement = 1 if callable(line[i]) else 2
            outs.append(_FakeOutcome(outs_placement))
        return _FakeResult(outs)

    monkeypatch.setattr(ladder_mod, "run_episode", fake_run_episode)

    # Run jobs serially in-process so the stub applies (avoid real multiprocessing).
    import multiprocessing.pool as _pool

    class _SerialPool:
        def __init__(self, *a, **k):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def imap_unordered(self, fn, jobs):
            return [fn(j) for j in jobs]

    monkeypatch.setattr(ladder_mod.mp, "Pool", _SerialPool)

    csv_path = tmp_path / "ladder_log.csv"
    summary = evaluate_ladder(
        "roi_greedy",  # a brain (callable) so the stub can spot the agent's seat
        ladder=_TINY_LADDER,
        n_seeds=2,
        n_workers=1,
        build_id="testbuild",
        csv_path=str(csv_path),
        verbose=False,
    )

    # Agent won every game -> first_rate 1.0 everywhere, no faults.
    assert summary["faults"] == 0
    assert all(r["first_rate"] == 1.0 for r in summary["rows"])
    # Beating everything pushes the Elo proxy above the default start.
    assert summary["elo"] > 600.0
    # One row per (opponent, format): 2 opponents × {1v1, 4p} = 4 rows.
    assert len(summary["rows"]) == 4

    # Ledger written with the superset schema, header + 4 rows.
    with open(csv_path) as f:
        logged = list(csv.DictReader(f))
    assert len(logged) == 4
    assert {"build_id", "tier", "opponent", "format", "first_rate",
            "mean_placement", "elo_after"} <= set(logged[0].keys())


def test_evaluate_ladder_counts_faults(tmp_path, monkeypatch):
    def fault_run_episode(line, config):
        n = config.num_players
        outs = [_FakeOutcome(1, faulted=callable(line[i])) for i in range(n)]
        return _FakeResult(outs)

    monkeypatch.setattr(ladder_mod, "run_episode", fault_run_episode)

    class _SerialPool:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def imap_unordered(self, fn, jobs): return [fn(j) for j in jobs]

    monkeypatch.setattr(ladder_mod.mp, "Pool", _SerialPool)

    summary = evaluate_ladder(
        "roi_greedy", ladder={"panel": ["x"]},
        n_seeds=1, build_id="f", csv_path=None, verbose=False,
    )
    # Agent faulted in every game -> all excluded, counted as faults, no rows.
    assert summary["faults"] == 3   # 2 one-v-one seats + 1 four-player
    assert summary["rows"] == []
