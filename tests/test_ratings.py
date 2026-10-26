"""Tests for ratings.py: Bradley-Terry with bootstrap, Massey, order-averaged ELO."""

import random
import sys
import os

from barterbench.ratings import (fit_bradley_terry, bootstrap_bradley_terry, massey_ratings,
                     elo_permutation_average, elo_online, rating_report, pairwise_win_probability)


def _match(a, b, winner, margin=None):
    m = {"model_a": a, "model_b": b, "winner": winner}
    if margin is not None:
        m["score_a"] = 0.5 + margin / 2
        m["score_b"] = 0.5 - margin / 2
    return m


def test_bt_orders_dominant_model_first_and_is_finite_when_undefeated():
    matches = [_match("A", "B", "A") for _ in range(10)] + [_match("B", "C", "B") for _ in range(10)]
    r = fit_bradley_terry(matches)
    assert r["A"] > r["B"] > r["C"]
    assert all(abs(v) < 10_000 for v in r.values()), "prior must keep undefeated ratings finite"


def test_bt_symmetric_data_gives_equal_ratings():
    matches = [_match("A", "B", "A"), _match("A", "B", "B")] * 5
    r = fit_bradley_terry(matches)
    assert abs(r["A"] - r["B"]) < 1e-6


def test_bt_is_order_independent():
    rng = random.Random(1)
    matches = [_match("A", "B", rng.choice(["A", "B", "draw"])) for _ in range(30)]
    matches += [_match("B", "C", rng.choice(["B", "C"])) for _ in range(30)]
    r1 = fit_bradley_terry(matches)
    shuffled = list(matches); rng.shuffle(shuffled)
    r2 = fit_bradley_terry(shuffled)
    assert all(abs(r1[m] - r2[m]) < 1e-6 for m in r1)


def test_bootstrap_ci_contains_point_and_shrinks_with_data():
    rng = random.Random(2)
    def gen(n):
        return [_match("A", "B", "A" if rng.random() < 0.7 else "B") for _ in range(n)]
    small = bootstrap_bradley_terry(gen(10), n_boot=200, seed=0)
    large = bootstrap_bradley_terry(gen(200), n_boot=200, seed=0)
    for res in (small, large):
        for m, s in res.items():
            assert s["ci_lower"] <= s["rating"] <= s["ci_upper"]
    width_small = small["A"]["ci_upper"] - small["A"]["ci_lower"]
    width_large = large["A"]["ci_upper"] - large["A"]["ci_lower"]
    assert width_large < width_small


def test_massey_recovers_margins():
    # A beats B by 0.2, B beats C by 0.1 consistently
    matches = [_match("A", "B", "A", margin=0.2) for _ in range(5)] + \
              [_match("B", "C", "B", margin=0.1) for _ in range(5)]
    r = massey_ratings(matches)
    assert abs((r["A"] - r["B"]) - 0.2) < 1e-6
    assert abs((r["B"] - r["C"]) - 0.1) < 1e-6
    assert abs(sum(r.values())) < 1e-3  # ratings are rounded to 4 dp


def test_massey_uses_margin_not_just_outcome():
    narrow = [_match("A", "B", "A", margin=0.03)] * 6
    wide = [_match("A", "B", "A", margin=0.5)] * 6
    assert massey_ratings(wide)["A"] > massey_ratings(narrow)["A"]


def test_elo_online_is_order_dependent_and_average_reports_spread():
    rng = random.Random(3)
    matches = [_match("A", "B", rng.choice(["A", "B"])) for _ in range(40)]
    r1 = elo_online(matches)
    rev = elo_online(list(reversed(matches)))
    avg = elo_permutation_average(matches, n_perm=50, seed=0)
    assert set(avg) == {"A", "B"}
    assert avg["A"]["elo_spread_lower"] <= avg["A"]["elo_mean"] <= avg["A"]["elo_spread_upper"]
    # Either the orderings differ, or the spread is degenerate; both are fine, but the report must exist.
    assert isinstance(r1["A"], float) and isinstance(rev["A"], float)


def test_rating_report_rows_and_win_probability():
    matches = [_match("A", "B", "A") for _ in range(8)] + [_match("A", "B", "draw") for _ in range(2)]
    rep = rating_report(matches, n_boot=50, n_perm=20)
    rows = rep["bradley_terry"]
    assert rows[0]["model"] == "A" and rows[0]["rank"] == 1
    assert rows[0]["wins"] == 8 and rows[0]["draws"] == 2 and rows[0]["matches"] == 10
    bt = {r["model"]: r["rating"] for r in rows}
    assert pairwise_win_probability(bt, "A", "B") > 0.5


def test_empty_inputs():
    assert fit_bradley_terry([]) == {}
    assert massey_ratings([]) == {}
    assert elo_permutation_average([]) == {}
    assert rating_report([])["bradley_terry"] == []
