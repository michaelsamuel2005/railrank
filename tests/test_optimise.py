"""The optimiser against hand-computable cases and against brute force."""

from __future__ import annotations

import random

import pytest

from railrank import journeys as J
from railrank import optimise as O

TINY = [("A", 0), ("B", 50), ("C", 100), ("D", 160)]


def test_a_through_fare_wins_when_it_is_cheapest():
    fares = {(0, 1): 1000, (1, 2): 1000, (0, 2): 1500}
    total, legs = O.cheapest_split(fares, TINY[:3])
    assert total == 1500
    assert len(legs) == 1
    assert (legs[0].origin, legs[0].destination) == ("A", "C")


def test_splitting_wins_when_the_fares_are_not_additive():
    fares = {(0, 1): 400, (1, 2): 400, (0, 2): 1500}
    total, legs = O.cheapest_split(fares, TINY[:3])
    assert total == 800
    assert [(leg.origin, leg.destination) for leg in legs] == [("A", "B"), ("B", "C")]


def test_a_split_point_the_service_does_not_call_at_is_not_available():
    """The rule that makes this a constrained problem rather than a lookup."""
    fares = {(0, 1): 400, (1, 2): 400, (0, 2): 1500}
    assert O.cheapest_split(fares, TINY[:3], calls_at={0, 1, 2})[0] == 800
    assert O.cheapest_split(fares, TINY[:3], calls_at={0, 2})[0] == 1500


def test_three_legs_beat_two_when_that_is_what_the_table_says():
    fares = {(0, 1): 300, (1, 2): 300, (2, 3): 300,
             (0, 2): 900, (1, 3): 900, (0, 3): 2000}
    total, legs = O.cheapest_split(fares, TINY)
    assert total == 900
    assert len(legs) == 3


def test_it_raises_rather_than_guessing_when_no_fare_covers_the_journey():
    with pytest.raises(ValueError):
        O.cheapest_split({(0, 1): 500}, TINY[:3])


def test_it_raises_when_the_service_skips_an_endpoint():
    fares = {(0, 1): 400, (1, 2): 400, (0, 2): 1500}
    with pytest.raises(ValueError):
        O.cheapest_split(fares, TINY[:3], calls_at={0, 1}, end=2)


def test_brute_force_refuses_an_instance_it_cannot_finish():
    """A verification that quietly stops verifying is worse than none."""
    calling = [(f"S{i}", i * 10) for i in range(40)]
    fares = O.fare_table(calling, seed=1)
    with pytest.raises(ValueError, match="combinations"):
        O.brute_force_split(fares, calling, max_intermediate=16)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_solver_agrees_with_brute_force_on_random_instances(seed):
    rng = random.Random(seed)
    for calling in J.ROUTES.values():
        fares = O.fare_table(calling, seed=seed * 31 + len(calling))
        n = len(calling)
        for _ in range(12):
            a = rng.randrange(0, n - 1)
            b = rng.randrange(a + 1, n)
            calls = {a, b} | {i for i in range(a + 1, b) if rng.random() < 0.7}
            assert (O.cheapest_split(fares, calling, calls, a, b)[0]
                    == O.brute_force_split(fares, calling, calls, a, b)[0])


def test_the_legs_actually_cover_the_journey_without_gaps():
    rng = random.Random(9)
    calling = J.ROUTES[("London Kings Cross", "Edinburgh")]
    fares = O.fare_table(calling, seed=5)
    for _ in range(25):
        a = rng.randrange(0, len(calling) - 1)
        b = rng.randrange(a + 1, len(calling))
        _, legs = O.cheapest_split(fares, calling, None, a, b)
        assert legs[0].i == a and legs[-1].j == b
        for left, right in zip(legs, legs[1:]):
            assert left.j == right.i        # consecutive, no gap, no overlap


def test_the_total_is_the_sum_of_the_legs():
    calling = J.ROUTES[("London Euston", "Glasgow Central")]
    fares = O.fare_table(calling, seed=12)
    total, legs = O.cheapest_split(fares, calling)
    assert total == sum(leg.pence for leg in legs)


def test_the_per_mile_rate_tapers_with_distance():
    """Longer journeys cost more in total and less per mile, as GB fares do."""
    assert O._rate(25) > O._rate(100) > O._rate(400)
    assert 25 * O._rate(25) < 400 * O._rate(400)


def test_no_fare_falls_below_the_market_floor():
    """An earlier table priced 290 miles at 13 pounds, which made the
    optimiser look brilliant by making the inputs impossible."""
    for calling in J.ROUTES.values():
        fares = O.fare_table(calling, seed=7)
        for (i, j), pence in fares.items():
            miles = calling[j][1] - calling[i][1]
            scale = O.FIXED_PENCE + O._rate(miles) * miles
            # Fares are published in whole pounds, so the floor is applied
            # before rounding and a fare can land up to 50p under it.
            assert pence >= scale * O.MARKET_FLOOR - 50


def test_summarise_reports_nothing_when_nothing_splits():
    inst = [{"through_pence": 1000, "split_pence": 1000, "saving_pence": 0,
             "saving_pct": 0.0, "n_tickets": 1}]
    out = O.summarise(inst)
    assert out["journeys_where_splitting_wins"] == 0
    assert out["median_saving_pence"] == 0
