"""The generator is the answer key, so it is tested before anything that reads it."""

from __future__ import annotations

import statistics as st
from collections import Counter, defaultdict

import pytest

from railrank import journeys as J

SMALL = 2500


@pytest.fixture(scope="module")
def data():
    return J.generate(n_searches=SMALL)


def test_generation_is_reproducible():
    a, _ = J.generate(n_searches=400)
    b, _ = J.generate(n_searches=400)
    assert [s.search_id for s in a] == [s.search_id for s in b]


def test_a_different_seed_gives_different_data():
    _, a = J.generate(n_searches=400)
    _, b = J.generate(n_searches=400, seed=J.SEED + 1)
    assert [o.clicked for o in a] != [o.clicked for o in b]


def test_every_option_belongs_to_a_search(data):
    searches, options = data
    ids = {s.search_id for s in searches}
    assert all(o.search_id in ids for o in options)


def test_positions_are_a_permutation_within_each_search(data):
    _, options = data
    by = defaultdict(list)
    for o in options:
        by[o.search_id].append(o.position)
    for positions in list(by.values())[:300]:
        assert sorted(positions) == list(range(1, len(positions) + 1))


def test_a_booking_implies_a_click(data):
    _, options = data
    assert not [o for o in options if o.booked and not o.clicked]


def test_at_most_one_booking_per_search(data):
    _, options = data
    booked = Counter(o.search_id for o in options if o.booked)
    assert not booked or max(booked.values()) == 1


def test_clicks_fall_with_position(data):
    _, options = data
    clicks, shown = Counter(), Counter()
    for o in options:
        shown[o.position] += 1
        clicks[o.position] += o.clicked
    rate = [clicks[p] / shown[p] for p in range(1, 9)]
    assert rate[0] > rate[2] > rate[6]


def test_exploration_traffic_is_roughly_the_configured_share(data):
    searches, _ = data
    share = sum(s.exploration for s in searches) / len(searches)
    assert share == pytest.approx(J.EXPLORATION_FRACTION, abs=0.03)


def test_exploration_searches_are_not_sorted_by_price(data):
    """If they were, they would not identify the examination curve."""
    searches, options = data
    by = defaultdict(list)
    for o in options:
        by[o.search_id].append(o)
    expl = [s for s in searches if s.exploration][:200]
    inversions = 0
    for s in expl:
        opts = sorted(by[s.search_id], key=lambda o: o.position)
        prices = [o.price_pence for o in opts]
        inversions += sum(1 for a, b in zip(prices, prices[1:]) if a > b)
    assert inversions > len(expl)


def test_the_logging_policy_is_roughly_a_price_sort(data):
    searches, options = data
    by = defaultdict(list)
    for o in options:
        by[o.search_id].append(o)
    normal = [s for s in searches if not s.exploration][:200]
    agree = 0
    for s in normal:
        opts = sorted(by[s.search_id], key=lambda o: o.position)
        prices = [o.price_pence for o in opts]
        agree += sum(1 for a, b in zip(prices, prices[1:]) if a <= b) / max(len(prices) - 1, 1)
    assert 0.6 < agree / len(normal) < 0.98


def test_the_utility_model_separates_traveller_types(data):
    """Leisure picks cheaper, business picks faster, measured within the
    search so route length cannot confound it."""
    searches, options = data
    by_id = {s.search_id: s for s in searches}
    by = defaultdict(list)
    for o in options:
        by[o.search_id].append(o)
    dur_rank, price_rank = defaultdict(list), defaultdict(list)
    for sid, opts in by.items():
        if len(opts) < 3:
            continue
        t = by_id[sid].passenger_type
        best = max(opts, key=lambda o: o.true_utility)
        n = len(opts) - 1
        dur_rank[t].append(sorted(opts, key=lambda o: o.duration_minutes).index(best) / n)
        price_rank[t].append(sorted(opts, key=lambda o: o.price_pence).index(best) / n)
    assert st.mean(price_rank["leisure"]) < st.mean(price_rank["business"])
    assert st.mean(dur_rank["business"]) < st.mean(dur_rank["leisure"])


def test_the_leak_carries_the_label_and_the_honest_column_does_not(data):
    _, options = data
    clicked = [o for o in options if o.clicked]
    rest = [o for o in options if not o.clicked]
    leak_gap = (st.mean(o.final_load_factor for o in clicked)
                - st.mean(o.final_load_factor for o in rest))
    honest_gap = (st.mean(o.load_factor_at_search for o in clicked)
                  - st.mean(o.load_factor_at_search for o in rest))
    assert leak_gap > 0.08
    assert abs(honest_gap) < 0.02


def test_the_final_load_never_falls_below_the_load_at_search(data):
    _, options = data
    assert all(o.final_load_factor >= o.load_factor_at_search - 1e-9
               for o in options)


def test_the_seat_reporting_change_starts_when_it_says_it_does(data):
    searches, options = data
    by_id = {s.search_id: s for s in searches}
    start = J.SEATS_REPORTING_CHANGE["from_day"]
    affected = J.SEATS_REPORTING_CHANGE["operators"]
    n = len(searches)
    early_missing = late_missing = early_zero = late_zero = 0
    for o in options:
        if o.operator not in affected:
            continue
        idx = int(by_id[o.search_id].search_id[1:])
        day = int(idx / n * J.PERIOD_DAYS)
        if day < start:
            early_missing += o.seats_remaining is None
            early_zero += o.seats_remaining == 0
        else:
            late_missing += o.seats_remaining is None
            late_zero += o.seats_remaining == 0
    assert early_missing > 0 and late_missing == 0
    assert late_zero > early_zero * 5


def test_fares_and_durations_are_plausible(data):
    _, options = data
    fares = sorted(o.price_pence / 100 for o in options)
    dur = sorted(o.duration_minutes for o in options)
    assert 20 < st.median(fares) < 140
    assert fares[0] > 5 and fares[-1] < 600
    assert 40 < st.median(dur) < 480


def test_revenue_relevant_fields_are_within_range(data):
    _, options = data
    for o in options:
        assert o.price_pence > 0
        assert 0 <= o.load_factor_at_search <= 1
        assert 0 <= o.final_load_factor <= 1
        assert o.changes in (0, 1, 2)
        assert o.ticket_type in J.TICKET_TYPES
        assert o.operator in J.OPERATORS


def test_the_answer_key_describes_what_was_planted():
    truth = J.planted()
    for key in ("preference_weights", "examination_eta", "logging_policy",
                "leak", "seats_reporting_change", "price_trend_per_day"):
        assert key in truth
    assert set(truth["preference_weights"]) == set(J.PASSENGER_TYPES)


def test_to_rows_joins_without_losing_anything(data):
    searches, options = data
    rows = J.to_rows(searches, options)
    assert len(rows) == len(options)
    assert {"search_id", "option_id", "price_pence", "true_utility"} <= set(rows[0])
