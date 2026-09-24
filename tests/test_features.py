"""The feature layer, its two detectors, and the contract it has to keep."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from railrank import features as F
from railrank import journeys as J

SMALL = 900


@pytest.fixture(scope="module")
def data():
    return J.generate(n_searches=SMALL)


@pytest.fixture(scope="module")
def built(data):
    searches, options = data
    cfg = F.FeatureConfig(seats_missing_fill=F.seats_missing_fill_from(options))
    X, g, meta, hist = F.build_matrix(searches, options, cfg)
    return searches, options, cfg, X, g, meta, hist


# ── the contract ────────────────────────────────────────────────────────────

def test_feature_names_are_unique_and_the_vector_matches_them(built):
    _, _, cfg, X, _, _, _ = built
    assert len(set(F.FEATURE_NAMES)) == len(F.FEATURE_NAMES)
    assert X.shape[1] == len(F.FEATURE_NAMES)


def test_compute_returns_exactly_the_named_features(built):
    searches, options, cfg, _, _, _, hist = built
    s = searches[0]
    opts = [o for o in options if o.search_id == s.search_id]
    ctx = F.SearchContext(s.search_id, s.searched_at, s.origin, s.destination,
                          s.travel_date, s.departure_target_hour,
                          s.passenger_type, s.advance_days)
    oc = [F.OptionContext(o.option_id, o.price_pence, o.duration_minutes,
                          o.changes, o.departure_hour, o.departure_minute,
                          o.operator, o.ticket_type, o.seats_remaining,
                          o.carbon_grams, o.load_factor_at_search) for o in opts]
    f = F.compute(ctx, oc[0], F.SearchAggregates.of(oc), hist, cfg)
    assert set(f) == set(F.FEATURE_NAMES)


def test_no_feature_is_nan_or_infinite(built):
    _, _, _, X, _, _, _ = built
    assert not np.isnan(X).any()
    assert not np.isinf(X).any()


def test_groups_account_for_every_row(built):
    _, _, _, X, g, meta, _ = built
    assert sum(g) == X.shape[0] == len(meta)


def test_rows_are_ordered_by_search_so_the_groups_mean_something(built):
    _, _, _, _, g, meta, _ = built
    start = 0
    for n in g:
        ids = {m["search_id"] for m in meta[start:start + n]}
        assert len(ids) == 1, "a group spans more than one search"
        start += n


# ── within-search features ──────────────────────────────────────────────────

def test_the_cheapest_option_is_marked_cheapest(built):
    _, _, _, X, g, meta, _ = built
    i = F.FEATURE_NAMES.index("is_cheapest")
    j = F.FEATURE_NAMES.index("price_over_cheapest")
    start = 0
    for n in g[:60]:
        block = X[start:start + n]
        assert block[:, i].sum() >= 1
        assert np.isclose(block[block[:, i] == 1][:, j], 1.0).all()
        assert block[:, j].min() >= 1.0 - 1e-6
        start += n


def test_rank_percentiles_span_zero_to_one_within_a_search(built):
    _, _, _, X, g, _, _ = built
    i = F.FEATURE_NAMES.index("price_rank_pct")
    start = 0
    for n in g[:40]:
        block = X[start:start + n, i]
        assert block.min() == pytest.approx(0.0)
        assert block.max() == pytest.approx(1.0)
        start += n


# ── point in time ───────────────────────────────────────────────────────────

def test_history_as_of_excludes_the_day_itself():
    """A search on the 3rd must not see prices from searches later that day."""
    rows = [{"origin": "A", "destination": "B", "searched_at": f"2026-07-0{d}T09:00:00",
             "price_pence": p}
            for d, p in [(1, 1000), (2, 2000), (3, 9000)]]
    h = F.RouteHistory.build(rows)
    assert h.median_price("A|B", date(2026, 7, 3), 7) == pytest.approx(15.0)
    assert h.median_price("A|B", date(2026, 7, 4), 7) == pytest.approx(20.0)


def test_history_returns_none_before_any_data_exists():
    h = F.RouteHistory.build([{"origin": "A", "destination": "B",
                               "searched_at": "2026-07-05T09:00:00",
                               "price_pence": 1000}])
    assert h.median_price("A|B", date(2026, 7, 1), 7) is None
    assert h.median_price("Z|Z", date(2026, 7, 9), 7) is None


def test_the_window_really_is_a_window():
    rows = [{"origin": "A", "destination": "B",
             "searched_at": f"2026-07-{d:02d}T09:00:00", "price_pence": 1000 * d}
            for d in range(1, 21)]
    h = F.RouteHistory.build(rows)
    wide = h.median_price("A|B", date(2026, 7, 20), 19)
    narrow = h.median_price("A|B", date(2026, 7, 20), 3)
    assert narrow > wide


def test_ignoring_as_of_lets_a_row_see_the_future():
    rows = [{"origin": "A", "destination": "B",
             "searched_at": f"2026-07-{d:02d}T09:00:00", "price_pence": 1000 * d}
            for d in range(1, 21)]
    h = F.RouteHistory.build(rows)
    assert h.median_price("A|B", date(2026, 7, 2), 7) == pytest.approx(10.0)
    assert h.median_price("A|B", date(2026, 7, 2), 7, ignore_as_of=True) > 50.0


def test_missing_history_is_flagged_rather_than_imputed(built):
    """Imputing 1.0 would be a lie a tree model cannot tell from a real ratio."""
    _, _, _, X, _, _, _ = built
    known = X[:, F.FEATURE_NAMES.index("route_median_price_known")]
    ratio = X[:, F.FEATURE_NAMES.index("price_over_route_median_7d")]
    assert (known == 0).sum() > 0
    assert np.isclose(ratio[known == 0], 1.0).all()


def test_point_in_time_report_finds_the_difference(built):
    searches, options, cfg, _, _, _, _ = built
    r = F.point_in_time_report(searches, options, cfg, sample=300)
    assert r["share_that_differ"] > 0.2
    assert r["max_absolute_difference"] > 0


# ── skew ────────────────────────────────────────────────────────────────────

def test_no_skew_between_identical_configs(built):
    searches, options, cfg, _, _, _, hist = built
    r = F.skew_report(searches, options, cfg, cfg, hist, sample=80)
    assert r["skew_detected"] is False
    assert r["features_that_differ"] == []
    assert r["rows_compared"] > 0


def test_skew_from_a_timezone_off_by_one(built):
    searches, options, cfg, _, _, _, hist = built
    r = F.skew_report(searches, options, cfg,
                      replace(cfg, advance_days_off_by_one=True), hist, sample=80)
    assert r["features_that_differ"] == ["advance_days"]
    assert r["largest_absolute_difference"]["advance_days"] == pytest.approx(1.0)


def test_skew_from_a_different_missing_value_fill(built):
    searches, options, cfg, _, _, _, hist = built
    r = F.skew_report(searches, options, cfg,
                      replace(cfg, seats_missing_fill=0.0), hist, sample=80)
    assert r["features_that_differ"] == ["seats_remaining"]
    assert r["mismatch_counts"]["seats_remaining"] > 0


def test_skew_from_the_leak_being_switched_on(built):
    searches, options, cfg, _, _, _, hist = built
    r = F.skew_report(searches, options, cfg,
                      replace(cfg, use_final_load_factor=True), hist, sample=80)
    assert "load_factor" in r["features_that_differ"]


def test_config_fingerprint_changes_with_the_config():
    a = F.FeatureConfig()
    assert a.fingerprint() == F.FeatureConfig().fingerprint()
    assert a.fingerprint() != replace(a, seats_missing_fill=1.0).fingerprint()
    assert len(a.fingerprint()) == 12


def test_the_fill_is_a_statistic_of_the_data(data):
    _, options = data
    fill = F.seats_missing_fill_from(options)
    known = [o.seats_remaining for o in options if o.seats_remaining is not None]
    assert min(known) <= fill <= max(known)


def test_a_missing_seat_count_is_flagged_as_well_as_filled(built):
    _, _, cfg, X, _, _, _ = built
    miss = X[:, F.FEATURE_NAMES.index("seats_missing")]
    seats = X[:, F.FEATURE_NAMES.index("seats_remaining")]
    assert miss.sum() > 0
    assert np.isclose(seats[miss == 1], cfg.seats_missing_fill).all()
