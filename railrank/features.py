"""
features.py — one function, called by training and by serving.

This module exists because of a specific failure. A model is trained on a
feature table assembled by a batch job, and served by a request handler that
computes the same features again, in different code, written at a different
time, by somebody who reasonably assumed the definitions were obvious. The two
drift apart. The model keeps returning confident predictions, offline metrics
keep looking fine, and the only symptom is that the thing does not work.

So there is exactly one function here — `compute` — and both callers use it.
That is the whole design. Everything below is in service of it.

Two failure modes get their own machinery, because both are silent:

**Point in time.** A feature may only use information that existed when the
search happened. `RouteHistory` is built from daily aggregates and every lookup
takes an `as_of` date, so a feature computed for a search on the 3rd cannot see
the 4th. The tempting alternative -- compute the aggregate once over the whole
table and join it -- is faster, simpler, and wrong, and it is wrong in the
direction that makes the offline metric go up. `leak_check` measures exactly
how far up.

**Training and serving skew.** The same function still gives different answers
if the two callers pass different configuration. The fill value for a missing
seat count is the obvious one: training uses a statistic of the training set,
serving has to be told what that statistic was. So `FeatureConfig` travels
with the model as an artefact, and `skew_report` replays logged requests
through the serving path and compares the result to the training row, field by
field. A mismatch is a failed build, not a warning.

The feature names are deliberately boring and stable. Renaming a feature is a
schema migration, and a model artefact records the order it was trained on.
"""

from __future__ import annotations

import bisect
import math
import statistics
from dataclasses import dataclass, field, replace
from datetime import date, timedelta

# The order is part of the contract. XGBoost takes a dense matrix and does not
# know the names, so a reordering here silently rebinds every feature to a
# different column. `train` records this list in the model artefact and
# `service` asserts the loaded model agrees with it.
FEATURE_NAMES: tuple[str, ...] = (
    "price_gbp",
    "duration_hours",
    "changes",
    "is_direct",
    "departure_hour",
    "departure_penalty_hours",
    "advance_days",
    "ticket_flexibility",
    "seats_remaining",
    "seats_missing",
    "load_factor",
    "carbon_kg",
    "operator_rank",
    "price_rank_pct",
    "duration_rank_pct",
    "price_over_cheapest",
    "duration_over_fastest",
    "is_cheapest",
    "is_fastest",
    "price_over_route_median_7d",
    "route_median_price_known",
)

# Ordinal, not one-hot: the ticket types form a genuine flexibility ordering
# and a tree model splits an ordinal cleanly. One-hot would spend four columns
# on the same information.
TICKET_FLEXIBILITY = {"Advance": 0.0, "Super Off-Peak": 1.0,
                      "Off-Peak": 2.0, "Anytime": 3.0}

# Operators ranked by the punctuality and comfort a traveller would notice.
# This is a judgement encoded as data, so it lives here and not inside the
# model, where nobody would find it.
OPERATOR_RANK = {"LNER": 5.0, "GWR": 4.0, "Avanti West Coast": 4.0,
                 "TransPennine Express": 3.0, "CrossCountry": 2.0,
                 "Northern": 1.0}


@dataclass(frozen=True)
class FeatureConfig:
    """Everything the feature code needs that is not in the request.

    It is frozen and it is written into the model artefact. A serving process
    that constructs its own config instead of loading the model's has
    reintroduced the bug this class exists to prevent, which is why
    `service.py` loads it and never builds one.
    """
    seats_missing_fill: float = 40.0
    history_window_days: int = 7
    # Deliberate faults, off by default. They are parameters rather than
    # branches in the feature code so the fault can be switched on in a test
    # and measured, instead of being described in a comment nobody can check.
    use_final_load_factor: bool = False      # the leak
    ignore_as_of: bool = False               # the point-in-time violation
    advance_days_off_by_one: bool = False    # the skew

    def fingerprint(self) -> str:
        """A short stable digest, logged with every model."""
        import hashlib
        import json
        payload = json.dumps(self.__dict__, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


# ── the history a feature is allowed to see ─────────────────────────────────

class RouteHistory:
    """Daily price aggregates per route, queryable as of a date.

    Daily rather than per-row on purpose: it is how a feature store actually
    stores this, it makes the as-of lookup a binary search instead of a scan,
    and it means the trailing median is reproducible from a table somebody can
    inspect rather than from a window function nobody can.
    """

    def __init__(self) -> None:
        self._days: dict[str, list[date]] = {}
        self._median: dict[str, list[float]] = {}

    @classmethod
    def build(cls, rows) -> RouteHistory:
        buckets: dict[tuple[str, date], list[float]] = {}
        for r in rows:
            key = (f"{r['origin']}|{r['destination']}",
                   date.fromisoformat(r["searched_at"][:10]))
            buckets.setdefault(key, []).append(r["price_pence"] / 100.0)
        h = cls()
        for (route, day), prices in sorted(buckets.items()):
            h._days.setdefault(route, []).append(day)
            h._median.setdefault(route, []).append(statistics.median(prices))
        return h

    def median_price(self, route: str, as_of: date, window_days: int,
                     ignore_as_of: bool = False) -> float | None:
        """Median fare on this route over the `window_days` before `as_of`.

        `as_of` is exclusive: a search on the 3rd sees the 2nd and earlier.
        Including the same day would let a search see prices from searches that
        happened after it, which is the leak in miniature.
        """
        days = self._days.get(route)
        if not days:
            return None
        if ignore_as_of:
            # The fault: every row sees the whole table. This is what a join
            # against a precomputed aggregate does when nobody filters it.
            vals = self._median[route]
            return statistics.median(vals) if vals else None
        hi = bisect.bisect_left(days, as_of)
        lo = bisect.bisect_left(days, as_of - timedelta(days=window_days))
        window = self._median[route][lo:hi]
        return statistics.median(window) if window else None


@dataclass
class SearchContext:
    """The parts of a search a feature needs, and nothing else."""
    search_id: str
    searched_at: str
    origin: str
    destination: str
    travel_date: str
    departure_target_hour: int
    passenger_type: str
    advance_days: int

    @property
    def route(self) -> str:
        return f"{self.origin}|{self.destination}"


@dataclass
class OptionContext:
    option_id: str
    price_pence: int
    duration_minutes: int
    changes: int
    departure_hour: int
    departure_minute: int
    operator: str
    ticket_type: str
    seats_remaining: int | None
    carbon_grams: int
    load_factor_at_search: float
    final_load_factor: float | None = None


@dataclass
class SearchAggregates:
    """The within-search statistics a ranking feature needs.

    Computed once per search and passed in, because every option in a search
    shares them and recomputing per option would be quadratic for no reason.
    """
    n: int
    cheapest_pence: int
    fastest_minutes: int
    price_sorted: list[int] = field(default_factory=list)
    duration_sorted: list[int] = field(default_factory=list)

    @classmethod
    def of(cls, options) -> SearchAggregates:
        prices = sorted(o.price_pence for o in options)
        durations = sorted(o.duration_minutes for o in options)
        return cls(n=len(options), cheapest_pence=prices[0],
                   fastest_minutes=durations[0],
                   price_sorted=prices, duration_sorted=durations)


def _pct_rank(sorted_values: list, value) -> float:
    n = len(sorted_values)
    if n <= 1:
        return 0.0
    return bisect.bisect_left(sorted_values, value) / (n - 1)


def _advance_days(search: SearchContext, cfg: FeatureConfig) -> float:
    """Days between the search and the travel date.

    The off-by-one fault reproduces a real one: training computed the
    difference between two dates in UTC, and the serving host was on local
    time, so a search after 23:00 in summer resolved to the next day and every
    advance window shifted by one. It changes a feature by a single unit and
    nothing anywhere throws.
    """
    travel = date.fromisoformat(search.travel_date)
    searched = date.fromisoformat(search.searched_at[:10])
    days = (travel - searched).days
    if cfg.advance_days_off_by_one:
        days -= 1
    return float(max(days, 0))


# ── the one function ────────────────────────────────────────────────────────

def compute(search: SearchContext, option: OptionContext,
            aggregates: SearchAggregates, history: RouteHistory,
            cfg: FeatureConfig) -> dict[str, float]:
    """Features for one option in one search. Used by training and by serving.

    Every value here is knowable at `search.searched_at`, with two exceptions
    that only appear when the corresponding fault is switched on in `cfg`.
    """
    price_gbp = option.price_pence / 100.0
    duration_hours = option.duration_minutes / 60.0
    departure = option.departure_hour + option.departure_minute / 60.0
    penalty = abs(departure - search.departure_target_hour)

    seats_missing = option.seats_remaining is None
    seats = cfg.seats_missing_fill if seats_missing else float(option.seats_remaining)

    if cfg.use_final_load_factor:
        # The leak. Same column, read after the train left.
        load = (option.final_load_factor if option.final_load_factor is not None
                else option.load_factor_at_search)
    else:
        load = option.load_factor_at_search

    as_of = date.fromisoformat(search.searched_at[:10])
    route_median = history.median_price(search.route, as_of,
                                        cfg.history_window_days,
                                        ignore_as_of=cfg.ignore_as_of)

    return {
        "price_gbp": price_gbp,
        "duration_hours": duration_hours,
        "changes": float(option.changes),
        "is_direct": 1.0 if option.changes == 0 else 0.0,
        "departure_hour": float(option.departure_hour),
        "departure_penalty_hours": penalty,
        "advance_days": _advance_days(search, cfg),
        "ticket_flexibility": TICKET_FLEXIBILITY.get(option.ticket_type, 1.0),
        "seats_remaining": seats,
        "seats_missing": 1.0 if seats_missing else 0.0,
        "load_factor": float(load),
        "carbon_kg": option.carbon_grams / 1000.0,
        "operator_rank": OPERATOR_RANK.get(option.operator, 2.0),
        "price_rank_pct": _pct_rank(aggregates.price_sorted, option.price_pence),
        "duration_rank_pct": _pct_rank(aggregates.duration_sorted,
                                       option.duration_minutes),
        "price_over_cheapest": option.price_pence / max(aggregates.cheapest_pence, 1),
        "duration_over_fastest": option.duration_minutes / max(aggregates.fastest_minutes, 1),
        "is_cheapest": 1.0 if option.price_pence == aggregates.cheapest_pence else 0.0,
        "is_fastest": 1.0 if option.duration_minutes == aggregates.fastest_minutes else 0.0,
        # A ratio needs a denominator. Early in the period there is no history
        # for a route, and the honest answer is "not known" -- carried as its
        # own flag rather than imputed to 1.0, which would be a lie a tree
        # model cannot distinguish from a genuine ratio of one.
        "price_over_route_median_7d": (price_gbp / route_median
                                       if route_median else 1.0),
        "route_median_price_known": 0.0 if route_median is None else 1.0,
    }


def vector(features: dict[str, float]) -> list[float]:
    """Dict to dense row, in the one order the model was trained on."""
    return [features[name] for name in FEATURE_NAMES]


# ── building a training table ───────────────────────────────────────────────

def build_matrix(searches, options, cfg: FeatureConfig,
                 history: RouteHistory | None = None):
    """Feature rows, labels and group sizes for every search, in search order.

    Groups matter: a ranking objective needs to know which rows compete with
    each other, and `xgboost` takes that as a list of group sizes over rows
    sorted by group. Getting this wrong does not raise -- it trains a model
    that ranks across searches, which is meaningless and scores about as well
    as it deserves to.
    """
    import numpy as np

    by_search: dict[str, list] = {}
    for o in options:
        by_search.setdefault(o.search_id, []).append(o)
    searches_sorted = sorted(searches, key=lambda s: (s.searched_at, s.search_id))

    if history is None:
        history = RouteHistory.build(
            [{"origin": s.origin, "destination": s.destination,
              "searched_at": s.searched_at, "price_pence": o.price_pence}
             for s in searches for o in by_search.get(s.search_id, [])])

    rows, groups, meta = [], [], []
    for s in searches_sorted:
        opts = by_search.get(s.search_id)
        if not opts:
            continue
        ctx = SearchContext(
            search_id=s.search_id, searched_at=s.searched_at, origin=s.origin,
            destination=s.destination, travel_date=s.travel_date,
            departure_target_hour=s.departure_target_hour,
            passenger_type=s.passenger_type, advance_days=s.advance_days)
        oc = [OptionContext(
            option_id=o.option_id, price_pence=o.price_pence,
            duration_minutes=o.duration_minutes, changes=o.changes,
            departure_hour=o.departure_hour, departure_minute=o.departure_minute,
            operator=o.operator, ticket_type=o.ticket_type,
            seats_remaining=o.seats_remaining, carbon_grams=o.carbon_grams,
            load_factor_at_search=o.load_factor_at_search,
            final_load_factor=o.final_load_factor) for o in opts]
        agg = SearchAggregates.of(oc)
        for o, c in zip(opts, oc):
            rows.append(vector(compute(ctx, c, agg, history, cfg)))
            meta.append({"search_id": s.search_id, "option_id": o.option_id,
                         "position": o.position, "clicked": o.clicked,
                         "booked": o.booked, "true_utility": o.true_utility,
                         "exploration": s.exploration,
                         "passenger_type": s.passenger_type,
                         "searched_at": s.searched_at})
        groups.append(len(opts))
    return np.asarray(rows, dtype=np.float32), groups, meta, history


def seats_missing_fill_from(options) -> float:
    """The fill value, computed from training data and then frozen.

    It has to be recorded rather than recomputed, because a serving process
    that recomputes it from its own traffic will drift away from the value the
    model was fitted against, one request at a time.
    """
    known = [o.seats_remaining for o in options if o.seats_remaining is not None]
    return float(statistics.median(known)) if known else 40.0


# ── the two detectors ───────────────────────────────────────────────────────

def skew_report(searches, options, train_cfg: FeatureConfig,
                serve_cfg: FeatureConfig, history: RouteHistory,
                sample: int = 400) -> dict:
    """Replay logged searches through both configs and compare, field by field.

    This is the check a deployment should run before it takes traffic. It does
    not ask whether the serving code looks right; it asks whether it produces
    the same numbers, on real logged inputs, as the code that trained the model.
    """
    by_search: dict[str, list] = {}
    for o in options:
        by_search.setdefault(o.search_id, []).append(o)

    mismatches: dict[str, int] = {}
    worst: dict[str, float] = {}
    compared = 0
    for s in sorted(searches, key=lambda x: x.search_id)[:sample]:
        opts = by_search.get(s.search_id)
        if not opts:
            continue
        ctx = SearchContext(
            search_id=s.search_id, searched_at=s.searched_at, origin=s.origin,
            destination=s.destination, travel_date=s.travel_date,
            departure_target_hour=s.departure_target_hour,
            passenger_type=s.passenger_type, advance_days=s.advance_days)
        oc = [OptionContext(
            option_id=o.option_id, price_pence=o.price_pence,
            duration_minutes=o.duration_minutes, changes=o.changes,
            departure_hour=o.departure_hour, departure_minute=o.departure_minute,
            operator=o.operator, ticket_type=o.ticket_type,
            seats_remaining=o.seats_remaining, carbon_grams=o.carbon_grams,
            load_factor_at_search=o.load_factor_at_search,
            final_load_factor=o.final_load_factor) for o in opts]
        agg = SearchAggregates.of(oc)
        for c in oc:
            a = compute(ctx, c, agg, history, train_cfg)
            b = compute(ctx, c, agg, history, serve_cfg)
            compared += 1
            for name in FEATURE_NAMES:
                if not math.isclose(a[name], b[name], rel_tol=1e-9, abs_tol=1e-9):
                    mismatches[name] = mismatches.get(name, 0) + 1
                    worst[name] = max(worst.get(name, 0.0), abs(a[name] - b[name]))

    return {
        "rows_compared": compared,
        "train_config": train_cfg.fingerprint(),
        "serve_config": serve_cfg.fingerprint(),
        "configs_identical": train_cfg == serve_cfg,
        "features_that_differ": sorted(mismatches),
        "mismatch_counts": mismatches,
        "largest_absolute_difference": {k: round(v, 6) for k, v in worst.items()},
        "skew_detected": bool(mismatches),
        "verdict": (
            f"SKEW: {len(mismatches)} feature(s) differ between the training and "
            f"serving configurations over {compared:,} replayed rows "
            f"({', '.join(sorted(mismatches))}). The model was fitted against "
            f"the first and would be served the second."
            if mismatches else
            f"No skew: {compared:,} replayed rows agree on all "
            f"{len(FEATURE_NAMES)} features."),
    }


def point_in_time_report(searches, options, cfg: FeatureConfig,
                         sample: int = 600) -> dict:
    """Compare the as-of feature against the same feature computed over everything.

    The leaky version is not a different feature. It is the same definition
    with the date filter left off, which is what makes it so easy to ship.
    """
    import numpy as np
    honest = replace(cfg, ignore_as_of=False)
    leaky = replace(cfg, ignore_as_of=True)
    subset = sorted(searches, key=lambda s: s.searched_at)[:sample]
    ids = {s.search_id for s in subset}
    opts = [o for o in options if o.search_id in ids]

    idx = FEATURE_NAMES.index("price_over_route_median_7d")
    a, _, meta, hist = build_matrix(subset, opts, honest)
    b, _, _, _ = build_matrix(subset, opts, leaky, history=hist)
    diff = np.abs(a[:, idx] - b[:, idx])
    known = a[:, FEATURE_NAMES.index("route_median_price_known")]
    return {
        "rows": int(a.shape[0]),
        "feature": "price_over_route_median_7d",
        "rows_that_differ": int((diff > 1e-9).sum()),
        "share_that_differ": round(float((diff > 1e-9).mean()), 4),
        "mean_absolute_difference": round(float(diff.mean()), 5),
        "max_absolute_difference": round(float(diff.max()), 5),
        "rows_with_no_history_yet": int((known == 0).sum()),
        "note": ("Rows early in the period have no trailing window, so the "
                 "honest feature reports 'not known' while the leaky one "
                 "quietly supplies a median computed from days that had not "
                 "happened. Those rows are the ones the leak helps most, which "
                 "is why an offline metric rises when it is introduced."),
    }
