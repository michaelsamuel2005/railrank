"""
journeys.py — rail journey searches, and the utility function that decides
which result was actually the best one.

A rail search returns a list of journey options: a price, a duration, some
number of changes, a departure time, an operator and a ticket type. Somebody
picks one. A ranker's job is to put the one they would pick near the top.

The problem with learning that from a click log is that **a click is not a
judgement about the journey**. It is examination multiplied by preference: the
traveller looks at position one nearly always and position twelve nearly never,
so a click says as much about where the ranker put an option as about whether
the option was any good. Train on raw clicks and the model learns to reproduce
the ranker that produced the log.

So this generator separates the two and writes the answer down.

  UTILITY          every option has a latent utility under a preference model
                   that depends on the traveller — a business traveller prices
                   time differently from a leisure traveller. `oracle_order`
                   returns the ranking that utility implies. No real log has
                   this column; it is here so a ranker can be scored on how
                   close it gets to the truth rather than on how well it
                   reproduces the log.

  LOGGING POLICY   the production ranker that generated the log is deliberately
                   mediocre — it sorts by price with a little noise. That is
                   what leaves room to measure an improvement, and it is also
                   realistic: the first ranker any product ships is a sort.

  POSITION BIAS    examination falls off as position ** -eta, and a click
                   requires both examination and preference.

  EXPLORATION      a slice of searches is shown in random order. With
                   preference independent of position there, the click-rate
                   curve identifies the examination probabilities, which is how
                   propensities get estimated without guessing.

  DRIFT            route prices trend over the period, so a feature computed
                   over the whole dataset is not the feature that existed at
                   the time of the search. That is what makes point-in-time
                   correctness in `features.py` a real constraint rather than
                   a talking point.

Ground truth for all of it is returned by `planted()`.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

SEED = 20260924

# ── the network ─────────────────────────────────────────────────────────────
# Real GB stations on real long-distance routes, with the intermediate calling
# points that make split ticketing possible. Distances are approximate route
# miles; they set the fare scale in `optimise.py`.

ROUTES: dict[tuple[str, str], list[tuple[str, int]]] = {
    ("London Kings Cross", "Edinburgh"): [
        ("London Kings Cross", 0), ("Stevenage", 27), ("Peterborough", 76),
        ("Grantham", 105), ("Newark North Gate", 120), ("Doncaster", 156),
        ("York", 188), ("Darlington", 232), ("Durham", 254),
        ("Newcastle", 268), ("Berwick-upon-Tweed", 335), ("Edinburgh", 393),
    ],
    ("London Euston", "Glasgow Central"): [
        ("London Euston", 0), ("Milton Keynes Central", 50), ("Rugby", 82),
        ("Crewe", 158), ("Warrington Bank Quay", 182), ("Wigan North Western", 194),
        ("Preston", 209), ("Lancaster", 230), ("Oxenholme Lake District", 249),
        ("Penrith", 281), ("Carlisle", 299), ("Glasgow Central", 401),
    ],
    ("London Paddington", "Penzance"): [
        ("London Paddington", 0), ("Reading", 36), ("Swindon", 77),
        ("Bristol Parkway", 112), ("Taunton", 163), ("Exeter St Davids", 174),
        ("Plymouth", 226), ("Truro", 299), ("Penzance", 326),
    ],
    ("Manchester Piccadilly", "London Euston"): [
        ("Manchester Piccadilly", 0), ("Stockport", 7), ("Macclesfield", 18),
        ("Stoke-on-Trent", 38), ("Milton Keynes Central", 133), ("London Euston", 184),
    ],
    ("Bristol Temple Meads", "Birmingham New Street"): [
        ("Bristol Temple Meads", 0), ("Bristol Parkway", 6), ("Gloucester", 38),
        ("Cheltenham Spa", 45), ("Worcester Shrub Hill", 72),
        ("Birmingham New Street", 94),
    ],
}
ROUTE_KEYS = list(ROUTES)

OPERATORS = ["LNER", "Avanti West Coast", "GWR", "CrossCountry",
             "TransPennine Express", "Northern"]
# Operator quality shifts preference a little, the way punctuality and seat
# comfort really do. Small, because it must not swamp price and time.
OPERATOR_BIAS = {"LNER": 0.12, "Avanti West Coast": 0.05, "GWR": 0.06,
                 "CrossCountry": -0.06, "TransPennine Express": -0.02,
                 "Northern": -0.14}

TICKET_TYPES = ["Advance", "Off-Peak", "Super Off-Peak", "Anytime"]
PASSENGER_TYPES = ["leisure", "business", "commuter"]
PASSENGER_WEIGHTS = [0.55, 0.22, 0.23]

# Preference weights per traveller type. A business traveller will pay to save
# an hour; a leisure traveller will spend an hour to save fifteen pounds. A
# ranker that ignores this ranks the same list for both, which is the gap
# personalisation exists to close.
# Units: price in tens of pence (price_pence / 1000), duration in hours. The
# weights are set so that the sentence above is true of the numbers. For a
# leisure traveller twenty pounds is worth about half an hour; for a business
# traveller half an hour is worth about eighty. `test_journeys.py` asserts the
# consequence -- that the oracle-best option is cheaper for leisure and faster
# for business -- because a comment claiming this would otherwise be unchecked.
PREFERENCE = {
    #                price     duration   changes   departure   direct
    "leisure":  dict(w_price=-0.110, w_duration=-0.34, w_changes=-0.22,
                     w_departure=-0.10, w_direct=0.18),
    "business": dict(w_price=-0.020, w_duration=-1.05, w_changes=-0.60,
                     w_departure=-0.38, w_direct=0.45),
    "commuter": dict(w_price=-0.060, w_duration=-0.58, w_changes=-0.42,
                     w_departure=-0.50, w_direct=0.28),
}

EXAMINATION_ETA = 0.80          # e(position) = position ** -eta
EXPLORATION_FRACTION = 0.14     # searches shown in a randomised order
BASE_CLICK_SCALE = 0.42         # converts utility into a click probability
BOOKING_GIVEN_CLICK = 0.34
MAX_CLICKS = 4                  # a session compares a few journeys, not all of them
STOP_AFTER_CLICK = 0.10         # chance of giving up after an unconverted click
# Once a traveller has opened one journey they are comparing rather than
# skimming, so the next few results get examined far more than their raw
# position would suggest. This is a real effect in search logs and it is also
# what puts clicks at low positions into the data at all -- without it the
# propensity correction has almost nothing to correct.
ENGAGED_EXAMINATION_FLOOR = 0.55
LOGGING_POLICY_NOISE = 0.55     # how scrambled the production ranker is

PERIOD_DAYS = 45
PRICE_TREND_PER_DAY = 0.0042    # fares drift up over the period; ~20% overall

# A planted upstream change, of the kind a model monitor exists to catch. From
# this day, one operator's client starts reporting an unknown seat count as 0
# instead of omitting the field. Nothing is missing, no row count moves, every
# value is a legal integer and the schema is unchanged -- so freshness,
# volume and null-rate checks all stay green while a feature the model reads
# shifts underneath it.
SEATS_REPORTING_CHANGE = dict(operators=("CrossCountry", "Northern"),
                              from_day=30, reports=0)


# ── records ─────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class Search:
    search_id: str
    user_id: str
    searched_at: str            # ISO timestamp — the as-of moment for features
    origin: str
    destination: str
    travel_date: str
    departure_target_hour: int  # when the traveller wants to leave
    passenger_type: str
    advance_days: int
    exploration: int            # 1 if this search was shown in random order


@dataclass(slots=True)
class Option:
    option_id: str
    search_id: str
    position: int               # where the logging policy put it, 1-based
    price_pence: int
    duration_minutes: int
    changes: int
    departure_hour: int
    departure_minute: int
    operator: str
    ticket_type: str
    seats_remaining: int | None  # None when the operator does not report it
    carbon_grams: int
    # Two load factors, and the difference between them is the whole of the
    # point-in-time problem. `load_factor_at_search` is what the reservation
    # system knew when the search happened, and is a legitimate feature.
    # `final_load_factor` is what the train left with -- it is the same column
    # in the same table, read later. Popular journeys fill up, so it correlates
    # with the label, and joining it without an as-of filter is a leak that
    # looks like a strong feature.
    load_factor_at_search: float
    final_load_factor: float
    clicked: int                # the OBSERVED label a real log has
    booked: int
    true_utility: float         # the ORACLE, for scoring only


def _departure_penalty(target_hour: int, hour: int, minute: int) -> float:
    """Hours away from when the traveller actually wanted to leave."""
    return abs((hour + minute / 60) - target_hour)


def utility(pref: dict, price_pence: int, duration_minutes: int, changes: int,
            departure_penalty: float, operator: str) -> float:
    """The latent preference for one option. This is the answer key."""
    return (pref["w_price"] * (price_pence / 1000)
            + pref["w_duration"] * (duration_minutes / 60)
            + pref["w_changes"] * changes
            + pref["w_departure"] * departure_penalty
            + pref["w_direct"] * (1 if changes == 0 else 0)
            + OPERATOR_BIAS[operator])


def oracle_order(options: list[Option]) -> list[str]:
    """The ranking the utility model implies — best first."""
    return [o.option_id for o in sorted(options, key=lambda o: -o.true_utility)]


def _build_options(rng: random.Random, search: Search, route_miles: int,
                   day_index: int, n: int) -> list[Option]:
    pref = PREFERENCE[search.passenger_type]
    trend = 1.0 + PRICE_TREND_PER_DAY * day_index
    raw: list[Option] = []

    for i in range(n):
        changes = rng.choices([0, 1, 2], [0.42, 0.40, 0.18])[0]
        # Fewer changes costs more and takes less time, which is what creates
        # a genuine trade-off rather than a dominated set of options.
        # Roughly the GB fare scale: a per-mile rate plus a small fixed
        # component, before the ticket-type multiplier. London to Edinburgh
        # lands near 70 pounds Advance and 165 Anytime, which is about right.
        base = route_miles * rng.uniform(0.18, 0.34) + 12
        ticket = rng.choices(TICKET_TYPES, [0.38, 0.30, 0.17, 0.15])[0]
        ticket_mult = {"Advance": 0.62, "Off-Peak": 0.88,
                       "Super Off-Peak": 0.74, "Anytime": 1.45}[ticket]
        advance_mult = 1.0 + max(0.0, (14 - search.advance_days)) * 0.028
        price = int(base * ticket_mult * advance_mult * trend
                    * (1.0 - 0.06 * changes) * rng.uniform(0.92, 1.10) * 100)

        speed = rng.uniform(52, 78) - 6 * changes
        duration = int(route_miles / speed * 60 + changes * rng.uniform(11, 34)
                       + rng.uniform(-6, 9))
        hour = max(5, min(22, int(rng.gauss(search.departure_target_hour, 2.4))))
        minute = rng.choice([0, 3, 7, 12, 18, 22, 27, 33, 39, 44, 48, 52, 57])
        operator = rng.choice(OPERATORS)
        # Not every operator reports seat availability. A missing value here is
        # real and the feature layer has to decide what to do with it.
        seats = None if rng.random() < 0.18 else rng.randint(0, 120)
        if (operator in SEATS_REPORTING_CHANGE["operators"]
                and day_index >= SEATS_REPORTING_CHANGE["from_day"]):
            # The release: the client starts sending a constant where an
            # availability count used to be. Legal integer, no nulls, no
            # schema change, no row-count change.
            seats = SEATS_REPORTING_CHANGE["reports"]
        carbon = int(route_miles * rng.uniform(38, 62))   # ~35 g per pax-km

        u = utility(pref, price, duration, changes,
                    _departure_penalty(search.departure_target_hour, hour, minute),
                    operator)
        # Load at search time is mostly a function of how far ahead the
        # search is; the load the train finally leaves with also reflects how
        # appealing the journey was, which is what makes it a leak.
        lf_now = min(0.97, max(0.03, 0.16 + (56 - search.advance_days) / 145
                               + rng.gauss(0, 0.10)))
        raw.append(Option(
            option_id=f"{search.search_id}-o{i:02d}", search_id=search.search_id,
            position=0, price_pence=price, duration_minutes=duration,
            changes=changes, departure_hour=hour, departure_minute=minute,
            operator=operator, ticket_type=ticket, seats_remaining=seats,
            carbon_grams=carbon,
            load_factor_at_search=round(lf_now, 4),
            final_load_factor=0.0,          # filled in below, after appeal is known
            clicked=0, booked=0, true_utility=u))

    # ── the logging policy ────────────────────────────────────────────────
    # The production ranker sorts by price with noise. It is not stupid — price
    # correlates with utility — but it ignores duration, changes and departure
    # time entirely, so there is real room above it. That is the baseline the
    # learned ranker has to beat, and beating a straw man would prove nothing.
    if search.exploration:
        order = list(range(len(raw)))
        rng.shuffle(order)
    else:
        order = sorted(range(len(raw)),
                       key=lambda j: (raw[j].price_pence / 1000
                                      + rng.gauss(0, LOGGING_POLICY_NOISE)))
    ranked = [raw[j] for j in order]
    for pos, opt in enumerate(ranked, start=1):
        opt.position = pos

    # ── clicks: examination times preference ──────────────────────────────
    u_vals = [o.true_utility for o in ranked]
    lo, hi = min(u_vals), max(u_vals)
    span = (hi - lo) or 1.0
    for opt in ranked:
        appeal_all = (opt.true_utility - lo) / span
        opt.final_load_factor = round(
            min(0.995, max(opt.load_factor_at_search,
                           opt.load_factor_at_search
                           + 0.30 * appeal_all + rng.gauss(0, 0.05))), 4)
    # A traveller opens several journeys before choosing one, and that matters
    # more than it sounds. With exactly one click per search, an inverse
    # propensity correction is arithmetically incapable of changing a listwise
    # ranking objective: NDCG normalises gains within the query, so scaling the
    # single positive scales the ideal by the same factor and cancels. The
    # first version of this generator clicked once and the corrected model
    # scored identically to the uncorrected one, to four decimal places.
    n_clicks = 0
    for opt in ranked:
        appeal = BASE_CLICK_SCALE * ((opt.true_utility - lo) / span) ** 1.4
        p_exam = opt.position ** -EXAMINATION_ETA
        if n_clicks:
            p_exam = max(p_exam, ENGAGED_EXAMINATION_FLOOR)
        examined = rng.random() < p_exam
        if not (examined and rng.random() < appeal):
            continue
        opt.clicked = 1
        n_clicks += 1
        # Conversion depends on the journey too: a sold-out cheap advance
        # fare gets clicked and not booked.
        avail = 1.0 if opt.seats_remaining is None else min(
            1.0, 0.25 + opt.seats_remaining / 60)
        if rng.random() < BOOKING_GIVEN_CLICK * avail:
            opt.booked = 1
            # The seat is now sold, and the reservation system knows. This is
            # what makes `final_load_factor` a leak rather than a weak feature:
            # it carries the outcome directly.
            opt.final_load_factor = round(
                min(0.999, opt.final_load_factor + 0.22), 4)
            break                       # a booking ends the session
        opt.final_load_factor = round(
            min(0.999, opt.final_load_factor + 0.09), 4)
        if n_clicks >= MAX_CLICKS or rng.random() < STOP_AFTER_CLICK:
            break
    return ranked


def generate(n_searches: int = 24_000, seed: int = SEED,
             period_days: int = PERIOD_DAYS) -> tuple[list[Search], list[Option]]:
    rng = random.Random(seed)
    start = datetime(2026, 7, 1, 6, 0, 0)
    searches: list[Search] = []
    options: list[Option] = []

    for i in range(n_searches):
        day_index = int(i / n_searches * period_days)
        searched_at = start + timedelta(
            days=day_index, seconds=rng.randint(0, 57_600))
        key = rng.choice(ROUTE_KEYS)
        calling = ROUTES[key]
        origin, destination = calling[0][0], calling[-1][0]
        route_miles = calling[-1][1]
        passenger = rng.choices(PASSENGER_TYPES, PASSENGER_WEIGHTS)[0]
        advance = rng.choices([0, 1, 3, 7, 14, 28, 56],
                              [0.20, 0.14, 0.16, 0.18, 0.15, 0.11, 0.06])[0]
        target = rng.choices(list(range(6, 22)),
                             [3, 6, 9, 7, 5, 5, 5, 5, 5, 6, 7, 8, 7, 5, 3, 2])[0]

        search = Search(
            search_id=f"s{i:07d}", user_id=f"u{rng.randint(1, 90_000):06d}",
            searched_at=searched_at.isoformat(timespec="seconds"),
            origin=origin, destination=destination,
            travel_date=str((searched_at + timedelta(days=advance)).date()),
            departure_target_hour=target, passenger_type=passenger,
            advance_days=advance,
            exploration=1 if rng.random() < EXPLORATION_FRACTION else 0)
        searches.append(search)
        options += _build_options(rng, search, route_miles, day_index,
                                  rng.randint(8, 14))
    return searches, options


def planted() -> dict:
    """The answer key. Every claim in the scorecard is checked against this."""
    return {
        "seed": SEED,
        "preference_weights": PREFERENCE,
        "operator_bias": OPERATOR_BIAS,
        "examination_eta": EXAMINATION_ETA,
        "max_clicks_per_search": MAX_CLICKS,
        "engaged_examination_floor": ENGAGED_EXAMINATION_FLOOR,
        "exploration_fraction": EXPLORATION_FRACTION,
        "logging_policy": "sort by price ascending with gaussian noise "
                          f"(sd {LOGGING_POLICY_NOISE}); ignores duration, "
                          "changes and departure time entirely",
        "price_trend_per_day": PRICE_TREND_PER_DAY,
        "seats_reporting_change": SEATS_REPORTING_CHANGE,
        "leak": ("final_load_factor is known only after the train has gone and "
                 "rises with how appealing the journey was, so it encodes the "
                 "label. load_factor_at_search is the same column read as of "
                 "the search, and is legitimate."),
        "period_days": PERIOD_DAYS,
        "routes": {f"{a} - {b}": len(v) for (a, b), v in ROUTES.items()},
        "notes": (
            "true_utility is an oracle column. No click log has it. It exists "
            "so a ranker can be scored against what the traveller would "
            "actually have preferred, rather than against the log produced by "
            "the ranker being replaced."),
    }


def to_rows(searches: list[Search], options: list[Option]) -> list[dict]:
    """Join searches to options, the way a training table is assembled."""
    by_id = {s.search_id: s for s in searches}
    rows = []
    for o in options:
        row = asdict(by_id[o.search_id])
        row.update(asdict(o))
        rows.append(row)
    return rows
