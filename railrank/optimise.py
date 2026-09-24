"""
optimise.py — split ticketing, which is a shortest path wearing a fare table.

Two tickets for the two halves of a journey are often cheaper than one ticket
for the whole of it, and it is legal as long as the train calls at the station
where the tickets change. That is not a loophole, it is a consequence of how
fares are set: a fare is a price for a market, not a price for a distance, so
the fare from A to C is under no obligation to be less than the fare from A to
B plus the fare from B to C.

Which makes it an optimisation problem with a real objective and a real
constraint:

    minimise   the sum of the fares bought
    subject to the tickets covering the journey end to end, consecutively,
               and every split point being a station the chosen service calls at

Written as a graph, the calling points are nodes, every purchasable fare is a
directed edge, and the cheapest combination is the shortest path from origin to
destination. The whole thing is a dynamic program over a topologically ordered
DAG: `O(n^2)` edges, `O(n^2)` work, and the reconstruction falls out of the
predecessor array.

The reason to write the brute force as well is that a solver returning a
plausible number is indistinguishable from a solver returning the right one.
`brute_force_split` enumerates every subset of intermediate calling points,
which is `2^(n-2)` and therefore useless above about twenty stations and exact
below it. `evaluate.py` runs both on every instance it can afford and asserts
they agree to the penny.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass

# ── the fare scale ──────────────────────────────────────────────────────────
# GB fares taper: the per-mile rate falls as distance rises, which on its own
# would make splitting always lose. What makes splitting win is that each
# station pair is also a market, and markets are priced against their own
# competition -- a corridor with a parallel operator or a strong coach service
# is cheaper per mile than the taper alone would suggest. Those two forces
# pulling against each other are the whole of the problem.

FIXED_PENCE = 380           # the part of any fare that is not distance
BASE_RATE = 2.0             # pence per mile before the taper
TAPER = 0.34                # rate multiplier falls as distance ** -TAPER
MARKET_SIGMA = 0.22         # how far a pair's price strays from the scale
COMPETITIVE_SHARE = 0.22    # share of pairs priced against competition
COMPETITIVE_DISCOUNT = 0.72
# No market is priced below this fraction of the distance scale. Without a
# floor the lognormal tail and the competitive discount compound, and the first
# version of this table priced Reading to Penzance -- 290 miles -- at 13 pounds,
# which made the optimiser look brilliant by making the fares absurd. A
# generator that produces impossible inputs cannot validate anything.
MARKET_FLOOR = 0.55


def _rate(miles: float) -> float:
    """Pence per mile at this distance. Falls with distance, as real fares do."""
    return BASE_RATE * 100 * max(miles, 1.0) ** -TAPER


@dataclass(frozen=True)
class Leg:
    """One ticket: from calling-point index `i` to index `j`, at `pence`."""
    i: int
    j: int
    origin: str
    destination: str
    pence: int


def fare_table(calling: list[tuple[str, int]], seed: int = 7) -> dict[tuple[int, int], int]:
    """Every purchasable fare between calling points, as a pence lookup.

    `calling` is the route's stations with their route mileage, in order. The
    returned keys are index pairs rather than names, because the optimiser
    works on positions along one journey and names would only be translated
    back at the end.
    """
    rng = random.Random(seed)
    n = len(calling)
    fares: dict[tuple[int, int], int] = {}
    for i in range(n):
        for j in range(i + 1, n):
            miles = calling[j][1] - calling[i][1]
            if miles <= 0:
                continue
            scale = FIXED_PENCE + _rate(miles) * miles
            factor = math.exp(rng.gauss(0.0, MARKET_SIGMA))
            if rng.random() < COMPETITIVE_SHARE:
                factor *= COMPETITIVE_DISCOUNT
            price = scale * max(factor, MARKET_FLOOR)
            # Fares are published in whole pounds far more often than not.
            fares[(i, j)] = int(round(price / 100.0) * 100)
    return fares


# ── the solver ──────────────────────────────────────────────────────────────

def cheapest_split(fares: dict[tuple[int, int], int], calling: list[tuple[str, int]],
                   calls_at: set[int] | None = None,
                   start: int = 0, end: int | None = None) -> tuple[int, list[Leg]]:
    """Cheapest legal combination of tickets from `start` to `end`.

    `calls_at` is the set of calling-point indices the chosen service actually
    stops at. A fast service that runs past a station cannot be split there,
    however cheap the fares would be, so the constraint is not cosmetic: the
    same fare table gives different answers for different services, which is
    why a real implementation has to take the service as an input.
    """
    end = len(calling) - 1 if end is None else end
    if calls_at is not None and not {start, end} <= calls_at:
        # You cannot board where the train does not stop. An earlier version
        # unioned the endpoints into the calling set, which quietly answered a
        # different question -- the cheapest fare for a journey the passenger
        # could not make.
        raise ValueError(
            f"the service does not call at "
            f"{calling[start][0] if start not in calls_at else calling[end][0]}")
    stops = sorted(calls_at if calls_at is not None
                   else set(range(len(calling))))
    stops = [s for s in stops if start <= s <= end]
    if not stops or stops[0] != start or stops[-1] != end:
        raise ValueError("the service does not call at both ends of the journey")

    INF = float("inf")
    best: dict[int, float] = {s: INF for s in stops}
    prev: dict[int, int | None] = {s: None for s in stops}
    best[start] = 0.0

    # Topological order is just increasing position along the route.
    for idx, a in enumerate(stops):
        if best[a] == INF:
            continue
        for b in stops[idx + 1:]:
            fare = fares.get((a, b))
            if fare is None:
                continue
            if best[a] + fare < best[b]:
                best[b] = best[a] + fare
                prev[b] = a

    if best[end] == INF:
        raise ValueError("no combination of fares covers this journey")

    legs: list[Leg] = []
    node = end
    while prev[node] is not None:
        a = prev[node]
        legs.append(Leg(a, node, calling[a][0], calling[node][0], fares[(a, node)]))
        node = a
    legs.reverse()
    return int(best[end]), legs


def brute_force_split(fares: dict[tuple[int, int], int],
                      calling: list[tuple[str, int]],
                      calls_at: set[int] | None = None,
                      start: int = 0, end: int | None = None,
                      max_intermediate: int = 16) -> tuple[int, list[Leg]]:
    """The same answer by enumerating every subset of split points.

    Exponential and therefore not a solver -- it is the oracle the solver is
    checked against. It refuses rather than silently taking minutes when the
    instance is too large, because a verification that quietly stops verifying
    is worse than none.
    """
    end = len(calling) - 1 if end is None else end
    if calls_at is not None and not {start, end} <= calls_at:
        raise ValueError("the service does not call at both ends of the journey")
    stops = sorted(calls_at if calls_at is not None
                   else set(range(len(calling))))
    stops = [s for s in stops if start <= s <= end]
    middle = [s for s in stops if start < s < end]
    if len(middle) > max_intermediate:
        raise ValueError(
            f"{len(middle)} intermediate calls is 2**{len(middle)} combinations; "
            f"raise max_intermediate deliberately or use cheapest_split")

    best_total, best_legs = None, None
    for r in range(len(middle) + 1):
        for chosen in itertools.combinations(middle, r):
            points = [start, *chosen, end]
            total, legs = 0, []
            ok = True
            for a, b in zip(points, points[1:]):
                fare = fares.get((a, b))
                if fare is None:
                    ok = False
                    break
                total += fare
                legs.append(Leg(a, b, calling[a][0], calling[b][0], fare))
            if ok and (best_total is None or total < best_total):
                best_total, best_legs = total, legs
    if best_total is None:
        raise ValueError("no combination of fares covers this journey")
    return best_total, best_legs


def through_fare(fares: dict[tuple[int, int], int], start: int = 0,
                 end: int | None = None, n: int | None = None) -> int | None:
    """What a passenger pays if nobody tells them about splitting."""
    end = (n - 1) if end is None and n is not None else end
    return fares.get((start, end))


def saving(fares: dict[tuple[int, int], int], calling: list[tuple[str, int]],
           calls_at: set[int] | None = None, start: int = 0,
           end: int | None = None) -> dict:
    """The through fare, the cheapest split, and the difference."""
    end = len(calling) - 1 if end is None else end
    through = fares.get((start, end))
    total, legs = cheapest_split(fares, calling, calls_at, start, end)
    out = {
        "origin": calling[start][0], "destination": calling[end][0],
        "through_pence": through,
        "split_pence": total,
        "n_tickets": len(legs),
        "legs": [{"from": leg.origin, "to": leg.destination, "pence": leg.pence}
                 for leg in legs],
    }
    if through is not None:
        out["saving_pence"] = through - total
        out["saving_pct"] = round(100 * (through - total) / through, 2)
    return out


def summarise(instances: list[dict]) -> dict:
    """Aggregate a batch of solved journeys into something quotable."""
    priced = [i for i in instances if i.get("through_pence")]
    saved = [i for i in priced if i["saving_pence"] > 0]
    savings = sorted(i["saving_pence"] for i in saved)
    pcts = sorted(i["saving_pct"] for i in saved)
    return {
        "journeys": len(priced),
        "journeys_where_splitting_wins": len(saved),
        "share_where_splitting_wins": round(len(saved) / max(len(priced), 1), 4),
        "median_saving_pence": savings[len(savings) // 2] if savings else 0,
        "median_saving_pct": pcts[len(pcts) // 2] if pcts else 0.0,
        "max_saving_pence": savings[-1] if savings else 0,
        "mean_tickets_when_split": round(
            sum(i["n_tickets"] for i in saved) / max(len(saved), 1), 2),
    }
