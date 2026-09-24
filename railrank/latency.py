"""
latency.py — how long the model takes to answer, and whether that is allowed.

A better model that cannot answer in time is not a better model. Ranking sits
on the search path, in front of a person waiting for a page, so the budget is
a product decision before it is an engineering one and it belongs in the
promotion gate next to the accuracy metric.

Two measurements, because they answer different questions.

`measure_inprocess` times the work the model itself causes: build the feature
rows, call the booster, sort. That is the number a model change moves, and the
one the gate should read, because it is not contaminated by how fast the
machine running the test happens to be at HTTP.

`measure_http` times a real request against a running container. It is larger
and noisier and it is the only number that answers "what would a caller see".
Both are reported. Quoting only the first is how a service ships with a 6 ms
model and a 300 ms p99.

Percentiles, not means. A mean latency hides exactly the requests people
complain about, and the tail here has a real cause: searches return between
eight and fourteen options and the work is linear in that.
"""

from __future__ import annotations

import statistics
import time

from . import features as F


def _percentiles(samples_ms: list[float]) -> dict:
    s = sorted(samples_ms)
    n = len(s)

    def p(q: float) -> float:
        return s[min(n - 1, int(q * n))]

    return {
        "n": n,
        "mean_ms": round(statistics.fmean(s), 3),
        "p50_ms": round(p(0.50), 3),
        "p90_ms": round(p(0.90), 3),
        "p95_ms": round(p(0.95), 3),
        "p99_ms": round(p(0.99), 3),
        "max_ms": round(s[-1], 3),
    }


def measure_inprocess(model, searches, options_by_search, n: int = 600,
                      warmup: int = 40) -> dict:
    """Time the ranking path itself, one search at a time.

    One search per call on purpose. Batching every search into a single matrix
    would measure throughput, and a traveller does not wait for a batch.
    """
    picked = [s for s in searches if options_by_search.get(s.search_id)][:n + warmup]
    samples: list[float] = []
    sizes: list[int] = []

    for i, s in enumerate(picked):
        opts = options_by_search[s.search_id]
        ctx = F.SearchContext(
            search_id=s.search_id, searched_at=s.searched_at, origin=s.origin,
            destination=s.destination, travel_date=s.travel_date,
            departure_target_hour=s.departure_target_hour,
            passenger_type=s.passenger_type, advance_days=s.advance_days)
        oc = [F.OptionContext(
            option_id=o.option_id, price_pence=o.price_pence,
            duration_minutes=o.duration_minutes, changes=o.changes,
            departure_hour=o.departure_hour, departure_minute=o.departure_minute,
            operator=o.operator, ticket_type=o.ticket_type,
            seats_remaining=o.seats_remaining, carbon_grams=o.carbon_grams,
            load_factor_at_search=o.load_factor_at_search,
            final_load_factor=None) for o in opts]

        t0 = time.perf_counter()
        agg = F.SearchAggregates.of(oc)
        rows = [F.vector(F.compute(ctx, c, agg, model.history, model.config))
                for c in oc]
        scores = model.score(rows)
        sorted(range(len(rows)), key=lambda j: -scores[j])
        elapsed = (time.perf_counter() - t0) * 1000

        if i >= warmup:          # the first calls pay for lazy imports and caches
            samples.append(elapsed)
            sizes.append(len(oc))

    out = _percentiles(samples)
    out["mean_options_per_search"] = round(statistics.fmean(sizes), 2)
    out["measurement"] = "in-process: features, predict and sort, one search per call"
    return out


def measure_http(url: str, payloads: list[dict], n: int = 300,
                 warmup: int = 20, timeout: float = 10.0) -> dict:
    """Time real requests against a running service."""
    import httpx

    samples: list[float] = []
    statuses: dict[int, int] = {}
    with httpx.Client(timeout=timeout) as client:
        for i in range(n + warmup):
            body = payloads[i % len(payloads)]
            t0 = time.perf_counter()
            r = client.post(url, json=body)
            elapsed = (time.perf_counter() - t0) * 1000
            statuses[r.status_code] = statuses.get(r.status_code, 0) + 1
            if i >= warmup and r.status_code == 200:
                samples.append(elapsed)
    if not samples:
        return {"n": 0, "statuses": statuses,
                "measurement": "http: no successful responses"}
    out = _percentiles(samples)
    out["statuses"] = statuses
    out["measurement"] = f"http: POST {url}, one search per request"
    return out


def budget_report(measured: dict, budget_p99_ms: float) -> dict:
    """State the budget, the measurement, and the headroom in one place."""
    p99 = measured.get("p99_ms")
    return {
        "budget_p99_ms": budget_p99_ms,
        "measured_p99_ms": p99,
        "within_budget": bool(p99 is not None and p99 <= budget_p99_ms),
        "headroom_ms": round(budget_p99_ms - p99, 3) if p99 is not None else None,
        "headroom_pct": (round(100 * (budget_p99_ms - p99) / budget_p99_ms, 1)
                         if p99 is not None else None),
        "measurement": measured.get("measurement"),
    }
