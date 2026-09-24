"""
service.py — the model behind an endpoint.

The half of the job that a notebook never reaches. A request arrives with a
search and its journey options; the response is the same options in a different
order, with scores, inside a latency budget.

Three decisions worth stating, because each one is a failure this avoids.

**The features are computed by `features.compute`, the same function the
training job called.** Not a reimplementation, not a port, not a copy kept in
sync by good intentions. If the serving path needs a feature the training path
does not have, that is a change to one function and both callers get it.

**The configuration is loaded from the model artefact and never constructed
here.** The fill value for a missing seat count is a statistic of the training
set. A serving process that recomputes it from its own traffic drifts away from
the value the model was fitted against, silently, one request at a time.

**The process refuses to start if the model's feature contract does not match
this code.** A model trained before a feature was inserted will produce scores
against the wrong columns and raise nothing at all. `load_artefact` compares
the recorded names against the current ones and raises, which turns a wrong
answer into a failed deploy.

Run it:

    RAILRANK_MODEL=artefacts/current uvicorn railrank.service:app --port 8080
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import features as F
from . import registry as REG

MODEL_DIR = os.environ.get("RAILRANK_MODEL", "artefacts/current")
MAX_OPTIONS = 60

_state: dict = {"model": None, "loaded_at": None, "requests": 0,
                "latencies_ms": []}


# ── request and response ────────────────────────────────────────────────────

class OptionIn(BaseModel):
    option_id: str
    price_pence: int = Field(ge=0)
    duration_minutes: int = Field(gt=0)
    changes: int = Field(ge=0, le=9)
    departure_hour: int = Field(ge=0, le=23)
    departure_minute: int = Field(ge=0, le=59)
    operator: str
    ticket_type: str
    seats_remaining: int | None = None
    carbon_grams: int = Field(ge=0)
    load_factor_at_search: float = Field(ge=0.0, le=1.0)


class RankRequest(BaseModel):
    search_id: str
    searched_at: str
    origin: str
    destination: str
    travel_date: str
    departure_target_hour: int = Field(ge=0, le=23)
    passenger_type: str = "leisure"
    advance_days: int = Field(ge=0, le=365)
    options: list[OptionIn]


class RankedOption(BaseModel):
    option_id: str
    rank: int
    score: float


class RankResponse(BaseModel):
    search_id: str
    model_version: str
    ranked: list[RankedOption]
    latency_ms: float


# ── lifecycle ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    path = Path(MODEL_DIR)
    if path.exists():
        # Any mismatch raises here, before the process accepts traffic.
        _state["model"] = REG.load_artefact(path)
        _state["loaded_at"] = time.time()
    yield
    _state["model"] = None


app = FastAPI(title="railrank", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    m: REG.LoadedModel | None = _state["model"]
    lat = sorted(_state["latencies_ms"])
    return {
        "status": "ok" if m else "no model loaded",
        "model_dir": MODEL_DIR,
        "model_created_at": m.card["created_at"] if m else None,
        "feature_count": len(m.feature_names) if m else 0,
        "feature_config_fingerprint": m.card["feature_config_fingerprint"] if m else None,
        "requests_served": _state["requests"],
        "p50_ms": round(lat[len(lat) // 2], 2) if lat else None,
        "p99_ms": round(lat[min(len(lat) - 1, int(0.99 * len(lat)))], 2) if lat else None,
    }


@app.post("/rank", response_model=RankResponse)
def rank(req: RankRequest) -> RankResponse:
    model: REG.LoadedModel | None = _state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="no model loaded")
    if not req.options:
        raise HTTPException(status_code=422, detail="no options to rank")
    if len(req.options) > MAX_OPTIONS:
        # A bound on the work one request can cause. Without it a caller with a
        # bug decides this service's p99.
        raise HTTPException(
            status_code=413,
            detail=f"{len(req.options)} options exceeds the {MAX_OPTIONS} limit")

    t0 = time.perf_counter()
    search = F.SearchContext(
        search_id=req.search_id, searched_at=req.searched_at, origin=req.origin,
        destination=req.destination, travel_date=req.travel_date,
        departure_target_hour=req.departure_target_hour,
        passenger_type=req.passenger_type, advance_days=req.advance_days)
    options = [F.OptionContext(
        option_id=o.option_id, price_pence=o.price_pence,
        duration_minutes=o.duration_minutes, changes=o.changes,
        departure_hour=o.departure_hour, departure_minute=o.departure_minute,
        operator=o.operator, ticket_type=o.ticket_type,
        seats_remaining=o.seats_remaining, carbon_grams=o.carbon_grams,
        load_factor_at_search=o.load_factor_at_search,
        # Never populated at serving time. It is the leak, and the only way to
        # be certain it cannot reach the model is for the request schema to
        # have no field for it.
        final_load_factor=None) for o in req.options]

    agg = F.SearchAggregates.of(options)
    rows = [F.vector(F.compute(search, o, agg, model.history, model.config))
            for o in options]
    scores = model.score(rows)

    order = sorted(range(len(options)), key=lambda i: -scores[i])
    elapsed = (time.perf_counter() - t0) * 1000
    _state["requests"] += 1
    _state["latencies_ms"].append(elapsed)
    if len(_state["latencies_ms"]) > 5000:
        del _state["latencies_ms"][:1000]

    return RankResponse(
        search_id=req.search_id,
        model_version=model.card["created_at"],
        ranked=[RankedOption(option_id=options[i].option_id, rank=r,
                             score=round(scores[i], 6))
                for r, i in enumerate(order, start=1)],
        latency_ms=round(elapsed, 3))


def request_from(search, options) -> dict:
    """Build a request body from generator records, for tests and load runs."""
    return {
        "search_id": search.search_id, "searched_at": search.searched_at,
        "origin": search.origin, "destination": search.destination,
        "travel_date": search.travel_date,
        "departure_target_hour": search.departure_target_hour,
        "passenger_type": search.passenger_type,
        "advance_days": search.advance_days,
        "options": [{
            "option_id": o.option_id, "price_pence": o.price_pence,
            "duration_minutes": o.duration_minutes, "changes": o.changes,
            "departure_hour": o.departure_hour,
            "departure_minute": o.departure_minute, "operator": o.operator,
            "ticket_type": o.ticket_type, "seats_remaining": o.seats_remaining,
            "carbon_grams": o.carbon_grams,
            "load_factor_at_search": o.load_factor_at_search,
        } for o in options],
    }
