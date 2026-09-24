"""The artefact, the gate, the endpoint and the monitors.

Several of these encode failures that shipped and were caught by the
scorecard: a solver that assumed the train stopped where you boarded, a
promotion gate with no refusal path, a drift statistic that re-binned on the
drifted sample.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from railrank import features as F
from railrank import journeys as J
from railrank import latency as L
from railrank import monitoring as M
from railrank import ranker as R
from railrank import registry as REG
from railrank import service as SV


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    searches, options = J.generate(n_searches=4000)
    cfg = F.FeatureConfig(seats_missing_fill=F.seats_missing_fill_from(options))
    X, g, meta, hist = F.build_matrix(searches, options, cfg)
    tr, trg, trm, te, teg, tem = R.temporal_split(g, meta, 0.7)
    prop = R.estimate_propensities(trm)
    model = R.train(X[tr], trg, trm, cfg, labels=R.ips_labels(trm, prop),
                    n_rounds=60)
    metrics = R.evaluate(model.score(X[te]), teg, tem, propensities=prop)
    path = tmp_path_factory.mktemp("artefacts") / "current"
    REG.save_artefact(path, model, prop, hist, metrics)
    return dict(searches=searches, options=options, cfg=cfg, X=X, g=g,
                meta=meta, hist=hist, prop=prop, model=model,
                metrics=metrics, path=path)


# ── the artefact ────────────────────────────────────────────────────────────

def test_the_artefact_is_self_contained(trained):
    files = {p.name for p in trained["path"].iterdir()}
    assert files == set(REG.ARTEFACT_FILES)


def test_loading_restores_the_configuration_rather_than_rebuilding_it(trained):
    m = REG.load_artefact(trained["path"])
    assert m.config == trained["cfg"]
    assert m.card["feature_config_fingerprint"] == trained["cfg"].fingerprint()


def test_loading_refuses_a_reordered_feature_contract(trained, tmp_path):
    import shutil
    dst = tmp_path / "copy"
    shutil.copytree(trained["path"], dst)
    card = json.loads((dst / "model_card.json").read_text())
    card["feature_names"] = list(reversed(card["feature_names"]))
    (dst / "model_card.json").write_text(json.dumps(card))
    with pytest.raises(ValueError, match="feature contract"):
        REG.load_artefact(dst)


def test_loading_refuses_a_model_missing_a_feature(trained, tmp_path):
    import shutil
    dst = tmp_path / "copy2"
    shutil.copytree(trained["path"], dst)
    card = json.loads((dst / "model_card.json").read_text())
    card["feature_names"] = card["feature_names"][:-1]
    (dst / "model_card.json").write_text(json.dumps(card))
    with pytest.raises(ValueError):
        REG.load_artefact(dst)


def test_the_route_history_survives_the_round_trip(trained):
    m = REG.load_artefact(trained["path"])
    route = next(iter(trained["hist"]._days))
    assert m.history._days[route] == trained["hist"]._days[route]
    assert m.history._median[route] == trained["hist"]._median[route]


# ── the gate ────────────────────────────────────────────────────────────────

CLEAN = {"skew_detected": False, "verdict": "no skew"}
DIRTY = {"skew_detected": True, "verdict": "advance_days differs"}


def test_the_gate_promotes_when_everything_holds():
    r = REG.promotion_gate({"dcg@5_ips": 1.3}, {"dcg@5_ips": 1.0}, CLEAN, 5.0)
    assert r.promoted
    assert r.verdict.startswith("PROMOTED")


def test_the_gate_refuses_on_skew():
    r = REG.promotion_gate({"dcg@5_ips": 1.3}, {"dcg@5_ips": 1.0}, DIRTY, 5.0)
    assert not r.promoted
    assert "no_training_serving_skew" in r.to_dict()["failed"]


def test_the_gate_refuses_on_latency():
    r = REG.promotion_gate({"dcg@5_ips": 1.3}, {"dcg@5_ips": 1.0}, CLEAN, 40.0,
                           budget_p99_ms=25.0)
    assert not r.promoted
    assert "latency_budget" in r.to_dict()["failed"]


def test_the_gate_refuses_a_model_that_is_no_better():
    r = REG.promotion_gate({"dcg@5_ips": 1.0}, {"dcg@5_ips": 1.0}, CLEAN, 5.0)
    assert not r.promoted
    assert "beats_incumbent" in r.to_dict()["failed"]


def test_a_tiny_gain_does_not_clear_the_bar():
    r = REG.promotion_gate({"dcg@5_ips": 1.0001}, {"dcg@5_ips": 1.0}, CLEAN, 5.0,
                           min_improvement=0.01)
    assert not r.promoted


def test_the_first_model_is_accepted_without_an_incumbent():
    r = REG.promotion_gate({"dcg@5_ips": 0.9}, None, CLEAN, 5.0)
    assert r.promoted


def test_the_gate_refuses_without_a_latency_measurement():
    r = REG.promotion_gate({"dcg@5_ips": 1.3}, {"dcg@5_ips": 1.0}, CLEAN, None)
    assert not r.promoted


def test_the_gate_reports_its_reasons_either_way():
    for inc in ({"dcg@5_ips": 1.0}, {"dcg@5_ips": 9.0}):
        r = REG.promotion_gate({"dcg@5_ips": 1.3}, inc, CLEAN, 5.0)
        assert len(r.checks) == 4
        assert all(c.detail for c in r.checks)


# ── the endpoint ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client(trained, monkeypatch_module=None):
    from fastapi.testclient import TestClient
    SV.MODEL_DIR = str(trained["path"])
    with TestClient(SV.app) as c:
        yield c


def _body(trained):
    by = {}
    for o in trained["options"]:
        by.setdefault(o.search_id, []).append(o)
    s = next(s for s in trained["searches"] if by.get(s.search_id))
    return SV.request_from(s, by[s.search_id])


def test_health_reports_a_loaded_model(client, trained):
    j = client.get("/health").json()
    assert j["status"] == "ok"
    assert j["feature_count"] == len(F.FEATURE_NAMES)
    assert j["feature_config_fingerprint"] == trained["cfg"].fingerprint()


def test_rank_returns_every_option_exactly_once(client, trained):
    body = _body(trained)
    j = client.post("/rank", json=body).json()
    assert len(j["ranked"]) == len(body["options"])
    assert ({r["option_id"] for r in j["ranked"]}
            == {o["option_id"] for o in body["options"]})


def test_rank_returns_them_in_descending_score_order(client, trained):
    j = client.post("/rank", json=_body(trained)).json()
    scores = [r["score"] for r in j["ranked"]]
    assert scores == sorted(scores, reverse=True)
    assert [r["rank"] for r in j["ranked"]] == list(range(1, len(scores) + 1))


def test_rank_is_deterministic(client, trained):
    body = _body(trained)
    a = client.post("/rank", json=body).json()["ranked"]
    b = client.post("/rank", json=body).json()["ranked"]
    assert [x["option_id"] for x in a] == [x["option_id"] for x in b]


def test_an_empty_option_list_is_rejected(client, trained):
    assert client.post("/rank", json={**_body(trained), "options": []}).status_code == 422


def test_an_oversized_request_is_rejected(client, trained):
    body = _body(trained)
    big = {**body, "options": body["options"] * 20}
    assert client.post("/rank", json=big).status_code == 413


def test_an_out_of_range_field_is_rejected(client, trained):
    assert client.post("/rank", json={**_body(trained),
                                      "departure_target_hour": 99}).status_code == 422


def test_the_request_schema_has_no_field_for_the_leak():
    """The only reliable way to keep a leaked column out of serving is for the
    wire format to have nowhere to put it."""
    assert "final_load_factor" not in SV.OptionIn.model_fields
    assert "load_factor_at_search" in SV.OptionIn.model_fields


def test_serving_matches_training_feature_for_feature(trained):
    """The whole point of one shared function."""
    m = REG.load_artefact(trained["path"])
    by = {}
    for o in trained["options"]:
        by.setdefault(o.search_id, []).append(o)
    s = next(s for s in trained["searches"] if by.get(s.search_id))
    opts = by[s.search_id]
    ctx = F.SearchContext(s.search_id, s.searched_at, s.origin, s.destination,
                          s.travel_date, s.departure_target_hour,
                          s.passenger_type, s.advance_days)
    oc = [F.OptionContext(o.option_id, o.price_pence, o.duration_minutes,
                          o.changes, o.departure_hour, o.departure_minute,
                          o.operator, o.ticket_type, o.seats_remaining,
                          o.carbon_grams, o.load_factor_at_search, None)
          for o in opts]
    agg = F.SearchAggregates.of(oc)
    served = [F.vector(F.compute(ctx, c, agg, m.history, m.config)) for c in oc]
    trainrows = [F.vector(F.compute(ctx, c, agg, trained["hist"], trained["cfg"]))
                 for c in oc]
    assert np.allclose(np.array(served), np.array(trainrows))


# ── latency ─────────────────────────────────────────────────────────────────

def test_latency_reports_ordered_percentiles(trained):
    m = REG.load_artefact(trained["path"])
    by = {}
    for o in trained["options"]:
        by.setdefault(o.search_id, []).append(o)
    r = L.measure_inprocess(m, trained["searches"][:200], by, n=60, warmup=10)
    assert r["n"] == 60
    assert r["p50_ms"] <= r["p90_ms"] <= r["p99_ms"] <= r["max_ms"]
    assert r["mean_options_per_search"] > 1


def test_the_budget_report_states_the_headroom():
    r = L.budget_report({"p99_ms": 5.0, "measurement": "x"}, 25.0)
    assert r["within_budget"] and r["headroom_ms"] == pytest.approx(20.0)
    assert L.budget_report({"p99_ms": 40.0}, 25.0)["within_budget"] is False


# ── monitoring ──────────────────────────────────────────────────────────────

def test_psi_is_zero_for_the_same_sample():
    rng = np.random.default_rng(0)
    x = rng.normal(size=4000)
    assert M.psi(x, x) == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_with_the_size_of_the_shift():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=6000)
    small = M.psi(ref, rng.normal(0.15, 1, 6000))
    large = M.psi(ref, rng.normal(1.50, 1, 6000))
    assert large > small
    assert large > M.PSI_SHIFTED


def test_psi_bins_on_the_reference_not_the_pooled_data():
    """Re-binning on the drifted sample absorbs the drift being measured."""
    rng = np.random.default_rng(2)
    ref = rng.normal(0, 1, 5000)
    cur = rng.normal(0, 1, 5000) * 4
    assert M.psi(ref, cur) > M.PSI_STABLE


def test_psi_handles_a_constant_reference():
    assert M.psi(np.zeros(100), np.zeros(100)) == 0.0
    assert M.psi(np.zeros(100), np.ones(100)) > M.PSI_SHIFTED


def test_psi_handles_empty_input():
    assert M.psi(np.array([]), np.array([1.0])) == 0.0


def test_bands_are_ordered():
    assert M.band(0.01) == "stable"
    assert M.band(0.15) == "watch"
    assert M.band(0.40) == "shifted"


def test_segment_drift_notices_a_changed_mix():
    a = ["leisure"] * 80 + ["business"] * 20
    b = ["leisure"] * 20 + ["business"] * 80
    assert M.segment_drift(a, b)["drifted"]
    assert not M.segment_drift(a, a)["drifted"]


def test_every_owner_entry_is_complete():
    for check, meta in M.OWNERS.items():
        assert meta["owner"] and meta["severity"] in ("P1", "P2", "P3")
        assert len(meta["runbook"]) > 40, check


def test_monitoring_run_is_silent_on_identical_inputs():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(3000, len(F.FEATURE_NAMES)))
    s = rng.normal(size=3000)
    out = M.run(X, X.copy(), s, s.copy(), F.FEATURE_NAMES)
    assert out["n_alerts"] == 0
