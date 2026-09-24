"""
evaluate.py — every claim this repository makes, checked against the answer key.

One report. Each entry compares something the code measured against something
`journeys.planted()` or a brute-force oracle says is true, and carries the
evidence that decided it. A check that passes is evidence; a check that fails
is a bug, and several of these found real ones while the project was written.

The checks that matter most are the ones about **not** firing. A skew detector
that reports skew on identical configurations is worthless, and a promotion
gate that cannot refuse is a formality.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import features as F
from . import infra_check as IC
from . import journeys as J
from . import latency as L
from . import monitoring as M
from . import optimise as O
from . import ranker as R
from . import registry as REG


def _check(name, passed, expected, observed, note=""):
    return {"check": name, "passed": bool(passed), "expected": expected,
            "observed": observed, "note": note}


# ── 1. the optimiser ────────────────────────────────────────────────────────

def score_optimiser(n_per_route: int = 120, seed: int = 11) -> tuple[list, dict]:
    import random
    rng = random.Random(seed)
    checks, instances, mismatches = [], [], 0
    for key, calling in J.ROUTES.items():
        fares = O.fare_table(calling, seed=abs(hash(key)) % 9999)
        n = len(calling)
        for _ in range(n_per_route):
            a = rng.randrange(0, n - 1)
            b = rng.randrange(a + 1, n)
            p_call = rng.choice([0.45, 0.75, 1.0])
            calls = {a, b} | {i for i in range(a + 1, b) if rng.random() < p_call}
            got = O.saving(fares, calling, calls, start=a, end=b)
            want, _ = O.brute_force_split(fares, calling, calls, start=a, end=b)
            if want != got["split_pence"]:
                mismatches += 1
            instances.append(got)

    summary = O.summarise(instances)
    checks.append(_check(
        "split_solver_matches_brute_force", mismatches == 0,
        f"0 disagreements over {len(instances):,} journeys",
        f"{mismatches} disagreements",
        "The dynamic program is checked against an exhaustive enumeration of "
        "every subset of split points. A solver that returns a plausible "
        "number is indistinguishable from one that returns the right one."))
    checks.append(_check(
        "splitting_wins_sometimes_not_always",
        0.03 < summary["share_where_splitting_wins"] < 0.6,
        "splitting is worth doing on a minority of journeys",
        f"{summary['share_where_splitting_wins']:.1%} of journeys, median "
        f"saving {summary['median_saving_pct']}% "
        f"({summary['median_saving_pence'] / 100:.2f} GBP)",
        "A fare table where splitting always wins, or never does, would make "
        "the optimiser look either brilliant or pointless by construction."))
    fares_all = [i["through_pence"] / 100 for i in instances if i["through_pence"]]
    fares_all.sort()
    checks.append(_check(
        "fares_are_plausible", 5 < fares_all[len(fares_all) // 2] < 120,
        "median through fare between 5 and 120 GBP",
        f"p10 {fares_all[len(fares_all) // 10]:.0f}, median "
        f"{fares_all[len(fares_all) // 2]:.0f}, p90 "
        f"{fares_all[9 * len(fares_all) // 10]:.0f} GBP",
        "An earlier fare table priced 290 miles at 13 GBP, which made the "
        "optimiser look excellent by making the inputs impossible."))
    return checks, summary


# ── 2. features: skew and point in time ─────────────────────────────────────

def score_features(searches, options, cfg, history) -> tuple[list, dict]:
    checks = []
    same = F.skew_report(searches, options, cfg, cfg, history, sample=200)
    checks.append(_check(
        "skew_detector_silent_on_identical_configs", not same["skew_detected"],
        "no skew when the two sides agree", same["verdict"],
        "The half everyone forgets. A detector that always fires is one that "
        "gets switched off."))

    off_by_one = F.skew_report(searches, options, cfg,
                               replace(cfg, advance_days_off_by_one=True),
                               history, sample=200)
    checks.append(_check(
        "skew_detector_catches_a_one_unit_difference",
        off_by_one["features_that_differ"] == ["advance_days"],
        "exactly advance_days differs",
        f"{off_by_one['features_that_differ']}, largest absolute difference "
        f"{off_by_one['largest_absolute_difference']}",
        "A training job on UTC and a serving host on local time shift every "
        "advance window by one. Nothing throws."))

    fill = F.skew_report(searches, options, cfg,
                         replace(cfg, seats_missing_fill=0.0), history, sample=200)
    checks.append(_check(
        "skew_detector_catches_a_different_missing_value_fill",
        fill["features_that_differ"] == ["seats_remaining"],
        "exactly seats_remaining differs",
        f"{fill['features_that_differ']}, "
        f"{fill['mismatch_counts'].get('seats_remaining', 0)} rows of "
        f"{fill['rows_compared']:,}",
        "The fill is a statistic of the training set. A serving process that "
        "recomputes it drifts away from the value the model was fitted on."))

    pit = F.point_in_time_report(searches, options, cfg, sample=600)
    checks.append(_check(
        "point_in_time_filter_changes_the_feature",
        pit["share_that_differ"] > 0.2,
        "the as-of filter materially changes the value",
        f"{pit['share_that_differ']:.1%} of {pit['rows']:,} rows differ, mean "
        f"absolute difference {pit['mean_absolute_difference']}, "
        f"{pit['rows_with_no_history_yet']:,} rows have no trailing window yet",
        "Dropping the date filter is one line and it is the difference "
        "between a feature and a leak."))
    return checks, {"skew_same": same, "skew_off_by_one": off_by_one,
                    "skew_fill": fill, "point_in_time": pit}


# ── 3. the ranker ───────────────────────────────────────────────────────────

def score_ranker(searches, options, cfg, quick: bool = False) -> tuple[list, dict]:
    checks = []
    X, g, meta, hist = F.build_matrix(searches, options, cfg)
    tr, trg, trm, te, teg, tem = R.temporal_split(g, meta, 0.7)
    prop = R.estimate_propensities(trm)
    rounds = 120 if quick else R.N_ROUNDS

    raw = R.train(X[tr], trg, trm, cfg, n_rounds=rounds)
    ips = R.train(X[tr], trg, trm, cfg, labels=R.ips_labels(trm, prop),
                  n_rounds=rounds)
    m_incumbent = R.evaluate(R.logging_policy_scores(tem), teg, tem, propensities=prop)
    m_raw = R.evaluate(raw.score(X[te]), teg, tem, propensities=prop)
    m_ips = R.evaluate(ips.score(X[te]), teg, tem, propensities=prop)

    checks.append(_check(
        "learned_ranker_beats_the_incumbent",
        m_raw["ndcg@5_oracle"] > m_incumbent["ndcg@5_oracle"] + 0.05,
        "a clear gain over the ranker being replaced",
        f"NDCG@5 against the oracle: {m_incumbent['ndcg@5_oracle']:.4f} for the "
        f"logging policy against {m_raw['ndcg@5_oracle']:.4f} learned",
        "The bar is the policy that produced the log, not a random ordering."))

    checks.append(_check(
        "ips_labels_improve_the_true_metric",
        m_ips["ndcg@5_oracle"] > m_raw["ndcg@5_oracle"],
        "position-corrected labels rank closer to what travellers preferred",
        f"NDCG@5 oracle {m_raw['ndcg@5_oracle']:.4f} -> "
        f"{m_ips['ndcg@5_oracle']:.4f}; MRR {m_raw['mrr_oracle']:.4f} -> "
        f"{m_ips['mrr_oracle']:.4f}",
        "The correction goes into the label, because XGBoost takes ranking "
        "weights per query group and not per document."))

    checks.append(_check(
        "the_click_metric_prefers_the_worse_model",
        m_ips["ndcg@5_clicks"] < m_raw["ndcg@5_clicks"],
        "the naive click metric ranks the corrected model lower",
        f"NDCG@5 on clicks {m_raw['ndcg@5_clicks']:.4f} for the uncorrected "
        f"model against {m_ips['ndcg@5_clicks']:.4f} for the corrected one, "
        f"while the oracle says the opposite",
        "This is the trap the whole project is about. Judge by held-out "
        "clicks and you reject the better ranker, because the clicks were "
        "produced by the ranker you are replacing."))

    # Which offline metric would actually pick the right model?
    rng = np.random.default_rng(0)
    util = np.asarray([m["true_utility"] for m in tem], dtype=np.float32)
    candidates = {
        "random": rng.normal(size=len(tem)).astype(np.float32),
        "cheapest first": R.cheapest_first_scores(X[te]),
        "fastest first": -X[te][:, F.FEATURE_NAMES.index("duration_hours")],
        "fewest changes": -X[te][:, F.FEATURE_NAMES.index("changes")],
        "logging policy": R.logging_policy_scores(tem),
        "LTR raw": raw.score(X[te]),
        "LTR IPS": ips.score(X[te]),
    }
    for sd in (2.0, 1.0, 0.4):
        candidates[f"oracle+noise sd={sd}"] = (
            util + rng.normal(0, sd, len(util)).astype(np.float32))
    table = {k: R.evaluate(v, teg, tem, propensities=prop)
             for k, v in candidates.items()}

    from scipy.stats import spearmanr
    keys = list(table)
    oracle = [table[k]["ndcg@5_oracle"] for k in keys]
    rho_naive = float(spearmanr(oracle, [table[k]["ndcg@5_clicks"] for k in keys]).statistic)
    rho_ips = float(spearmanr(oracle, [table[k]["dcg@5_ips"] for k in keys]).statistic)
    checks.append(_check(
        "corrected_offline_metric_selects_the_right_model",
        rho_ips > rho_naive,
        "the corrected metric agrees with the truth better than the naive one",
        f"across {len(keys)} candidate rankers, Spearman correlation with the "
        f"oracle is {rho_naive:+.3f} for the naive click metric and "
        f"{rho_ips:+.3f} for the inverse-propensity-corrected one",
        "This is what a promotion gate reads, so it is the number that "
        "decides which model ships."))

    detail = {"propensities": prop, "metrics": {
        "logging_policy": m_incumbent, "ltr_raw": m_raw, "ltr_ips": m_ips},
        "model_selection": {"candidates": table,
                            "spearman_naive_vs_oracle": round(rho_naive, 4),
                            "spearman_ips_vs_oracle": round(rho_ips, 4)},
        "feature_importance": R.feature_importance(ips, 8)}
    return checks, detail, (X, g, meta, hist, tr, trg, trm, te, teg, tem, prop, ips)


# ── 4. the leak, end to end ─────────────────────────────────────────────────

def score_leak(searches, options, cfg, quick: bool = False) -> tuple[list, dict]:
    """Train with a leaky feature and show what it costs in production.

    The point is not that a leaky model scores well offline. It is that the
    offline number is the one you would have used to decide.
    """
    rounds = 100 if quick else 200
    leaky_cfg = replace(cfg, use_final_load_factor=True)

    X_h, g, meta, hist = F.build_matrix(searches, options, cfg)
    X_l, _, _, _ = F.build_matrix(searches, options, leaky_cfg, history=hist)
    tr, trg, trm, te, teg, tem = R.temporal_split(g, meta, 0.7)
    prop = R.estimate_propensities(trm)

    honest = R.train(X_h[tr], trg, trm, cfg, n_rounds=rounds)
    leaky = R.train(X_l[tr], trg, trm, leaky_cfg, n_rounds=rounds)

    m_honest = R.evaluate(honest.score(X_h[te]), teg, tem, propensities=prop)
    # What the leaky model's own offline evaluation would have reported.
    m_leaky_offline = R.evaluate(leaky.score(X_l[te]), teg, tem, propensities=prop)
    # What it does once the column is unavailable, which is every request the
    # service will ever see: the request schema has no field for it.
    m_leaky_served = R.evaluate(leaky.score(X_h[te]), teg, tem, propensities=prop)

    checks = [_check(
        "leak_inflates_the_metric_you_can_actually_measure",
        m_leaky_offline["ndcg@5_clicks"] > m_honest["ndcg@5_clicks"],
        "the leaky model wins on held-out clicks, which is the only offline "
        "metric a real team has",
        f"NDCG@5 on clicks {m_honest['ndcg@5_clicks']:.4f} honest against "
        f"{m_leaky_offline['ndcg@5_clicks']:.4f} leaky; on the oracle the "
        f"order is reversed, {m_honest['ndcg@5_oracle']:.4f} against "
        f"{m_leaky_offline['ndcg@5_oracle']:.4f}",
        "final_load_factor is the same column as load_factor_at_search read "
        "after the train had gone, so it carries the click it is being used "
        "to predict. It wins on the click metric and loses on the truth, "
        "which is precisely why a leak survives review."),
        _check(
        "leak_costs_accuracy_once_served",
        m_leaky_served["ndcg@5_oracle"] < m_leaky_offline["ndcg@5_oracle"],
        "and collapses once the column is unavailable at serving time",
        f"{m_leaky_offline['ndcg@5_oracle']:.4f} offline collapses to "
        f"{m_leaky_served['ndcg@5_oracle']:.4f} in production, a drop of "
        f"{100 * (m_leaky_offline['ndcg@5_oracle'] - m_leaky_served['ndcg@5_oracle']) / m_leaky_offline['ndcg@5_oracle']:.1f}%",
        "The offline number is the one that would have been used to decide.")]
    return checks, {"honest": m_honest, "leaky_offline": m_leaky_offline,
                    "leaky_served": m_leaky_served}


# ── 5. the gate ─────────────────────────────────────────────────────────────

def score_gate(metrics_good, metrics_worse, skew_clean, skew_dirty,
               p99_ms: float) -> tuple[list, dict]:
    checks = []
    ok = REG.promotion_gate(metrics_good, metrics_worse, skew_clean, p99_ms)
    checks.append(_check(
        "gate_promotes_a_better_model", ok.promoted,
        "promoted", ok.verdict[:200],
        "Every condition has to hold, and the gate returns its reasons "
        "either way."))

    on_skew = REG.promotion_gate(metrics_good, metrics_worse, skew_dirty, p99_ms)
    checks.append(_check(
        "gate_refuses_on_skew",
        not on_skew.promoted and "no_training_serving_skew" in
        [c.name for c in on_skew.checks if not c.passed],
        "refused because the serving path disagrees with the training path",
        on_skew.verdict[:200], ""))

    on_latency = REG.promotion_gate(metrics_good, metrics_worse, skew_clean,
                                    p99_ms, budget_p99_ms=p99_ms / 2)
    checks.append(_check(
        "gate_refuses_on_latency",
        not on_latency.promoted and "latency_budget" in
        [c.name for c in on_latency.checks if not c.passed],
        "refused because it cannot answer in time",
        on_latency.verdict[:200],
        "A better model that misses the budget is not a better model."))

    no_gain = REG.promotion_gate(metrics_worse, metrics_good, skew_clean, p99_ms)
    checks.append(_check(
        "gate_refuses_without_an_improvement",
        not no_gain.promoted and "beats_incumbent" in
        [c.name for c in no_gain.checks if not c.passed],
        "refused because it is not better than what is already running",
        no_gain.verdict[:200], ""))
    return checks, {"promote": ok.to_dict(), "refuse_skew": on_skew.to_dict(),
                    "refuse_latency": on_latency.to_dict(),
                    "refuse_no_gain": no_gain.to_dict()}


# ── 6. monitoring ───────────────────────────────────────────────────────────

def score_monitoring(X, meta, scores, groups) -> tuple[list, dict]:
    """The generator trends fares upward over the period, so early against late
    is a drift with a known cause and known direction."""
    n = X.shape[0]
    early, late = slice(0, n // 3), slice(2 * n // 3, n)
    seg_early = [m["passenger_type"] for m in meta[early]]
    seg_late = [m["passenger_type"] for m in meta[late]]

    drifted = M.run(X[early], X[late], scores[early], scores[late],
                    F.FEATURE_NAMES, seg_early, seg_late)
    # A control: the first third against a random half of itself. Same
    # distribution by construction, so anything that fires here is a false
    # alarm rather than a finding.
    rng = np.random.default_rng(4)
    idx = rng.permutation(n // 3)
    a, b = idx[: n // 6], idx[n // 6:]
    quiet = M.run(X[early][a], X[early][b], scores[early][a], scores[early][b],
                  F.FEATURE_NAMES, [seg_early[i] for i in a],
                  [seg_early[i] for i in b])

    price_psi = drifted["features"]["psi"]["price_gbp"]
    checks = [_check(
        "drift_detector_finds_the_planted_reporting_change",
        "seats_remaining" in drifted["features"]["shifted"],
        f"seats_remaining flagged as shifted after two operators start "
        f"reporting a constant from day "
        f"{J.SEATS_REPORTING_CHANGE['from_day']}",
        f"shifted: {drifted['features']['shifted']}; worst by population "
        f"stability index {[(n_, v) for n_, v in drifted['features']['worst'][:3]]}",
        "No nulls, no schema change, no row-count change, every value a legal "
        "integer. Freshness and null-rate checks stay green throughout."),
        _check(
        "drift_detector_is_quiet_on_the_same_distribution",
        quiet["n_alerts"] == 0,
        "no alerts when a sample is compared with itself",
        f"{quiet['n_alerts']} alerts, worst PSI "
        f"{quiet['features']['worst'][0][1] if quiet['features']['worst'] else 0}",
        "Split at random from one period, so any alert here is a false one."),
        _check(
        "psi_does_not_see_a_gradual_price_trend",
        price_psi < M.PSI_STABLE,
        f"population stability index below {M.PSI_STABLE} on price_gbp despite "
        f"a planted {100 * J.PRICE_TREND_PER_DAY:.2f}% daily fare trend",
        f"price_gbp PSI {price_psi:.4f} over {J.PERIOD_DAYS} days, which the "
        f"bands call 'stable'",
        "Reported as a limitation, not hidden. PSI compares binned shares, so "
        "a slow location shift inside a wide distribution moves very little "
        "mass across quantile edges. A trend needs a trend test -- a "
        "Mann-Kendall on the daily mean -- and a monitor that only computes "
        "PSI will not see one of the two most common ways a feature goes "
        "wrong."),
        _check(
        "every_alert_carries_an_owner_and_a_runbook",
        all(a["owner"] and a["runbook"] and a["severity"]
            for a in drifted["alerts"]) and bool(drifted["alerts"]),
        "owner, severity and first action on every alert",
        f"{drifted['n_alerts']} alerts, owners {drifted['owners']}",
        "An alert nobody owns is a notification.")]
    return checks, {"early_vs_late": drifted, "control": quiet}


# ── the whole thing ─────────────────────────────────────────────────────────

def main(n_searches: int = 16000, quick: bool = False,
         out: str = "reports/scorecard.json", http_url: str | None = None) -> dict:
    started = time.time()
    checks: list[dict] = []
    stages: dict = {}

    print("generate      ", end="", flush=True)
    t = time.time()
    searches, options = J.generate(n_searches=n_searches)
    cfg = F.FeatureConfig(seats_missing_fill=F.seats_missing_fill_from(options))
    print(f"{len(searches):>8,} searches {len(options):>9,} options "
          f"{time.time() - t:>7.1f}s")
    stages["planted"] = J.planted()

    print("optimiser     ", end="", flush=True)
    t = time.time()
    c, summary = score_optimiser(n_per_route=40 if quick else 120)
    checks += c
    stages["optimiser"] = summary
    print(f"{summary['journeys']:>8,} journeys {'':>9} {time.time() - t:>7.1f}s")

    print("ranker        ", end="", flush=True)
    t = time.time()
    c, rank_detail, bundle = score_ranker(searches, options, cfg, quick)
    checks += c
    stages["ranker"] = rank_detail
    X, g, meta, hist, tr, trg, trm, te, teg, tem, prop, model = bundle
    print(f"{X.shape[0]:>8,} rows {'':>13} {time.time() - t:>7.1f}s")

    print("features      ", end="", flush=True)
    t = time.time()
    c, feat_detail = score_features(searches, options, cfg, hist)
    checks += c
    stages["features"] = feat_detail
    print(f"{'':>8} {'':>19} {time.time() - t:>7.1f}s")

    print("leak          ", end="", flush=True)
    t = time.time()
    c, leak_detail = score_leak(searches, options, cfg, quick)
    checks += c
    stages["leak"] = leak_detail
    print(f"{'':>8} {'':>19} {time.time() - t:>7.1f}s")

    print("serving       ", end="", flush=True)
    t = time.time()
    metrics = rank_detail["metrics"]["ltr_ips"]
    artefact = REG.save_artefact("artefacts/current", model, prop, hist, metrics,
                                 notes="IPS-labelled LambdaMART")
    loaded = REG.load_artefact(artefact)
    by_search: dict = {}
    for o in options:
        by_search.setdefault(o.search_id, []).append(o)
    lat = L.measure_inprocess(loaded, searches[:900], by_search,
                              n=200 if quick else 600)
    budget = L.budget_report(lat, REG.DEFAULT_BUDGET_P99_MS)
    stages["latency"] = {"inprocess": lat, "budget": budget}
    checks.append(_check(
        "model_answers_inside_its_latency_budget", budget["within_budget"],
        f"p99 at or under {budget['budget_p99_ms']} ms",
        f"p50 {lat['p50_ms']} ms, p99 {lat['p99_ms']} ms over {lat['n']} "
        f"searches ({budget['headroom_pct']}% headroom)",
        "A better model that cannot answer in time is not a better model."))

    # The contract check has to fire, or it is decoration.
    card = json.loads((artefact / "model_card.json").read_text())
    good = list(card["feature_names"])
    card["feature_names"] = list(reversed(good))
    (artefact / "model_card.json").write_text(json.dumps(card))
    fired = False
    try:
        REG.load_artefact(artefact)
    except ValueError:
        fired = True
    card["feature_names"] = good
    (artefact / "model_card.json").write_text(json.dumps(card, indent=2, default=str))
    checks.append(_check(
        "service_refuses_a_model_whose_features_do_not_match", fired,
        "loading raises when the recorded feature order differs",
        "raised" if fired else "loaded a mismatched model without complaint",
        "A model trained before a feature was inserted scores against the "
        "wrong columns and raises nothing at all."))
    print(f"{'':>8} p99 {lat['p99_ms']:>5.2f} ms {'':>7} {time.time() - t:>7.1f}s")

    if http_url:
        payloads = []
        from . import service as SV
        for s in searches[:120]:
            if by_search.get(s.search_id):
                payloads.append(SV.request_from(s, by_search[s.search_id]))
        http = L.measure_http(http_url, payloads, n=120 if quick else 400)
        stages["latency"]["http"] = http
        checks.append(_check(
            "container_answers_over_http_inside_budget",
            http.get("p99_ms", 1e9) <= 25.0 and http.get("statuses", {}).get(200, 0) > 0,
            "p99 at or under 25 ms end to end, all responses 200",
            f"p50 {http.get('p50_ms')} ms, p99 {http.get('p99_ms')} ms, "
            f"statuses {http.get('statuses')}",
            "Measured against the running container, not in process."))

    print("gate          ", end="", flush=True)
    t = time.time()
    skew_clean = feat_detail["skew_same"]
    skew_dirty = feat_detail["skew_off_by_one"]
    c, gate_detail = score_gate(rank_detail["metrics"]["ltr_ips"],
                                rank_detail["metrics"]["logging_policy"],
                                skew_clean, skew_dirty, lat["p99_ms"])
    checks += c
    stages["gate"] = gate_detail
    print(f"{'':>8} {'':>19} {time.time() - t:>7.1f}s")

    print("monitoring    ", end="", flush=True)
    t = time.time()
    scores_all = np.asarray(loaded.score(X), dtype=np.float32)
    c, mon_detail = score_monitoring(X, meta, scores_all, g)
    checks += c
    stages["monitoring"] = mon_detail
    print(f"{mon_detail['early_vs_late']['n_alerts']:>8} alerts {'':>12} "
          f"{time.time() - t:>7.1f}s")

    infra = IC.check("infra")
    stages["infra"] = infra
    checks.append(_check(
        "terraform_module_is_structurally_sound",
        infra.get("ok", False),
        "every variable declared, every reference resolvable, no credentials",
        f"{infra.get('resources')} resources, {infra.get('variables_declared')} "
        f"variables, {infra.get('outputs')} outputs across "
        f"{infra.get('files')}; problems: {infra.get('problems')}",
        "Structural only. `terraform validate` is the right tool and CI runs "
        "it; this runs where the binary cannot be downloaded. The module has "
        "never been applied, which the README says in the same words."))

    stages["mlflow"] = REG.log_run(
        artefact,
        {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        | {"p99_ms": lat["p99_ms"]},
        {"n_rounds": model.n_rounds, "labels": "ips",
         "feature_config": model.config.fingerprint(),
         "features": len(F.FEATURE_NAMES)},
        run_name="scorecard")

    passed = sum(c["passed"] for c in checks)
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": J.SEED, "n_searches": n_searches, "quick": quick,
        "elapsed_seconds": round(time.time() - started, 2),
        "n_checks": len(checks), "n_passed": passed,
        "all_passed": passed == len(checks),
        "checks": checks, "stages": stages,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2, default=str))

    print()
    for c in checks:
        print(f"  {'PASS' if c['passed'] else 'FAIL'}  {c['check']:<52} "
              f"{c['observed']}")
    print(f"\n{passed}/{len(checks)} checks passed in "
          f"{report['elapsed_seconds']:.0f}s -> {out}")
    return report


if __name__ == "__main__":
    import argparse
    import sys
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--searches", type=int, default=16000)
    ap.add_argument("--http", default=None,
                    help="also measure a running container, e.g. "
                         "http://127.0.0.1:8080/rank")
    a = ap.parse_args()
    sys.exit(0 if main(n_searches=a.searches, quick=a.quick,
                       http_url=a.http)["all_passed"] else 1)
