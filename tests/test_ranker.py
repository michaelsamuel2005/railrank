"""The ranker, its metrics, and the two facts that shape the whole design."""

from __future__ import annotations

import numpy as np
import pytest

from railrank import features as F
from railrank import journeys as J
from railrank import ranker as R


@pytest.fixture(scope="module")
def built():
    searches, options = J.generate(n_searches=5000)
    cfg = F.FeatureConfig(seats_missing_fill=F.seats_missing_fill_from(options))
    X, g, meta, hist = F.build_matrix(searches, options, cfg)
    return searches, options, cfg, X, g, meta, hist


# ── metric primitives ───────────────────────────────────────────────────────

def test_ndcg_is_one_for_a_perfect_ordering():
    assert R.ndcg_at_k(np.array([3.0, 2.0, 1.0]), 3) == pytest.approx(1.0)


def test_ndcg_is_lower_for_a_reversed_ordering():
    assert R.ndcg_at_k(np.array([1.0, 2.0, 3.0]), 3) < 1.0


def test_ndcg_is_zero_when_nothing_is_relevant():
    assert R.ndcg_at_k(np.zeros(5), 5) == 0.0


def test_ndcg_is_scale_invariant_and_dcg_is_not():
    """The reason the IPS correction cannot live in an NDCG denominator."""
    g = np.array([0.0, 1.0, 0.0, 0.0])
    assert R.ndcg_at_k(g, 4) == pytest.approx(R.ndcg_at_k(g * 7.0, 4))
    assert R._dcg(g * 7.0, 4) == pytest.approx(R._dcg(g, 4) * 7.0)


def test_dcg_truncates_at_k():
    g = np.array([1.0, 1.0, 1.0, 1.0])
    assert R._dcg(g, 2) < R._dcg(g, 4)


def test_oracle_gains_preserve_the_ordering():
    u = np.array([-2.0, 0.5, -1.0, 3.0])
    gains = R.oracle_gains(u)
    assert np.argsort(-gains).tolist() == np.argsort(-u).tolist()
    assert gains.min() == pytest.approx(0.0)
    assert gains.max() == pytest.approx(1.0)


# ── propensities ────────────────────────────────────────────────────────────

def test_propensities_fall_with_position(built):
    _, _, _, _, _, meta, _ = built
    p = R.estimate_propensities(meta)["propensity"]
    assert p[0] == pytest.approx(1.0)
    assert p[1] < p[0]
    assert p[7] < p[1]


def test_propensities_come_only_from_exploration_traffic(built):
    _, _, _, _, _, meta, _ = built
    r = R.estimate_propensities(meta)
    assert r["exploration_rows"] == sum(1 for m in meta if m["exploration"])
    assert r["exploration_rows"] < len(meta)


def test_propensities_are_clipped_from_below(built):
    _, _, _, _, _, meta, _ = built
    r = R.estimate_propensities(meta, floor=0.2)
    assert min(r["propensity"]) >= 0.2


# ── labels ──────────────────────────────────────────────────────────────────

def test_relevance_grades_booked_above_clicked(built):
    _, _, _, _, _, meta, _ = built
    rel = R.relevance(meta)
    for i, m in enumerate(meta[:4000]):
        assert rel[i] == (2.0 if m["booked"] else 1.0 if m["clicked"] else 0.0)


def test_ips_labels_only_move_observed_positives(built):
    _, _, _, _, _, meta, _ = built
    prop = R.estimate_propensities(meta)
    rel = R.relevance(meta)
    lab = R.ips_labels(meta, prop, raw=rel)
    assert np.all(lab[rel == 0] == 0.0)
    assert np.all(lab[rel > 0] >= rel[rel > 0] - 1e-6)


def test_ips_labels_up_weight_a_click_further_down_the_page(built):
    _, _, _, _, _, meta, _ = built
    prop = R.estimate_propensities(meta)
    lab = R.ips_labels(meta, prop)
    top = [lab[i] for i, m in enumerate(meta) if m["clicked"] and m["position"] == 1]
    deep = [lab[i] for i, m in enumerate(meta) if m["clicked"] and m["position"] >= 6]
    assert np.mean(deep) > np.mean(top)


def test_ips_labels_are_clipped(built):
    _, _, _, _, _, meta, _ = built
    prop = R.estimate_propensities(meta, floor=0.01)
    assert R.ips_labels(meta, prop, clip=5.0).max() <= 5.0


def test_a_search_can_have_more_than_one_click(built):
    """With one click per search an IPS correction cannot change a listwise
    ranking objective at all: NDCG normalises within the query, so scaling the
    single positive scales the ideal by the same factor."""
    _, _, _, _, g, meta, _ = built
    start, multi = 0, 0
    for n in g:
        if sum(m["clicked"] for m in meta[start:start + n]) >= 2:
            multi += 1
        start += n
    assert multi / len(g) > 0.05


# ── splits and training ─────────────────────────────────────────────────────

def test_temporal_split_does_not_train_on_the_future(built):
    _, _, _, _, g, meta, _ = built
    tr, trg, trm, te, teg, tem = R.temporal_split(g, meta, 0.7)
    assert max(m["searched_at"] for m in trm) <= min(m["searched_at"] for m in tem)
    assert sum(trg) + sum(teg) == len(meta)


def test_the_split_keeps_groups_whole(built):
    _, _, _, _, g, meta, _ = built
    tr, trg, trm, te, teg, tem = R.temporal_split(g, meta, 0.7)
    assert sum(trg) == len(trm)
    assert len({m["search_id"] for m in trm} & {m["search_id"] for m in tem}) == 0


def test_a_trained_ranker_beats_the_logging_policy(built):
    _, _, cfg, X, g, meta, _ = built
    tr, trg, trm, te, teg, tem = R.temporal_split(g, meta, 0.7)
    m = R.train(X[tr], trg, trm, cfg, n_rounds=60)
    learned = R.evaluate(m.score(X[te]), teg, tem)["ndcg@5_oracle"]
    incumbent = R.evaluate(R.logging_policy_scores(tem), teg, tem)["ndcg@5_oracle"]
    assert learned > incumbent + 0.05


def test_evaluate_reports_both_views_and_they_disagree(built):
    _, _, cfg, X, g, meta, _ = built
    tr, trg, trm, te, teg, tem = R.temporal_split(g, meta, 0.7)
    prop = R.estimate_propensities(trm)
    m = R.train(X[tr], trg, trm, cfg, n_rounds=60)
    r = R.evaluate(m.score(X[te]), teg, tem, propensities=prop)
    assert r["ndcg@5_oracle"] > r["ndcg@5_clicks"]
    assert r["dcg@5_ips"] is not None
    assert r["searches"] == len(teg)


def test_evaluate_omits_the_corrected_metric_without_propensities(built):
    _, _, _, X, g, meta, _ = built
    r = R.evaluate(R.logging_policy_scores(meta), g, meta)
    assert r["dcg@5_ips"] is None


def test_feature_importance_sums_to_something_sensible(built):
    _, _, cfg, X, g, meta, _ = built
    tr, trg, trm, *_ = R.temporal_split(g, meta, 0.7)
    m = R.train(X[tr], trg, trm, cfg, n_rounds=60)
    imp = R.feature_importance(m, 5)
    assert len(imp) == 5
    assert all(name in F.FEATURE_NAMES for name, _ in imp)
    assert all(0 <= v <= 1 for _, v in imp)


def test_the_model_uses_linear_gains():
    """Exponential gains would make an inverse-propensity label meaningless."""
    assert R.DEFAULT_PARAMS["ndcg_exp_gain"] is False
