"""
ranker.py — a learning-to-rank model, and the reason its labels are wrong.

The training signal for a ranker is a click log, and a click log is not a
record of what travellers preferred. It is a record of what they preferred
**among the things the previous ranker showed them near the top**. Position one
is examined nearly always and position twelve nearly never, so an option's
click count confounds how good it was with where it was put.

Train on that directly and the model learns the logging policy. It will score
well against held-out clicks -- which were produced by the same policy -- and
it will not be better, because reproducing the ranker you are replacing is not
an improvement. The metric agrees with you the whole way down.

The correction is inverse propensity scoring: weight each observed click by one
over the probability that its position was examined at all, so a click at
position nine counts for more than a click at position one. The propensities
come from the randomised slice of traffic, where position and preference are
independent by construction and the click-rate curve therefore measures
examination and nothing else. Estimating them from ordinary traffic would fit
the ranker's own ordering and remove the signal along with the bias.

Everything here is scored two ways. Against the clicks, which is what you can
do in production, and against `true_utility`, which no real log has and which
exists in this repository so that the click-based metrics can be shown to be
wrong in a specific direction by a specific amount.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import features as F

DEFAULT_PARAMS = dict(
    objective="rank:ndcg",
    eval_metric="ndcg@5",
    # Linear gains, not the 2**r - 1 default. See `ips_labels`: with
    # exponential gains an inverse-propensity label is not an unbiased
    # estimator of anything, and both models have to share a gain function for
    # the comparison between them to mean what it says.
    ndcg_exp_gain=False,
    # LambdaMART. Shallow trees because the feature count is small and the
    # signal is mostly in a handful of interactions; deeper ones memorised
    # search identity and scored worse on the temporal split.
    max_depth=6,
    eta=0.08,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=8,
    reg_lambda=1.5,
    tree_method="hist",
    nthread=2,
)
N_ROUNDS = 320
PROPENSITY_FLOOR = 0.05      # clip; see `estimate_propensities`
MAX_POSITION = 14


# ── propensities from the randomised slice ──────────────────────────────────

def estimate_propensities(meta, max_position: int = MAX_POSITION,
                          floor: float = PROPENSITY_FLOOR) -> dict:
    """Examination probability by position, from exploration traffic only.

    Normalised so position one is 1.0, because only the ratios matter to the
    estimator. Clipped from below: an unclipped inverse propensity at a rarely
    examined position turns a single lucky click into a weight of forty, and
    the variance that buys costs more than the bias it removes.
    """
    clicks = np.zeros(max_position + 1)
    shown = np.zeros(max_position + 1)
    for m in meta:
        p = m["position"]
        if not m["exploration"] or p > max_position:
            continue
        shown[p] += 1
        clicks[p] += m["clicked"]
    rate = np.where(shown > 0, clicks / np.maximum(shown, 1), np.nan)
    base = rate[1] if shown[1] > 0 else np.nan
    prop = np.clip(rate / base, floor, 1.0) if base and base > 0 else rate
    prop = np.nan_to_num(prop, nan=floor)
    return {
        "max_position": max_position, "floor": floor,
        "exploration_rows": int(shown.sum()),
        "click_rate_by_position": [round(float(rate[p]), 5)
                                   for p in range(1, max_position + 1)],
        "propensity": [round(float(prop[p]), 5)
                       for p in range(1, max_position + 1)],
        "note": ("Estimated only on searches shown in a randomised order. In "
                 "ordinary traffic position and preference are correlated by "
                 "design -- that is what a ranker is -- so a curve fitted "
                 "there would remove the signal along with the bias."),
    }


def relevance(meta) -> np.ndarray:
    """Graded relevance from the log: booked beats clicked beats nothing."""
    return np.asarray([2.0 if m["booked"] else 1.0 if m["clicked"] else 0.0
                       for m in meta], dtype=np.float32)


def ips_labels(meta, propensities: dict, raw: np.ndarray | None = None,
               clip: float = 20.0) -> np.ndarray:
    """Relevance divided by the examination probability of its position.

    **The correction goes into the label, not the weight, and that is forced.**
    XGBoost's ranking objectives take `weight` per query *group*, not per
    document -- passing one weight per row raises
    `Size of weight must equal to the number of query groups`. A per-document
    inverse propensity correction therefore cannot be expressed as a weight at
    all.

    It can be expressed as a label. With linear gains (`ndcg_exp_gain=False`)
    the gain LambdaMART optimises *is* the label, so a label of
    `clicked / P(examined at this position)` is exactly the inverse propensity
    estimator of the gain the traveller would have assigned had they examined
    every option. With the default exponential gain it is not, because
    `2**(r/p) - 1` is not an unbiased estimator of anything -- so both models
    here are trained with linear gains and differ only in the labels.

    Clipped at `clip`: the estimator is unbiased but its variance grows as the
    propensity falls, and one click at position twelve carrying a label of
    twenty distorts more than the bias it removes.
    """
    prop = propensities["propensity"]
    rel = relevance(meta) if raw is None else raw
    out = np.array(rel, dtype=np.float32, copy=True)
    for i, m in enumerate(meta):
        if rel[i] > 0:
            p = prop[min(m["position"], len(prop)) - 1]
            out[i] = min(rel[i] / max(p, propensities["floor"]), clip)
    return out


# ── metrics ─────────────────────────────────────────────────────────────────

def _dcg(gains: np.ndarray, k: int) -> float:
    g = gains[:k]
    return float(np.sum(g / np.log2(np.arange(2, len(g) + 2))))


def ndcg_at_k(order_gains: np.ndarray, k: int = 5) -> float:
    ideal = _dcg(np.sort(order_gains)[::-1], k)
    return _dcg(order_gains, k) / ideal if ideal > 0 else 0.0


def oracle_gains(utilities: np.ndarray) -> np.ndarray:
    """Utility rescaled to [0, 1] within a search, used as graded relevance.

    A ratio scale is needed because NDCG multiplies gains by discounts, and
    utilities are an interval scale with an arbitrary zero. Rescaling within
    the search is the standard fix and it keeps the ordering exactly.
    """
    lo, hi = utilities.min(), utilities.max()
    span = (hi - lo) or 1.0
    return (utilities - lo) / span


def evaluate(scores: np.ndarray, groups: list[int], meta, k: int = 5,
             propensities: dict | None = None) -> dict:
    """Score an ordering against the clicks and against the oracle.

    The two disagree, and the size of the disagreement is the point.
    """
    rel = relevance(meta)
    util = np.asarray([m["true_utility"] for m in meta], dtype=np.float32)
    # The estimator a production gate can actually use. The oracle column does
    # not exist outside this repository and the raw click metric is biased
    # toward the ranker that produced the log, so the corrected click metric is
    # the only one that is both available and roughly honest.
    rel_ips = ips_labels(meta, propensities, raw=rel) if propensities else None

    ndcg_click, ndcg_oracle, ndcg_ips, rr_oracle, rr_click = [], [], [], [], []
    start = 0
    for n in groups:
        sl = slice(start, start + n)
        start += n
        order = np.argsort(-scores[sl], kind="stable")
        g_click = rel[sl][order]
        g_oracle = oracle_gains(util[sl])[order]
        ndcg_click.append(ndcg_at_k(g_click, k))
        ndcg_oracle.append(ndcg_at_k(g_oracle, k))
        if rel_ips is not None:
            # UN-normalised, deliberately. NDCG divides by the ideal DCG of the
            # same gains, so correcting the gains also corrects the denominator
            # and most of the correction cancels: an earlier version reported
            # an IPS-corrected NDCG that ranked candidate models identically to
            # the naive one. The unbiased estimator of DCG under a
            # position-based model is the un-normalised sum of corrected gains,
            # and the scale is irrelevant because it is only ever compared
            # between models on the same traffic.
            ndcg_ips.append(_dcg(rel_ips[sl][order], k))
        # Reciprocal rank of the single best option under each definition.
        best_oracle = int(np.argmax(util[sl]))
        rr_oracle.append(1.0 / (1 + int(np.where(order == best_oracle)[0][0])))
        clicked = np.where(rel[sl] > 0)[0]
        if len(clicked):
            pos = min(int(np.where(order == c)[0][0]) for c in clicked)
            rr_click.append(1.0 / (1 + pos))
    return {
        f"ndcg@{k}_oracle": round(float(np.mean(ndcg_oracle)), 5),
        f"ndcg@{k}_clicks": round(float(np.mean(ndcg_click)), 5),
        f"dcg@{k}_ips": (round(float(np.mean(ndcg_ips)), 5) if ndcg_ips else None),
        f"dcg@{k}_clicks": round(float(np.mean(
            [_dcg(rel[slice(s0, s0 + n)][np.argsort(-scores[slice(s0, s0 + n)], kind="stable")], k)
             for s0, n in zip(np.cumsum([0] + groups[:-1]), groups)])), 5),
        "mrr_oracle": round(float(np.mean(rr_oracle)), 5),
        "mrr_clicks": round(float(np.mean(rr_click)), 5) if rr_click else None,
        "searches": len(groups),
    }


# ── baselines ───────────────────────────────────────────────────────────────

def logging_policy_scores(meta) -> np.ndarray:
    """The ranker already in production: whatever order the log was shown in.

    Beating this is the bar. Beating a random order would prove nothing, and
    a great many published ranking improvements are measured against one.
    """
    return np.asarray([-m["position"] for m in meta], dtype=np.float32)


def cheapest_first_scores(X: np.ndarray) -> np.ndarray:
    idx = F.FEATURE_NAMES.index("price_gbp")
    return -X[:, idx]


# ── training ────────────────────────────────────────────────────────────────

@dataclass
class TrainedRanker:
    booster: object
    feature_names: tuple[str, ...]
    config: F.FeatureConfig
    params: dict
    n_rounds: int

    def score(self, X: np.ndarray) -> np.ndarray:
        import xgboost as xgb
        return self.booster.predict(xgb.DMatrix(X, feature_names=list(self.feature_names)))


def train(X: np.ndarray, groups: list[int], meta, cfg: F.FeatureConfig,
          labels: np.ndarray | None = None, params: dict | None = None,
          n_rounds: int = N_ROUNDS) -> TrainedRanker:
    """Fit LambdaMART. `labels` defaults to the graded relevance from the log;
    pass `ips_labels(...)` to train on the position-corrected version."""
    import xgboost as xgb
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    y = relevance(meta) if labels is None else labels
    dtrain = xgb.DMatrix(X, label=y, feature_names=list(F.FEATURE_NAMES))
    dtrain.set_group(groups)
    booster = xgb.train(p, dtrain, num_boost_round=n_rounds)
    return TrainedRanker(booster=booster, feature_names=F.FEATURE_NAMES,
                         config=cfg, params=p, n_rounds=n_rounds)


def temporal_split(groups: list[int], meta, train_fraction: float = 0.7):
    """Split by time, not at random.

    A random split lets the model train on the 20th and test on the 3rd. Nobody
    deploys backwards in time, and on this data the difference is large enough
    to change which model you would ship.
    """
    n_train_groups = int(len(groups) * train_fraction)
    cut_rows = sum(groups[:n_train_groups])
    return (slice(0, cut_rows), groups[:n_train_groups], meta[:cut_rows],
            slice(cut_rows, len(meta)), groups[n_train_groups:], meta[cut_rows:])


def feature_importance(r: TrainedRanker, top: int = 8) -> list[tuple[str, float]]:
    score = r.booster.get_score(importance_type="gain")
    ranked = sorted(score.items(), key=lambda kv: -kv[1])[:top]
    total = sum(score.values()) or 1.0
    return [(k, round(v / total, 4)) for k, v in ranked]
