"""
monitoring.py — watching the model, not just the tables.

Data monitoring asks whether the rows arrived. Model monitoring asks whether
the rows still look like the ones the model was fitted on, and whether the
model is still saying the same kind of thing about them. A pipeline can be
green on every freshness and null-rate check while the model quietly stops
working, because nothing it depends on is missing -- it has simply moved.

Three signals, in the order they go wrong:

  feature drift       an input distribution shifts. Fares rise over a season,
                      a new operator appears, an upstream default changes.
  prediction drift    the score distribution shifts. This is the one that
                      matters, because a feature can move a long way without
                      the model caring, and a small input shift on a feature
                      the model leans on will show up here first.
  segment drift       the mix of traffic changes. The model is fine and the
                      population is not the one it was trained for.

Population stability index throughout, with the conventional bands: under 0.1
stable, 0.1 to 0.25 worth a look, above 0.25 a real shift. Those thresholds are
a convention rather than a derivation, and this module says so rather than
implying they were computed. What is measured here is whether the detector
fires on a shift that was planted and stays quiet on one that was not.

Alerts carry an owner, a severity and a first action, because an alert nobody
owns is a notification and notifications get muted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

# The conventional PSI bands. Stated as a convention, not as a result.
PSI_STABLE = 0.10
PSI_SHIFTED = 0.25
EPSILON = 1e-6

OWNERS = {
    "feature_drift": {
        "owner": "ml-platform", "severity": "P2",
        "runbook": "Compare the drifted feature against its upstream source "
                   "before retraining. A shift caused by a producer change is "
                   "a bug to fix; a shift caused by the world moving is a "
                   "reason to retrain, and retraining on the first one bakes "
                   "the bug into the model.",
    },
    "prediction_drift": {
        "owner": "ml-platform", "severity": "P1",
        "runbook": "Check feature drift first: a prediction shift with no "
                   "input shift means the model or the artefact changed, not "
                   "the traffic. Confirm the deployed model version against "
                   "the registry before anything else.",
    },
    "segment_drift": {
        "owner": "search-product", "severity": "P3",
        "runbook": "A traffic-mix change is usually a marketing or routing "
                   "change rather than a fault. Confirm it was intended, then "
                   "re-check the model's per-segment metrics.",
    },
}


@dataclass
class Alert:
    check: str
    subject: str
    severity: str
    owner: str
    summary: str
    runbook: str
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _alert(check: str, subject: str, summary: str, evidence: dict) -> Alert:
    m = OWNERS[check]
    return Alert(check=check, subject=subject, severity=m["severity"],
                 owner=m["owner"], summary=summary, runbook=m["runbook"],
                 evidence=evidence)


# ── the statistic ───────────────────────────────────────────────────────────

def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population stability index between two samples of one quantity.

    Bin edges come from the REFERENCE quantiles, not from the pooled data.
    Pooling lets the new data move the edges, which absorbs the very shift the
    statistic is supposed to report -- a drift detector that re-bins on the
    drifted sample understates every drift it sees.

    A constant reference has no quantiles to speak of, so the answer is zero
    drift unless the current sample is also not that constant, which is
    reported as a large value rather than a division by zero.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    if ref.size == 0 or cur.size == 0:
        return 0.0

    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if edges.size < 2:
        return 0.0 if np.allclose(cur, ref[0]) else float(PSI_SHIFTED * 4)
    edges[0], edges[-1] = -np.inf, np.inf

    r = np.histogram(ref, bins=edges)[0] / ref.size
    c = np.histogram(cur, bins=edges)[0] / cur.size
    r = np.clip(r, EPSILON, None)
    c = np.clip(c, EPSILON, None)
    return float(np.sum((c - r) * np.log(c / r)))


def band(value: float) -> str:
    if value < PSI_STABLE:
        return "stable"
    if value < PSI_SHIFTED:
        return "watch"
    return "shifted"


# ── the checks ──────────────────────────────────────────────────────────────

def feature_drift(reference: np.ndarray, current: np.ndarray,
                  names: tuple[str, ...], bins: int = 10) -> dict:
    """PSI per feature, worst first."""
    scores = {n: round(psi(reference[:, i], current[:, i], bins), 5)
              for i, n in enumerate(names)}
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    return {
        "reference_rows": int(reference.shape[0]),
        "current_rows": int(current.shape[0]),
        "psi": scores,
        "worst": ranked[:5],
        "shifted": [n for n, v in ranked if v >= PSI_SHIFTED],
        "watch": [n for n, v in ranked if PSI_STABLE <= v < PSI_SHIFTED],
    }


def prediction_drift(reference_scores: np.ndarray,
                     current_scores: np.ndarray, bins: int = 10) -> dict:
    """PSI on the score distribution, plus where the mass moved."""
    value = psi(reference_scores, current_scores, bins)
    return {
        "psi": round(value, 5), "band": band(value),
        "reference_mean": round(float(np.mean(reference_scores)), 5),
        "current_mean": round(float(np.mean(current_scores)), 5),
        "reference_p90": round(float(np.quantile(reference_scores, 0.9)), 5),
        "current_p90": round(float(np.quantile(current_scores, 0.9)), 5),
        "drifted": bool(value >= PSI_SHIFTED),
    }


def segment_drift(reference_labels, current_labels) -> dict:
    """Has the traffic mix moved, independent of any feature value."""
    keys = sorted(set(reference_labels) | set(current_labels))
    ref = np.array([sum(1 for x in reference_labels if x == k) for k in keys],
                   dtype=float)
    cur = np.array([sum(1 for x in current_labels if x == k) for k in keys],
                   dtype=float)
    ref = np.clip(ref / max(ref.sum(), 1), EPSILON, None)
    cur = np.clip(cur / max(cur.sum(), 1), EPSILON, None)
    value = float(np.sum((cur - ref) * np.log(cur / ref)))
    return {
        "psi": round(value, 5), "band": band(value),
        "reference_share": {k: round(float(r), 4) for k, r in zip(keys, ref)},
        "current_share": {k: round(float(c), 4) for k, c in zip(keys, cur)},
        "drifted": bool(value >= PSI_SHIFTED),
    }


def run(reference_X: np.ndarray, current_X: np.ndarray,
        reference_scores: np.ndarray, current_scores: np.ndarray,
        names: tuple[str, ...], reference_segments=None,
        current_segments=None) -> dict:
    """Every check, with the alerts they raise."""
    feats = feature_drift(reference_X, current_X, names)
    preds = prediction_drift(reference_scores, current_scores)
    segs = (segment_drift(reference_segments, current_segments)
            if reference_segments is not None else None)

    alerts: list[Alert] = []
    for name in feats["shifted"]:
        alerts.append(_alert(
            "feature_drift", name,
            f"`{name}` has a population stability index of "
            f"{feats['psi'][name]:.3f} against the training distribution, "
            f"above the {PSI_SHIFTED} threshold. Nothing is missing and no "
            f"row count changed; the values have moved.",
            {"feature": name, "psi": feats["psi"][name]}))
    if preds["drifted"]:
        alerts.append(_alert(
            "prediction_drift", "model scores",
            f"The score distribution has shifted (PSI {preds['psi']:.3f}); "
            f"mean {preds['reference_mean']:.3f} to {preds['current_mean']:.3f}.",
            preds))
    if segs and segs["drifted"]:
        alerts.append(_alert(
            "segment_drift", "traffic mix",
            f"The traffic mix has moved (PSI {segs['psi']:.3f}).", segs))

    return {
        "features": feats, "predictions": preds, "segments": segs,
        "alerts": [a.to_dict() for a in alerts],
        "n_alerts": len(alerts),
        "owners": sorted({a.owner for a in alerts}),
        "thresholds": {"stable_below": PSI_STABLE, "shifted_at_or_above": PSI_SHIFTED,
                       "note": "conventional PSI bands, not derived here"},
    }
