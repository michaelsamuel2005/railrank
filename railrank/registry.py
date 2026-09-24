"""
registry.py — tracking, a servable artefact, and a gate that can say no.

Three jobs, kept apart on purpose.

**Tracking** is MLflow. Every training run records its parameters, its metrics,
the feature configuration it was fitted against and the model itself, so two
runs a month apart can be compared without anyone remembering what they did.

**The artefact** is a plain directory: the booster, the feature names in order,
the feature configuration, the propensity curve and a snapshot of the route
history. The serving process reads that and nothing else. It does not import
MLflow, does not reach a tracking server, and cannot be broken by one being
down at three in the morning. MLflow is how the model is found; it is not a
runtime dependency of the thing serving traffic.

**The gate** is a function. Promotion is not a button somebody presses when the
number looks bigger -- it is a set of conditions evaluated against evidence,
and it returns its reasons either way. A gate that can only approve is a
formality, so the refusal path is exercised in the tests and in the scorecard.

The metric the gate reads is the inverse-propensity-corrected DCG, not the raw
click metric. `evaluate.py` measures why: across a family of candidate rankers
the naive click metric agrees with the truth at a Spearman correlation of about
0.6, and the corrected one at 1.0. Gating on the naive metric would reliably
promote the model that best reproduces the ranker being replaced.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

from . import features as F

ARTEFACT_FILES = ("booster.json", "model_card.json")
DEFAULT_BUDGET_P99_MS = 25.0
MIN_IMPROVEMENT = 0.002          # 0.2% on the corrected offline metric


# ── the servable artefact ───────────────────────────────────────────────────

def save_artefact(path: Path | str, ranker, propensities: dict,
                  history: F.RouteHistory, metrics: dict,
                  notes: str = "") -> Path:
    """Write everything the serving process needs, and nothing it does not."""
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)

    ranker.booster.save_model(str(path / "booster.json"))
    card = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        # The order is the contract between the training matrix and the
        # booster's column indices. It is written down so the service can
        # refuse to start if it ever disagrees.
        "feature_names": list(ranker.feature_names),
        "feature_config": asdict(ranker.config),
        "feature_config_fingerprint": ranker.config.fingerprint(),
        "xgb_params": {k: v for k, v in ranker.params.items()},
        "n_rounds": ranker.n_rounds,
        "propensities": propensities,
        "offline_metrics": metrics,
        "route_history": _history_to_json(history),
        "notes": notes,
    }
    (path / "model_card.json").write_text(json.dumps(card, indent=2, default=str))
    return path


def _history_to_json(history: F.RouteHistory) -> dict:
    return {route: {"days": [str(d) for d in history._days[route]],
                    "median": history._median[route]}
            for route in history._days}


def _history_from_json(blob: dict) -> F.RouteHistory:
    h = F.RouteHistory()
    for route, v in blob.items():
        h._days[route] = [date.fromisoformat(d) for d in v["days"]]
        h._median[route] = list(v["median"])
    return h


@dataclass
class LoadedModel:
    """A model as the serving process sees it."""
    booster: object
    feature_names: tuple[str, ...]
    config: F.FeatureConfig
    history: F.RouteHistory
    card: dict

    def score(self, rows) -> list[float]:
        import numpy as np
        import xgboost as xgb
        m = xgb.DMatrix(np.asarray(rows, dtype=np.float32),
                        feature_names=list(self.feature_names))
        return [float(x) for x in self.booster.predict(m)]


def load_artefact(path: Path | str) -> LoadedModel:
    """Load a model for serving, refusing anything that does not match the code.

    The feature-order check is the important line in this function. A model
    trained before a feature was inserted will happily produce scores against
    the new column order -- wrong scores, no exception, no alert. Comparing the
    recorded names against `F.FEATURE_NAMES` turns that into a process that
    will not start, which is the only failure mode anyone notices.
    """
    import xgboost as xgb
    path = Path(path)
    card = json.loads((path / "model_card.json").read_text())

    names = tuple(card["feature_names"])
    if names != F.FEATURE_NAMES:
        missing = set(F.FEATURE_NAMES) - set(names)
        extra = set(names) - set(F.FEATURE_NAMES)
        raise ValueError(
            "feature contract mismatch between the model and this code: "
            f"{len(names)} recorded vs {len(F.FEATURE_NAMES)} expected; "
            f"missing={sorted(missing)} unexpected={sorted(extra)}; "
            "order differs" if set(names) == set(F.FEATURE_NAMES) else "")

    booster = xgb.Booster()
    booster.load_model(str(path / "booster.json"))
    # The configuration is LOADED, never constructed. A serving process that
    # builds its own FeatureConfig has reintroduced training/serving skew by
    # the only route that matters.
    cfg = F.FeatureConfig(**card["feature_config"])
    return LoadedModel(booster=booster, feature_names=names, config=cfg,
                       history=_history_from_json(card["route_history"]),
                       card=card)


# ── MLflow tracking ─────────────────────────────────────────────────────────

def log_run(artefact_dir: Path | str, metrics: dict, params: dict,
            tracking_uri: str = "sqlite:///mlflow.db",
            experiment: str = "railrank-ltr",
            run_name: str | None = None,
            registered_model: str | None = "railrank-ranker") -> dict:
    """Record one training run, and register the model if asked.

    Returns a summary rather than an MLflow object so that callers -- and the
    scorecard -- do not depend on MLflow's types. Failures here are reported
    and do not raise: a tracking server being unavailable should not lose a
    model that trained successfully.
    """
    try:
        import mlflow
    except ImportError:
        return {"logged": False, "reason": "mlflow not installed"}

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()
                                if isinstance(v, (int, float))})
            mlflow.log_artifacts(str(artefact_dir), artifact_path="model")
            info = {"logged": True, "run_id": run.info.run_id,
                    "experiment": experiment, "tracking_uri": tracking_uri}
        if registered_model:
            uri = f"runs:/{info['run_id']}/model"
            mv = mlflow.register_model(uri, registered_model)
            info["registered_model"] = registered_model
            info["version"] = mv.version
        return info
    except Exception as exc:                        # noqa: BLE001
        return {"logged": False, "reason": f"{type(exc).__name__}: {exc}"}


# ── the gate ────────────────────────────────────────────────────────────────

@dataclass
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class GateResult:
    promoted: bool
    checks: list[Check] = field(default_factory=list)
    verdict: str = ""

    def to_dict(self) -> dict:
        return {"promoted": self.promoted,
                "checks": [asdict(c) for c in self.checks],
                "failed": [c.name for c in self.checks if not c.passed],
                "verdict": self.verdict}


def promotion_gate(candidate_metrics: dict, incumbent_metrics: dict | None,
                   skew: dict, measured_p99_ms: float | None,
                   feature_names: tuple[str, ...] = F.FEATURE_NAMES,
                   budget_p99_ms: float = DEFAULT_BUDGET_P99_MS,
                   min_improvement: float = MIN_IMPROVEMENT,
                   metric: str = "dcg@5_ips") -> GateResult:
    """Promote only if every condition holds. Return the reasons either way.

    The conditions are ordered so the report reads like an argument: is the
    model the same shape as the code, does the serving path agree with the
    training path, is it actually better, and can it answer in time.
    """
    checks: list[Check] = []

    checks.append(Check(
        "feature_contract", tuple(feature_names) == F.FEATURE_NAMES,
        f"{len(feature_names)} features in the recorded order"
        if tuple(feature_names) == F.FEATURE_NAMES else
        "recorded feature names do not match this code"))

    checks.append(Check(
        "no_training_serving_skew", not skew.get("skew_detected", True),
        skew.get("verdict", "no skew report supplied")))

    cand = candidate_metrics.get(metric)
    inc = incumbent_metrics.get(metric) if incumbent_metrics else None
    if cand is None:
        checks.append(Check("offline_metric_present", False,
                            f"candidate has no {metric}"))
    elif inc is None:
        checks.append(Check(
            "beats_incumbent", True,
            f"no incumbent; {metric} {cand:.4f} accepted as the first model"))
    else:
        required = inc * (1 + min_improvement)
        checks.append(Check(
            "beats_incumbent", cand >= required,
            f"{metric} {cand:.4f} against an incumbent {inc:.4f}; "
            f"needs {required:.4f} "
            f"({'+' if cand >= inc else ''}{100 * (cand - inc) / inc:.2f}% "
            f"vs a {100 * min_improvement:.1f}% bar)"))

    if measured_p99_ms is None:
        checks.append(Check("latency_budget", False,
                            "no latency measurement supplied"))
    else:
        checks.append(Check(
            "latency_budget", measured_p99_ms <= budget_p99_ms,
            f"p99 {measured_p99_ms:.1f} ms against a {budget_p99_ms:.0f} ms budget"))

    promoted = all(c.passed for c in checks)
    failed = [c for c in checks if not c.passed]
    verdict = (
        "PROMOTED: " + "; ".join(c.detail for c in checks)
        if promoted else
        "REFUSED — " + "; ".join(f"{c.name}: {c.detail}" for c in failed))
    return GateResult(promoted=promoted, checks=checks, verdict=verdict)
