"""
train.py — fit a ranker and write a servable artefact.

One command, because the step between "a model exists somewhere" and "a model
is in a directory the service can load" is where most of the mistakes live.
This does the whole of it: generate or read the data, fit the feature
configuration from the training set, estimate propensities from the randomised
slice, train, evaluate on a time-ordered split, write the artefact, log the run
and print the gate's decision.

    python -m railrank.train --out artefacts/current
    python -m railrank.train --labels raw     # the uncorrected comparison
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import features as F
from . import journeys as J
from . import latency as L
from . import ranker as R
from . import registry as REG


def run(n_searches: int = 24_000, out: str = "artefacts/current",
        labels: str = "ips", n_rounds: int = R.N_ROUNDS,
        train_fraction: float = 0.7, log_mlflow: bool = True,
        incumbent: str | None = None) -> dict:
    searches, options = J.generate(n_searches=n_searches)

    # The fill is a statistic of the TRAINING data and is then frozen into the
    # artefact. Computing it over everything would use the test period, which
    # is a small leak and exactly the kind nobody notices.
    n_train_searches = int(len(searches) * train_fraction)
    train_ids = {s.search_id for s in
                 sorted(searches, key=lambda s: s.searched_at)[:n_train_searches]}
    cfg = F.FeatureConfig(seats_missing_fill=F.seats_missing_fill_from(
        [o for o in options if o.search_id in train_ids]))

    X, g, meta, history = F.build_matrix(searches, options, cfg)
    tr, trg, trm, te, teg, tem = R.temporal_split(g, meta, train_fraction)
    propensities = R.estimate_propensities(trm)

    y = None if labels == "raw" else R.ips_labels(trm, propensities)
    model = R.train(X[tr], trg, trm, cfg, labels=y, n_rounds=n_rounds)
    metrics = R.evaluate(model.score(X[te]), teg, tem, propensities=propensities)
    baseline = R.evaluate(R.logging_policy_scores(tem), teg, tem,
                          propensities=propensities)

    artefact = REG.save_artefact(
        out, model, propensities, history, metrics,
        notes=f"LambdaMART on {labels} labels, {n_rounds} rounds, "
              f"{len(trg):,} training searches")

    # Load it back and time it through the serving path, not the training one.
    loaded = REG.load_artefact(artefact)
    by_search: dict = {}
    for o in options:
        by_search.setdefault(o.search_id, []).append(o)
    lat = L.measure_inprocess(loaded, searches[:800], by_search, n=400)
    skew = F.skew_report(searches[:300], options, cfg, loaded.config, history,
                         sample=150)

    incumbent_metrics = baseline
    if incumbent and Path(incumbent).exists():
        incumbent_metrics = REG.load_artefact(incumbent).card["offline_metrics"]

    gate = REG.promotion_gate(metrics, incumbent_metrics, skew, lat["p99_ms"])

    info = {"logged": False}
    if log_mlflow:
        info = REG.log_run(
            artefact,
            {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            | {"p99_ms": lat["p99_ms"], "p50_ms": lat["p50_ms"]},
            {"labels": labels, "n_rounds": n_rounds,
             "train_searches": len(trg), "test_searches": len(teg),
             "feature_config": cfg.fingerprint(),
             "features": len(F.FEATURE_NAMES)},
            run_name=f"train-{labels}")

    return {"artefact": str(artefact), "labels": labels,
            "train_searches": len(trg), "test_searches": len(teg),
            "metrics": metrics, "baseline": baseline,
            "latency": {k: lat[k] for k in ("p50_ms", "p95_ms", "p99_ms", "n")},
            "skew": skew["verdict"], "gate": gate.to_dict(), "mlflow": info}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--searches", type=int, default=24_000)
    ap.add_argument("--out", default="artefacts/current")
    ap.add_argument("--labels", choices=("ips", "raw"), default="ips")
    ap.add_argument("--rounds", type=int, default=R.N_ROUNDS)
    ap.add_argument("--incumbent", default=None,
                    help="an existing artefact directory to compare against")
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args(argv)

    out = run(n_searches=a.searches, out=a.out, labels=a.labels,
              n_rounds=a.rounds, log_mlflow=not a.no_mlflow,
              incumbent=a.incumbent)

    m, b = out["metrics"], out["baseline"]
    print(f"trained on {out['train_searches']:,} searches, tested on "
          f"{out['test_searches']:,}")
    print(f"  NDCG@5 against the oracle   {b['ndcg@5_oracle']:.4f} incumbent "
          f"-> {m['ndcg@5_oracle']:.4f}")
    print(f"  MRR against the oracle      {b['mrr_oracle']:.4f} -> "
          f"{m['mrr_oracle']:.4f}")
    print(f"  corrected offline metric    {b['dcg@5_ips']:.4f} -> "
          f"{m['dcg@5_ips']:.4f}")
    print(f"  latency p50 {out['latency']['p50_ms']} ms, p99 "
          f"{out['latency']['p99_ms']} ms")
    print(f"  {out['skew']}")
    print(f"  artefact -> {out['artefact']}")
    if out["mlflow"].get("logged"):
        print(f"  mlflow run {out['mlflow']['run_id'][:12]} "
              f"({out['mlflow'].get('registered_model')} "
              f"v{out['mlflow'].get('version')})")
    print()
    print(out["gate"]["verdict"])
    return 0 if out["gate"]["promoted"] else 1


if __name__ == "__main__":
    sys.exit(main())
