# railrank

A rail journey ranking service, and the reason its training labels are wrong.

A traveller searches a route, sees ten or so journeys with fares and durations,
clicks one or two, books at most one. A ranker decides the order. The obvious
way to train it is on the click log — and the click log is not a record of what
travellers preferred. It records what they preferred **among the things the
previous ranker put near the top**, which is a different thing, and training on
it teaches a model to reproduce the ranker you are replacing.

This repository builds the whole path: a feature pipeline shared by training
and serving, a LambdaMART ranker trained on position-debiased clicks, a
containerised endpoint under a latency budget, a promotion gate that can
refuse, drift monitors, and a split-ticket fare optimiser. Then it scores all
of it against an answer key the generator wrote down.

```bash
pip install -e ".[dev]"
pytest -q                                          # 110 tests
python -m railrank.train                           # fit, write artefacts/current
docker build -t railrank:local . && docker run -d -p 8080:8080 railrank:local
python -m railrank.evaluate --searches 24000 \
       --http http://127.0.0.1:8080/rank           # 27 checks, the committed run
python tools/check_readme.py                       # this file against that report
```

**27 of 27 checks pass.** Every number below came from
`reports/scorecard.json`, produced by that command — and
`tools/check_readme.py` fails the build if any figure in this file is not in
that report, or cannot be recomputed from it. Without a container to talk to,
`python -m railrank.evaluate` runs the other 26.

---

## The headline result

Across ten candidate rankers of varying quality — random, three sorts, the
incumbent, two learned models, and the oracle with three levels of noise —
which offline metric would have picked the right one?

| offline metric | rank correlation with the truth |
|---|---:|
| NDCG@5 on held-out clicks | **+0.624** |
| DCG@5, inverse propensity weighted | **+0.988** |

The naive metric is what almost everybody reports, and it is close to a coin
toss at picking between models. It ranks the **incumbent** above a ranker the
oracle says is far better, because the clicks it is scored on were produced by
the incumbent.

The correction uses nothing exotic: 14% of searches are shown in a randomised
order, and in that slice position and preference are independent, so the
click-rate-by-position curve measures examination and nothing else. Divide by
it and the metric agrees with the truth almost perfectly.

The same correction applied to the **labels** rather than the metric moves the
model itself:

| model | NDCG@5 vs truth | MRR vs truth | NDCG@5 on clicks |
|---|---:|---:|---:|
| the incumbent (a price sort) | 0.6886 | 0.3414 | 0.2879 |
| LambdaMART on raw clicks | 0.9432 | 0.6667 | **0.3074** |
| LambdaMART on corrected labels | **0.9528** | **0.6992** | 0.3031 |

**The click metric prefers the worse model.** Judge by held-out clicks and you
reject the better ranker.

---

## Personalisation, measured rather than assumed

A business traveller will pay to save an hour; a leisure traveller will spend
an hour to save fifteen pounds. The same list should not be ranked the same way
for both — and that is a claim, so it gets a number.

| ranker | NDCG@5 vs truth | MRR vs truth |
|---|---:|---:|
| without the traveller-type features | 0.9402 | 0.6908 |
| with them | **0.9528** | **0.6992** |

**+1.34%.** Two indicator columns, leisure as the baseline; a third would make
them linearly dependent and tell the model nothing extra.

---

## The leak

`final_load_factor` and `load_factor_at_search` are the same column in the same
table. One is read when the search happens; the other after the train has gone.
A booking takes a seat, so the second carries the outcome it is being used to
predict — and joining it without an as-of filter is one missing `WHERE` clause.

| | NDCG@5 on clicks | NDCG@5 vs truth |
|---|---:|---:|
| honest features | 0.3115 | **0.9442** |
| with the leak | **0.4106** | 0.8822 |

The leaky model looks **32% better on the only offline metric a real team has**,
and is worse on the truth. Then, because the serving request schema has no
field for a column that does not exist yet at search time, it goes to
production without it: **0.8822 collapses to 0.7818, a drop of 11.4%.**

That sequence is why leaks survive review. The number that would have been used
to decide went up.

---

## The two detectors

**Point in time.** `RouteHistory` is built from daily aggregates and every
lookup takes an `as_of` date, so a search on the 3rd cannot see the 4th. The
tempting alternative — compute the aggregate once over the whole table and join
it — is one line shorter and changes **100% of rows**, by a mean of 0.28.

**Training and serving skew.** One function computes features for both paths,
which is necessary and not sufficient: the same function gives different
answers if the two callers pass different configuration. So the config travels
inside the model artefact, and `skew_report` replays logged requests through
both paths and compares field by field.

| planted fault | detected |
|---|---|
| identical configurations | silent — 2,158 rows agree on all 23 features |
| training in UTC, serving on local time | `advance_days`, off by exactly 1.0 |
| a different fill for a missing seat count | `seats_remaining`, 375 of 2,158 rows |

The first row is the one that matters. A detector that always fires gets
switched off.

---

## Serving

```
in-process   p50 0.714 ms  p99 1.153 ms   over 600 searches
over HTTP    p50 3.254 ms  p99 8.215 ms   420 requests, all 200, against the container
budget       p99 25 ms
```

Both numbers are reported because they answer different questions. The first is
what a model change moves; the second is what a caller sees. Quoting only the
first is how a service ships with a 2 ms model and a 300 ms p99.

These four are wall-clock measurements on one laptop and the only figures in
this file that move between runs: across five runs of the same command the
in-process p99 ranged from 1.1 ms to 2.6 ms and the HTTP p99 from 5.2 ms to
8.3 ms. The four above are the committed run, exactly, rather than the best
one — `tools/check_readme.py` reads them straight out of
`reports/scorecard.json` and fails if they drift apart. Everything else in this
file is bit-identical run to run, and a test fails if that stops being true.

The process **refuses to start** if the model's recorded feature order does not
match the code. A model trained before a feature was inserted scores against
the wrong columns and raises nothing at all; comparing the names turns a wrong
answer into a failed deploy, which is the only failure mode anyone notices.

---

## The promotion gate

Promotion is a function with conditions, not a button. It reads the
**corrected** offline metric, for the reason at the top of this file.

| condition | what it refused, in the scorecard |
|---|---|
| feature contract matches the code | refused a model recording a feature this code no longer computes |
| no training/serving skew | refused on the planted `advance_days` skew |
| beats the incumbent by 0.2% | refused when handed the incumbent as the candidate |
| p99 inside the budget | refused when the budget was halved |

All four refusal paths are exercised, because a gate that can only approve is a
formality — and a condition nobody has ever watched fail is a comment.

---

## The split-ticket optimiser

Two tickets for the two halves of a journey are often cheaper than one for the
whole of it, and it is legal as long as the train calls where the tickets
change. A fare is a price for a market, not a price for a distance, so
`fare(A,C)` is under no obligation to be less than `fare(A,B) + fare(B,C)`.

Written as a graph it is a shortest path: calling points are nodes, purchasable
fares are edges, and the cheapest combination is a dynamic program over a
topologically ordered DAG.

```
600 journeys, solver against exhaustive enumeration     0 disagreements
splitting is worth doing on                             14.2% of journeys
median saving when it is                                12.82%  (7.00 GBP)
```

The brute force exists because a solver returning a plausible number is
indistinguishable from one returning the right one. It enumerates every subset
of intermediate calling points, and **refuses** rather than silently taking
minutes when the instance is too large.

---

## Monitoring, including what it cannot see

Two operators ship a client release that starts reporting a constant where an
availability count used to be. No nulls, no schema change, no row-count change,
every value a legal integer — so freshness and null-rate checks stay green
while a feature the model reads moves underneath it.

```
seats_remaining              PSI 0.54358  flagged, with an owner and a runbook
a random split of one period PSI 0.00087  silent, as a control must be
price_gbp                    PSI 0.0477  NOT flagged
```

That last line is reported rather than hidden. Fares trend upward by a planted
0.42% a day — about 21% over the period — and PSI calls it *stable*. PSI
compares binned shares, so a slow location shift inside a wide distribution
moves very little mass across quantile edges. **A monitor that only computes
PSI will miss one of the two commonest ways a feature goes wrong.** Catching a
trend needs a trend test.

---

## What the scorecard caught

Each of these was a defect in this code, found by a check rather than by
reading.

**An inverse propensity correction that could not correct anything.** With one
click per search, scaling the single positive label scales the ideal DCG by the
same factor and cancels: NDCG normalises within the query. The corrected model
scored identically to the uncorrected one, to four decimal places. Real sessions
compare several journeys, so the generator now does too — and the correction
began to bite.

**The correction cannot go where it obviously belongs.** XGBoost's ranking
objectives take `weight` per query *group*, not per document, so a per-document
IPS weight raises `Size of weight must equal to the number of query groups`. It
has to go into the label, and only with linear gains — with the default
`2**r - 1`, an inverse-propensity label is not an unbiased estimator of
anything.

**An IPS-corrected NDCG that ranked models identically to the naive one**, for
the same normalisation reason. The unbiased estimator is un-normalised DCG.

**A solver that assumed the train stopped where you boarded.** `cheapest_split`
unioned the journey's endpoints into the calling set, so it happily priced a
journey the passenger could not make. Caught by a test asserting it should
raise.

**A seed that was not a seed.** The optimiser stage seeded each route's fare
table with `hash(route_key)`. Python randomises string hashing per process
unless `PYTHONHASHSEED` is set, so two runs of identical code reported 12.0% and
then 13.2% of journeys worth splitting — and this file quoted one of them as
reproducible. `zlib.crc32` now, and a test runs the stage in three subprocesses
and fails unless all three agree. Every figure in this README except the four
wall-clock latencies is checked against `reports/scorecard.json` in CI, which is
how the discrepancy became visible.

**A fare table that priced 290 miles at £13**, by compounding a lognormal tail
with a competitive discount. It made the optimiser look brilliant by making the
inputs impossible. There is a floor now.

**A leak too weak to demonstrate a leak.** The first version raised the final
load factor with appeal only; the model barely used it and removing it cost
nothing. A booking takes a seat, which is both physically true and what makes
the column carry the label.

---

## Layout

```
railrank/
  journeys.py     the generator, the preference model, and planted()
  features.py     ONE compute() for training and serving; skew and as-of checks
  ranker.py       LambdaMART, propensities, IPS labels, oracle and click metrics
  registry.py     the servable artefact, MLflow tracking, the promotion gate
  service.py      FastAPI /rank and /health
  latency.py      in-process and HTTP percentiles against a stated budget
  monitoring.py   PSI feature, prediction and segment drift, with owners
  optimise.py     split-ticket shortest path, and the brute force that checks it
  train.py        one command: fit, evaluate, write the artefact, print the gate
  evaluate.py     the scorecard
tools/            check_readme.py: every figure in this file, against the report
infra/            Terraform: ECR, ECS Fargate, logs, and an alarm at the same
                  budget the gate enforces
tests/            110 tests; several encode the bugs listed above
```

**The Terraform has never been applied.** CI runs `terraform fmt -check` and
`terraform validate`; nothing has been run against an AWS account, because I am
not paying to host a portfolio project. A module the tooling agrees is
well-formed is a real artefact; a claimed deployment that does not exist is one
question away from falling apart.

**The data is simulated**, so that ground truth is known exactly — an
instrument cannot be calibrated against data whose true values are unknown. The
scale is 24,000 searches and 263,949 options on a laptop. I have not worked at
production volume and do not claim to.

---

MIT. Built by Michael Samuel.
