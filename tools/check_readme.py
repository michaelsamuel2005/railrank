"""
check_readme.py — every figure in the README must come from the scorecard.

A README is documentation that goes stale silently. This one quotes about forty
numbers, all of them produced by `python -m railrank.evaluate`, and the usual
failure is that the code improves, the report is regenerated, and the prose
keeps yesterday's figure. Nothing complains, because prose has no tests.

So this is the test. It pulls every number out of the README and asserts that
each one is either

  * a figure in `reports/scorecard.json`, matched at the precision the README
    prints it: 0.3414 in the prose is satisfied by 0.34145 in the report,
    because that is what rounding to four places gives, and is not satisfied by
    0.3391;
  * a figure DERIVED from the report by a stated formula, which this file
    recomputes -- so the arithmetic in a sentence like "33% better" is checked
    too, not taken on trust;
  * or EXEMPT: a constant of the project, a wall-clock timing, or a number in
    a sentence rather than a measurement. Every exemption carries a reason.

The exemption list is deliberately short and deliberately visible. A checker
with a growing list of exemptions is a checker being switched off one line at
a time.

    python tools/check_readme.py            # non-zero exit on a stale figure

It runs in CI after the scorecard, which is how "12.0% of journeys" was caught
still sitting in the prose after a seeding fix moved it to 14.2%.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── figures that are not measurements ───────────────────────────────────────
# Keyed by the string as the README prints it. The reason is the point of the
# entry: an exemption without one is indistinguishable from an oversight.
EXEMPT: dict[str, str] = {
    "25": "the latency budget, a constant in registry.py",
    "1.1": "the low end of the in-process p99 range observed over five runs",
    "2.6": "the high end of that range",
    "5.2": "the low end of the HTTP p99 range observed over five runs",
    "8.3": "the high end of that range",
    "600": "searches measured, a parameter of the run",
    "420": "requests measured over HTTP",
    "0.2": "the promotion gate's minimum improvement, a constant",
    "21": "0.42%/day compounded over 45 days; arithmetic on two constants",
    "2": "prose: 'a 2 ms model and a 300 ms p99'",
    "300": "prose: an illustrative bad p99",
    "1": "prose, and the '2**r - 1' gain formula",
    "5": "NDCG@5 / DCG@5 / precision@k: a cutoff, not a measurement",
    "13": "the withdrawn 290-miles-at-13-pounds fare, quoted as withdrawn",
    "290": "miles, in that same historical sentence",
    "12.0": "a superseded figure, quoted in the bug write-up as superseded",
    "13.2": "the other superseded figure in the same write-up",
    "32": "part of the identifier zlib.crc32",
    "8080": "the published port",
    "0": "'0 disagreements'; also in the report, but as an int",
}

# ── figures the README computes from the report ──────────────────────────────
# Each is (formula description, how to get it from the parsed report). Checked
# rather than exempted, so a sentence that does its own arithmetic is held to
# the same standard as one that quotes a number.


def _obs(report: dict, check: str) -> str:
    for c in report["checks"]:
        if c["check"] == check:
            return c["observed"]
    raise KeyError(check)


def _floats(text: str) -> list[float]:
    return [float(m) for m in re.findall(r"\d+\.\d+|\d+", text)]


DERIVED: dict[str, tuple[str, object]] = {
    "32": ("how much better the leak looks on clicks: 0.4106/0.3115 - 1",
           lambda r: 100 * (_floats(_obs(r, "leak_inflates_the_metric_you_can"
                                            "_actually_measure"))[2]
                            / _floats(_obs(r, "leak_inflates_the_metric_you_can"
                                              "_actually_measure"))[1] - 1)),
    "263949": ("options generated across all searches",
               lambda r: r["n_options"]),
    "23": ("the feature contract's length",
           lambda r: r["n_features"]),
    "24000": ("searches generated", lambda r: r["n_searches"]),
    # The four wall-clock figures. Exempting them would let the README quote
    # whichever run flattered the project; reading them out of the committed
    # report instead means the prose and the report move together or not at all.
    "0.714": ("in-process p50, from the committed run",
              lambda r: r["stages"]["latency"]["inprocess"]["p50_ms"]),
    "1.153": ("in-process p99, from the committed run",
              lambda r: r["stages"]["latency"]["inprocess"]["p99_ms"]),
    "3.254": ("HTTP p50, from the committed run",
              lambda r: r["stages"]["latency"]["http"]["p50_ms"]),
    "8.215": ("HTTP p99, from the committed run",
              lambda r: r["stages"]["latency"]["http"]["p99_ms"]),
    "27": ("checks in the scorecard", lambda r: r["n_checks"]),
    "26": ("the checks that do not need a running container",
           lambda r: r["n_checks"] - 1),
    "110": ("tests pytest can collect", lambda r: _collected_tests()),
}


def _collected_tests() -> int:
    """Ask pytest, rather than trusting a number typed into the README.

    Collection only -- it imports the test modules and counts, which takes
    about a second and does not run anything.
    """
    import subprocess
    import sys
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=ROOT, capture_output=True, text=True).stdout
    return sum(int(m) for m in re.findall(r"^tests/\S+\.py: (\d+)$",
                                          out, re.M))


def readme_numbers(text: str) -> list[tuple[str, str]]:
    """Every decimal literal in the README, with the line it sits on.

    A digit run must not be glued to a letter on either side, or `crc32`,
    `MiniLM-L6-v2` and `p99` each contribute a phantom figure. `p99.` also has
    to not yield `99.`, hence the optional fractional part requiring digits
    after the point.
    """
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line in text.splitlines():
        for m in re.finditer(r"(?<![A-Za-z0-9.])(\d[\d,]*(?:\.\d+)?)(?![A-Za-z0-9])",
                             line):
            n = m.group(1).replace(",", "")
            if n not in seen:
                seen.add(n)
                found.append((n, line.strip()))
    return found


def report_numbers(node, out: list[float]) -> list[float]:
    """Every number anywhere in the report, whatever the nesting."""
    if isinstance(node, bool):
        return out
    if isinstance(node, (int, float)):
        out.append(float(node))
    elif isinstance(node, str):
        out.extend(_floats(node))
    elif isinstance(node, dict):
        for k, v in node.items():
            out.extend(_floats(k))
            report_numbers(v, out)
    elif isinstance(node, list):
        for v in node:
            report_numbers(v, out)
    return out


def main() -> int:
    readme = (ROOT / "README.md").read_text()
    report = json.loads((ROOT / "reports" / "scorecard.json").read_text())
    pool = report_numbers(report, [])

    checked = derived = stale = 0
    for n, line in readme_numbers(readme):
        dp = len(n.split(".")[1]) if "." in n else 0
        want = float(n)

        if n in DERIVED:
            desc, fn = DERIVED[n]
            got = float(fn(report))
            derived += 1
            if round(got, dp) != want:
                stale += 1
                print(f"WRONG  {n:>10}  {desc} recomputes to {got:.4f}")
                print(f"       {line[:110]}")
            continue

        if n in EXEMPT:
            continue

        checked += 1
        # Satisfied if any figure in the report rounds to what the README
        # prints. Rounding at the README's precision, not the report's, is the
        # point: the prose is what a reader is asked to believe.
        if not any(round(v, dp) == want for v in pool):
            stale += 1
            print(f"STALE  {n:>10}  no figure in reports/scorecard.json "
                  f"rounds to this")
            print(f"       {line[:110]}")

    # An exemption for a figure the README no longer prints is dead weight that
    # would silently cover a future figure of the same value. Same for a
    # DERIVED formula nothing uses.
    printed = {n for n, _ in readme_numbers(readme)}
    for name, table in (("EXEMPT", EXEMPT), ("DERIVED", DERIVED)):
        for key in table:
            if key not in printed:
                stale += 1
                print(f"UNUSED {key:>10}  in {name}, but the README no longer "
                      f"prints it -- remove the entry")

    if stale:
        print(f"\n{stale} figure(s) wrong. Either the report is out of date or "
              f"the prose is.\nRegenerate with: python -m railrank.evaluate "
              f"--searches 24000 --http http://127.0.0.1:8080/rank")
        return 1
    print(f"all {checked} README figures verified against reports/scorecard.json, "
          f"{derived} recomputed from it, {len(EXEMPT)} exempt with a stated reason")
    return 0


if __name__ == "__main__":
    sys.exit(main())
