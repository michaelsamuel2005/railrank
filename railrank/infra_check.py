"""
infra_check.py — a structural check on the Terraform, for when Terraform is absent.

`terraform validate` is the right tool and CI runs it. This exists because the
environment this project was developed in cannot download the Terraform binary,
and a module that has never had anything run against it is a module nobody
should believe. So the checks below are the subset that can be done by reading
the files: every variable referenced is declared, every resource referenced
exists, the braces balance, and nothing that looks like a credential or an
account number is committed.

It is deliberately not a parser. It will not catch a type error or an invalid
provider argument, and it says so rather than implying it validates anything.
The division of labour is stated in the report: this runs everywhere, and
`terraform validate` runs in CI where the binary is available.
"""

from __future__ import annotations

import re
from pathlib import Path

SECRET_PATTERNS = {
    "aws access key id": r"\bAKIA[0-9A-Z]{16}\b",
    "twelve-digit account id": r"(?<![\w.])\d{12}(?![\w.])",
    "inline secret assignment": r"(?i)\b(secret|password|private_key)\s*=\s*\"[^\"$]{8,}\"",
}


def check(directory: Path | str = "infra") -> dict:
    directory = Path(directory)
    files = sorted(directory.glob("*.tf"))
    if not files:
        return {"checked": False, "reason": f"no .tf files in {directory}"}

    text = "\n".join(f.read_text() for f in files)
    problems: list[str] = []

    # Braces. A module that does not balance will not parse, and this is the
    # one class of error a reader misses and a machine never does.
    opens, closes = text.count("{"), text.count("}")
    if opens != closes:
        problems.append(f"{opens} opening braces against {closes} closing")

    declared_vars = set(re.findall(r'variable\s+"([^"]+)"', text))
    used_vars = set(re.findall(r"var\.([A-Za-z_][A-Za-z0-9_]*)", text))
    undeclared = sorted(used_vars - declared_vars)
    if undeclared:
        problems.append(f"variables used but never declared: {undeclared}")
    unused = sorted(declared_vars - used_vars)

    declared_res = set()
    for kind, name in re.findall(r'resource\s+"([^"]+)"\s+"([^"]+)"', text):
        declared_res.add(f"{kind}.{name}")
    for kind, name in re.findall(r'data\s+"([^"]+)"\s+"([^"]+)"', text):
        declared_res.add(f"data.{kind}.{name}")

    referenced = set()
    for ref in re.findall(r"(?<![\w.])((?:data\.)?aws_[a-z0-9_]+\.[A-Za-z_][A-Za-z0-9_]*)", text):
        referenced.add(ref)
    # Strip the trailing attribute: aws_ecs_cluster.this.id -> aws_ecs_cluster.this
    dangling = sorted(r for r in referenced if r not in declared_res)
    if dangling:
        problems.append(f"references with no matching block: {dangling}")

    leaked = {}
    for label, pattern in SECRET_PATTERNS.items():
        hits = re.findall(pattern, text)
        if hits:
            leaked[label] = len(hits)
    if leaked:
        problems.append(f"possible credentials committed: {leaked}")

    required = re.search(r"required_version\s*=\s*\"([^\"]+)\"", text)
    if not required:
        problems.append("no required_version, so a future Terraform can "
                        "silently reinterpret this module")

    return {
        "checked": True,
        "files": [f.name for f in files],
        "lines": text.count("\n") + 1,
        "resources": len([r for r in declared_res if not r.startswith("data.")]),
        "variables_declared": len(declared_vars),
        "variables_unused": unused,
        "outputs": len(re.findall(r'output\s+"', text)),
        "required_version": required.group(1) if required else None,
        "problems": problems,
        "ok": not problems,
        "scope": ("structural only: declarations, references, braces and "
                  "committed secrets. It does not type-check arguments or "
                  "contact a provider. `terraform validate` does that, and CI "
                  "runs it."),
    }
