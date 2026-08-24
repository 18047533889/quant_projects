# -*- coding: utf-8 -*-
"""R30 final artifact generation: binds current HEAD, canonical-set digest,
semantic-contract digest, and writes the acceptance report + manifest."""
from __future__ import annotations

import sys
import json
import hashlib
import csv
import subprocess
import os
import datetime
from pathlib import Path

sys.path.insert(0, ".")
from factor_engine.cleaned_operators import load_all
load_all()

from factor_engine.cleaned_operators.registry import OperatorRegistry as R
from factor_engine.cleaned_operators.operator_surface import classify_canonical
from factor_engine.cleaned_operators.tombstones import ALL_TOMBSTONED_NAMES
from factor_engine.cleaned_operators.operator_spec import _infer_status
from factor_engine.cleaned_operators.operator_policy import infer_operator_policy
from factor_engine.cleaned_operators.registry import _contract_hash

FE = Path(__file__).resolve().parents[1]
OUT = FE / "evidence" / "factor_engine" / "r30"
OUT.mkdir(parents=True, exist_ok=True)

def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(FE)).decode().strip()
    except Exception:
        return "unknown"

def _canonical_digest() -> str:
    h = hashlib.sha256()
    for c in sorted(R.list_canonical()):
        h.update(c.encode())
    return h.hexdigest()[:16]

def _semantic_digest() -> str:
    h = hashlib.sha256()
    for c in sorted(R.list_canonical()):
        op = R.get(c, mode="any")
        if op is not None:
            h.update(_contract_hash(op).encode())
    return h.hexdigest()[:16]

head = _git_head()
canon_digest = _canonical_digest()
sem_digest = _semantic_digest()

# ---- R30_HEAD.json ----
json.dump({"git_sha": head, "canonical_set_digest": canon_digest,
           "semantic_contract_digest": sem_digest, "timestamp": "2026-08-10"},
          open(OUT / "R30_HEAD.json", "w"), indent=2, sort_keys=True)

# ---- R30_ALIAS_AUDIT.csv ----
with open(OUT / "R30_ALIAS_AUDIT.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["alias", "canonical", "tombstoned"])
    for a, c in sorted(R._aliases.items()):
        w.writerow([a, c, a in ALL_TOMBSTONED_NAMES or c in ALL_TOMBSTONED_NAMES])

# ---- R30_DELETION_MANIFEST.csv ----
with open(OUT / "R30_DELETION_MANIFEST.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["name", "risk_class", "removed_since", "had_executable"])
    from factor_engine.cleaned_operators.tombstones import TOMBSTONES
    for name, tb in sorted(TOMBSTONES.items()):
        w.writerow([name, tb.risk_class, tb.removed_since,
                    name in R._operators])

# ---- R30_TOMBSTONE_MANIFEST.json ----
json.dump({name: {"risk_class": tb.risk_class, "removed_since": tb.removed_since,
                  "reason": tb.reason, "replacement": tb.replacement}
           for name, tb in sorted(TOMBSTONES.items())},
          open(OUT / "R30_TOMBSTONE_MANIFEST.json", "w"), indent=2, sort_keys=True)

# ---- R30_ALIAS_AUDIT / PRODUCTION_ADMISSION_AUDIT.json ----
admission = []
for c in sorted(R.list_canonical()):
    op = R.get(c, mode="any")
    if op is None:
        continue
    pol = infer_operator_policy(op, canonical=c)
    st = _infer_status(R._catalog.get(c, {}))
    from factor_engine.cleaned_operators.operator_spec import _compute_allow_in_production
    allow = _compute_allow_in_production(c, status=st, pit_safe=pol.pit_safe,
                                         shape_preserving=pol.shape_preserving)
    admission.append({"canonical": c, "surface": classify_canonical(c),
                      "status": str(st), "pit_safe": bool(pol.pit_safe),
                      "allow_in_production": bool(allow)})
json.dump(admission, open(OUT / "R30_PRODUCTION_ADMISSION_AUDIT.json", "w"), indent=2, sort_keys=True)

# ---- R30_PARAMETER_ROLE_AUDIT.csv ----
role_rows = []
for c in sorted(R.list_canonical()):
    if classify_canonical(c) not in ("daily", "extended"):
        continue
    op = R.get(c, mode="any")
    if op is None:
        continue
    meta = getattr(op, "metadata", None)
    if meta is None:
        continue
    from factor_engine.cleaned_operators.operator_spec import _infer_panel_params
    declared = tuple(getattr(meta, "panel_params", None) or ())
    panels = set(declared) or set(_infer_panel_params(op, meta, R._catalog.get(c, {})))
    specs = getattr(meta, "param_specs", None) or {}
    names = tuple(getattr(meta, "param_names", None) or ())
    for p in names:
        if p in panels:
            continue
        spec = specs.get(p)
        if spec is None:
            continue
        role = getattr(spec, "param_role", None)
        role_rows.append({"canonical": c, "param": p,
                          "role": getattr(role, "value", "NONE"),
                          "role_source": getattr(spec, "role_source", "authored")})
with open(OUT / "R30_PARAMETER_ROLE_AUDIT.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["canonical", "param", "role", "role_source"])
    w.writeheader()
    w.writerows(role_rows)

# ---- R30_ARTIFACT_MANIFEST.json ----
artifacts = sorted(p.name for p in OUT.iterdir() if p.is_file())
json.dump({"git_sha": head, "canonical_set_digest": canon_digest,
           "semantic_contract_digest": sem_digest, "artifacts": artifacts},
          open(OUT / "R30_ARTIFACT_MANIFEST.json", "w"), indent=2, sort_keys=True)

print(f"HEAD: {head}")
print(f"canonical-set digest: {canon_digest}")
print(f"semantic-contract digest: {sem_digest}")
print(f"artifacts written to {OUT}: {len(artifacts)}")
