#!/usr/bin/env python3
"""Run three finance-boundary mutations in isolated temporary source copies."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


CASES = (
    {
        "id": "empty_integrity_gate_all_true",
        "packages": ("factor_assets",),
        "file": "factor_assets/profiling/health_card.py",
        "old": "and all(g.passed is True and bool(g.evidence_ref.strip()) for g in gates)",
        "new": "and True  # QAT04 mutant: deleted per-gate check",
        "probe": """
from factor_assets.profiling.health_card import build_health_card
from factor_assets.profiling.dimensions import DimensionGradeArtifact, HEALTH_DIMENSIONS, EVIDENCE_TIER_MISSING
from factor_assets.profiling.policies import get_health_policy
p=get_health_policy()
dims=[DimensionGradeArtifact('f','e',d,(),None,None,(),False,(),EVIDENCE_TIER_MISSING,(),p.policy_id,p.policy_version) for d in HEALTH_DIMENSIONS]
card = build_health_card(factor_definition_id='f', evaluation_ref='e', dimension_grades=dims, integrity_gates={})
assert card.admission_relevant.hard_gates_passed is False
""",
    },
    {
        "id": "fp_max_lag_alias_deleted",
        "packages": ("factor_preprocess", "factor_engine", "data_access"),
        "file": "factor_preprocess/factor_preprocess/adapters/fe_operator.py",
        "old": '_PARAM_ALIASES = {"max_lag": "max_periods"}',
        "new": "_PARAM_ALIASES = {}  # QAT04 mutant: deleted public parameter mapping",
        "probe": """
import pandas as pd
from factor_preprocess.registry.transforms import get_default_registry
frame = pd.DataFrame({'date':[1,2,3,4], 'asset_id':[1,1,1,1], 'value':[1.0,float('nan'),float('nan'),4.0]})
op = get_default_registry().get_execution('forward_fill')
out = op(values=frame, max_lag=1)
assert out.isna().iloc[2]
assert op.effective_parameters == {'max_periods': 1}
""",
    },
    {
        "id": "gpu_factor_validity_staging_deleted",
        "packages": ("quant_evaluator",),
        "file": "quant_evaluator/runtime/gpu_executor.py",
        "old": "if factor_batch.validity is not None:\n                    chunk = np.where(factor_batch.validity[:, :, start:stop], chunk, np.nan)",
        "new": "if factor_batch.validity is not None:\n                    pass  # QAT04 mutant: deleted validity staging",
        "probe": """
import numpy as np
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime.evaluator import evaluate
t,n=40,20
rng=np.random.default_rng(41)
base=rng.normal(size=(t,n,1)); validity=np.ones_like(base,dtype=bool); validity[:,19,0]=False
clean=base.copy(); clean[:,19,0]=np.nan
poison=base.copy(); poison[:,19,0]=np.where(np.arange(t)%2,1e300,-1e300)
ta=AxisRef('time','int64',t,np.arange(t)); aa=AxisRef('asset','int64',n,np.arange(n))
labels=LabelBundle('ret',rng.normal(size=(t,n)),1,decision_time=tuple(range(t)),label_start_time=tuple(range(1,t+1)),label_end_time=tuple(range(2,t+2)))
def run(x):
    b=FactorBatch(('f',),ta,aa,x,validity=validity)
    return evaluate(b,labels,metrics=['rank_ic'],backend='cuda').get_metric('rank_ic','f').value
np.testing.assert_allclose(run(poison),run(clean),rtol=0,atol=1e-12,equal_nan=True)
""",
    },
)


def _run_probe(repo: Path, root: Path, probe: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    paths = [str(root / "factor_optimizer"), str(root / "factor_preprocess"), str(root)]
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["PYTHONNOUSERSITE"] = "1"
    return subprocess.run([sys.executable, "-c", probe], cwd=root, env=env,
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    records = []
    for case in CASES:
        with tempfile.TemporaryDirectory(prefix=f"qat04-{case['id']}-") as raw:
            root = Path(raw)
            for package in case["packages"]:
                shutil.copytree(repo / package, root / package)
            baseline = _run_probe(repo, root, case["probe"])
            target = root / case["file"]
            source = target.read_text()
            occurrences = source.count(case["old"])
            if occurrences != 1:
                raise RuntimeError(f"{case['id']}: expected one mutation site, found {occurrences}")
            target.write_text(source.replace(case["old"], case["new"], 1))
            mutant = _run_probe(repo, root, case["probe"])
            records.append({
                "mutation_id": case["id"], "source_file": case["file"],
                "baseline_exit_code": baseline.returncode,
                "mutant_exit_code": mutant.returncode,
                "killed": baseline.returncode == 0 and mutant.returncode != 0,
                "baseline_output": baseline.stdout[-2000:],
                "mutant_output": mutant.stdout[-4000:],
            })
    payload = {
        "schema_version": 1, "suite": "QAT-04-targeted-mutations",
        "python": sys.version, "repo": str(repo), "mutations": records,
        "status": "PASS" if all(r["killed"] for r in records) else "FAIL",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
