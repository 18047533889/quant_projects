#!/usr/bin/env python3
"""Certify production recipes with at least one real execution path.

The historical portable corpus is still executed on Pandas + native Polars +
real DuckDB SQL.  Every registered production recipe is additionally expanded
and executed on its Pandas semantic-reference path after factor-operator evidence
has been issued.  Commit SHA is audit metadata only.
"""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
FE_ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(FE_ROOT))

def _run(test:str,env)->int:
    return subprocess.run([sys.executable,"-m","pytest","-q",test],cwd=FE_ROOT,env=env).returncode

def main()->int:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--check",action="store_true");args=parser.parse_args()
    from factor_engine.backend.recipe_evidence import CASE_PATH,VERIFIED_PATH,current_hashes,recipe_evidence_valid
    if args.check:
        if not recipe_evidence_valid(require_commit_ancestor=False):print("recipe_verified.json invalid or stale",file=sys.stderr);return 1
        # Verify the recorded per-recipe execution fingerprints are reproducible.
        from factor_engine.backend.recipe_execution_fingerprint import fingerprints_current
        recorded=json.loads(VERIFIED_PATH.read_text(encoding="utf-8")).get("execution") or {}
        errors=fingerprints_current(recorded)
        if errors:
            print("recipe execution fingerprints stale:",file=sys.stderr)
            for error in errors:print(f"- {error}",file=sys.stderr)
            return 1
        print("recipe evidence valid (one-backend production + portable three-backend subset; execution fingerprints reproducible)");return 0
    # Bootstrap the registry and its production signature overlays before
    # validating primitive evidence.  Validation imports signatures lazily, and
    # checking it against a half-initialized module can produce false circular
    # import failures in a fresh subprocess.
    from factor_engine.cleaned_operators import load_all
    load_all()
    from factor_engine.backend.evidence_provenance import evidence_artifact_valid
    from factor_engine.backend.factor_operator_evidence import factor_operator_evidence_valid
    if not evidence_artifact_valid():print("primitive evidence must be valid before recipe certification",file=sys.stderr);return 1
    if not factor_operator_evidence_valid():print("factor-operator evidence must be valid before recipe certification",file=sys.stderr);return 1
    env=dict(os.environ);env["FACTOR_ENGINE_CERTIFY_RECIPE_EVIDENCE"]="1"
    if _run("tests/operators/test_recipe_backend_certification.py",env):return 1
    if _run("tests/operators/test_recipe_production_admission_v2.py",env):return 1
    cases=json.loads(CASE_PATH.read_text(encoding="utf-8"))
    from factor_engine.factor_recipes import FactorRecipeRegistry
    recipes=sorted(FactorRecipeRegistry.list_names(status="production"));portable=sorted(cases.get("recipes") or [])
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=FE_ROOT).decode().strip()
    certified={name:["pandas_numpy"] for name in recipes}
    for name in portable:
        if name in certified:certified[name]=["pandas_numpy","polars_long_native","duckdb_real_sql"]
    from factor_engine.backend.recipe_execution_fingerprint import record_recipe_execution_fingerprints
    execution = record_recipe_execution_fingerprints()
    if any("error" in fp for fp in execution.values()):
        failed = [name for name, fp in execution.items() if "error" in fp]
        print("recipe execution failed for: " + ", ".join(sorted(failed)[:10]), file=sys.stderr)
        return 1
    payload={
        "schema_version":2,"artifact_kind":"test_passed","commit_sha":commit,
        "passed_at":datetime.now(timezone.utc).isoformat(),"recipes":recipes,
        "portable_three_backend_recipes":portable,"certified_backends":certified,
        "checks":["safe_ast_expansion","pandas_reference_execution","underlying_operator_evidence","portable_subset_three_backend_parity","per_recipe_output_fingerprint"],
        "execution":execution,
        "hashes":current_hashes(),
    }
    VERIFIED_PATH.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(f"wrote {VERIFIED_PATH} ({len(recipes)} production recipes; {len(portable)} portable)");return 0
if __name__=="__main__":raise SystemExit(main())
