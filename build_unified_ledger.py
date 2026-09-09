import json, re
import yaml
import hashlib
import subprocess
from scripts.ledger_provenance import observe, reconcile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPEC = ROOT / "docs/V3_REMEDIATION_SPEC_20260906.md"
OUT = ROOT / "evidence/r2/v3_177_unified_ledger.json"

text = SPEC.read_text(encoding="utf-8")
matches = list(re.finditer(r"^#### ([A-Z]+-\d+) · (.+)$", text, re.M))
ids = [m.group(1) for m in matches if not m.group(1).startswith("GOLD-")]
if len(ids) != 177 or len(set(ids)) != 177:
    raise SystemExit(f"authoritative inventory is not 177 unique IDs: {len(ids)}/{len(set(ids))}")

merged = {}
source_history = {}
spec_sha256 = hashlib.sha256(SPEC.read_bytes()).hexdigest()
current_tree = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD^{tree}"], text=True).strip()

def record_source(iid, row, rel, manual=False):
    source = ROOT / rel
    raw = source.read_bytes() if source.is_file() else json.dumps(row, sort_keys=True).encode()
    observe(source_history, iid, row, source=rel, source_bytes=raw,
            spec_hash=spec_sha256, manual=manual)
    # Kept only as a display projection; authoritative status is reconciled
    # from every original report below, never this last-writer projection.
    merged[iid] = {**merged.get(iid, {}), **row, "source_ledger": rel}
additional_findings = []
sources = [
    "loop/v3_remediation_ledger.json",
    "evidence/r2/v3_assets_177_ledger.json",
    "evidence/r2/v3_platform_ledger.json",
    "evidence/r2/v3_preprocess_ledger_20260907.json",
    "evidence/r2/v3_optimizer_ledger_merge_20260907.json",
    "evidence/r2/v3_qe_ledger_merge_20260907.json",
]
yaml_sources=["evidence/r2/v3_assets_20260906.yaml","evidence/r2/v3_optimizer_20260906.yaml","evidence/r2/v3_preprocess_20260906.yaml"]
for rel in sources:
    p = ROOT / rel
    if not p.exists(): continue
    d = json.loads(p.read_text())
    additional_findings.extend(d.get("additional_findings", []))
    rows = d.get("items") or d.get("entries") or []
    if isinstance(d.get("issues"), list): rows += d["issues"]
    if isinstance(d.get("issues"), dict):
        rows += [dict(v, issue_id=k) for k, v in d["issues"].items()]
    for row in rows:
        iid = row.get("issue_id")
        if iid in ids: record_source(iid, row, rel)
for rel in yaml_sources:
    p=ROOT/rel
    if not p.exists(): continue
    d=yaml.safe_load(p.read_text()) or {}
    for row in d.get("results",[]):
        iid=row.get("issue_id")
        if iid in ids: record_source(iid, row, rel)
# JSON ledgers are the later merge authority; replay them after package YAML.
for rel in sources[1:]:
    p=ROOT/rel
    if not p.exists(): continue
    d=json.loads(p.read_text()); rows=d.get("items") or d.get("entries") or []
    if isinstance(d.get("issues"),list): rows += d["issues"]
    if isinstance(d.get("issues"),dict): rows += [dict(v,issue_id=k) for k,v in d["issues"].items()]
    for row in rows:
        iid=row.get("issue_id")
        if iid in ids: record_source(iid, row, rel)
local_overrides={
 "FA-10":{"status":"FIXED_LOCAL","public_path":"factor_assets.clustering.incremental.incremental_assign","tests":["factor_assets/tests/test_incremental_cluster_assign.py"],"migration":"rebuild affected ANN indices; old indices remain immutable"},
 "DTA-04":{"status":"FIXED_LOCAL","public_path":"factor_assets.adapters.data_access.resolve_universe_snapshot","tests":["factor_assets/tests/test_adapter_data_access.py","factor_assets/tests/assembly/test_engine.py"],"migration":"new assemblies bind dated universe snapshot; old lineage remains research-only"},
 "PL-07":{"status":"FIXED_LOCAL","public_path":"quant_platform.app.orchestrator.Pipeline._build_candidate_artifact","tests":["quant_platform/tests/test_orchestrator_pipeline.py"],"migration":"canonical payload bytes are newly published; old memory refs are not production certified"},
 "MOD-06":{"status":"FIXED_LOCAL","public_path":"quant_platform.app.orchestrator.Pipeline.modeling_feature_manifests","tests":["quant_platform/tests/test_modeling_manifest_bridge.py","quant_platform/tests/test_orchestrator_pipeline.py"],"migration":"incomplete provenance stays research-only and model_ready=false"},
 "OPS-08":{"status":"PARTIAL","public_path":["quant_platform.app.ops08_migration.plan_ops08_migration","quant_platform.app.ops08_migration.execute_ops08_shadow","jobs.ops08_replay_fixture"],"tests":["quant_platform/tests/test_ops08_migration.py","quant_platform/tests/test_outbox_claim.py"],"migration":"Frozen dependency graph selective invalidation and actual shadow risk evaluation-to-health plus recipe-to-FP-value-to-QE-to-health replay. Persisted audit-only envelopes, exact direct dependencies and scoped outbox publishing preserve original bytes/pointers; 12 focused replay tests and 23 outbox tests. Cluster/feature/model replay authorities and approved production pointer switch/rollback remain incomplete."},
}
for iid,row in local_overrides.items():
    record_source(iid, row, "manual:local_overrides", manual=True)

items=[]
targets={"FP":"factor_preprocess/factor_preprocess","FE":"factor_engine","FA":"factor_assets","FO":"factor_optimizer/factor_optimizer","SIM":"factor_assets/clustering","DTA":"data_access","STA":"quant_evaluator/metrics","MOD":"modeling","OPS":"quant_platform/app","QAT":"tests","QA":"tests"}
for n,m in enumerate(matches):
    iid=m.group(1)
    if iid.startswith("GOLD-"): continue
    end=matches[n+1].start() if n+1<len(matches) else len(text)
    sec=text[m.end():end]
    steps=re.search(r"\*\*实施步骤\*\*\s*(.*?)(?:\n\*\*必须落地)",sec,re.S)
    first=""
    if steps:
        q=re.search(r"\n1\.\s*(.+)",steps.group(1)); first=q.group(1).strip() if q else ""
    old=merged.get(iid,{})
    tests=old.get("tests") or old.get("test_evidence") or old.get("targeted_tests") or []
    if isinstance(tests,str): tests=[tests]
    paths=old.get("public_path") or old.get("public_paths") or []
    paths = paths or old.get("actual_entrypoints") or []
    if isinstance(paths,str): paths=[paths]
    hist=old.get("history_evidence") or old.get("migration") or old.get("evidence") or []
    hist = hist or old.get("affected_artifacts") or old.get("migration_or_no_migration_reason") or []
    if isinstance(hist,str): hist=[hist]
    status=old.get("status","OPEN_UNASSESSED")
    blockers=old.get("remaining_blockers") or old.get("blockers") or old.get("blocker") or old.get("remaining") or old.get("limitations") or []
    if isinstance(blockers,str): blockers=[blockers]
    test_rows=old.get("regression_results") or []
    tests = tests or [x if isinstance(x,str) else x.get("test") for x in test_rows if isinstance(x,str) or isinstance(x,dict) and x.get("test")]
    # Never promote agent-local package counts to closure, including qualified
    # VERIFIED labels. Preserve the original report separately for audit.
    if (status == "CLOSED" or status.endswith("VERIFIED")) and not (paths and tests and hist):
        status="EVIDENCE_INCOMPLETE"
    scope_match=re.search(r"\*\*现有定位范围\*\*：(.*?)。",sec)
    scopes=(scope_match.group(1).replace("`","").strip() if scope_match else "declared owner package")
    proposed_test=f"tests/{iid.split('-')[0].lower()}/test_{iid.lower().replace('-', '_')}_public_path.py"
    target=targets.get(iid.split("-")[0],scopes)
    action=first or f"Implement and validate {m.group(2).strip()}"
    precise=(blockers[0] if blockers else f"{action}; patch public caller under {target} and add success/failure coverage at {proposed_test}.")
    items.append({
        "issue_id":iid,"requirement_code":iid,"title":m.group(2).strip(),
        "status":status,"reported_owner_status":old.get("status","OPEN_UNASSESSED"),"public_paths":paths,"tests":tests,
        "history_evidence":hist,"source_ledger":old.get("source_ledger"),
        "next_local_step":old.get("next_local_step") or precise,
        **reconcile(source_history.get(iid, []), current_tree=current_tree),
        "spec_sha256": spec_sha256,
        "task_namespace": spec_sha256,
    })

qat_path=ROOT/"evidence/r2/qat_remaining_matrix_20260907.json"
qat=json.loads(qat_path.read_text()) if qat_path.exists() else {}
payload={
 "schema_version":"v3-177-unified-ledger/2","authoritative_spec":str(SPEC.relative_to(ROOT)),
 "aggregation_execution_status":"NOT_RUN",
 "working_tree_clean":not bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip()),
 "truth_policy":"Aggregation does not execute or authenticate tests. Source records are lossless; contradictory domains/statuses require reconciliation. Manual overrides are assertions, never verification.",
 "items":items,
 "additional_findings":additional_findings,
 "frozen_r5_verification":{
   "qe":{"result":"851 passed, 2 optional absent-kernel skips, 4 warnings","evidence":"evidence/r2/v3_r5_qe_capability_final.log"},
   "optimizer_preprocess":{"result":"1150 passed, 1 expected offline HP xfail, 12 warnings","evidence":"evidence/r2/v3_r5_frozen_fofp.log"},
   "modeling_jobs":{"result":"453 passed, 59 warnings","evidence":"evidence/r2/v3_r5_frozen_modeljobs.log"},
   "qat07":{"result":"PASS;8 wheels;1750 installed Python files match checkout;actual FP parameter/output parity and actual test collection reconciliation PASS","evidence":"evidence/r2/v3_qat07_r5_source_parity_20260907.json"},
   "platform":{"result":"469 passed before one added livePG isolation test;final livePG suite20passed","evidence":["evidence/r2/v3_r5_platform_final.log","evidence/r2/v3_r5_postgres_final.log"]},
   "scope":"HISTORICAL_REPORT: these runs were not executed by this generator and do not certify the current working tree or production."
 },
 "gold40":qat.get("GOLD-40",{"status":"NOT_FULLY_VERIFIED","gap":"missing targeted GOLD-40 evidence matrix"}),
 "e2e_chains":qat.get("E2E",{}),
 "latest_integrated_verification":{"result":"447 passed, zero skipped, 59 warnings","scope":"tests/modeling + jobs/tests with mature fixed-update monitor and E2E-C; before later finite-only modeling fit and V-05 runner fixes","evidence":"evidence/r2/v3_r5_modeling_jobs_final.log"},
 "reported_test_context":{"qe":"844 passed, 2 optional-kernel skips, 4 warnings after actual CUDA exposure and CPU residual validity fixes; evidence/r2/v3_r5_qe_final.log; subsequent capability-matrix tests tracked separately", "factor_assets":"1375 passed, zero skipped, 49 warnings with isolated real Faiss/Annoy/igraph; no FA source changes in current continuation", "platform":"469 passed, zero skipped, 3 warnings including real PostgreSQL and actual OPS08 shadow replay; then 20 live PostgreSQL tests including targeted outbox recovery isolation", "optimizer_preprocess":"1118 passed, 1 xfailed, 12 warnings before subsequent V-05/V-07 audit changes; current-source rerun tracked in R5 report", "modeling":"447 combined modeling/jobs passed before finite-only fit amendment; 5 independent finite-fit regressions failed before fix, 13 focused passed after; final modeling rerun tracked in R5 report", "classification":"supporting evidence only; no package count promotes any of the 177 issues, GOLD40 or E2E chains to CLOSED"},
 "priority_remaining_non_qe":[i["issue_id"]+": "+i["next_local_step"] for i in items if not i["issue_id"].startswith("QE-") and i["status"] in {"OPEN_UNASSESSED","TO_VERIFY","INTEGRATION_PENDING","PARTIAL_FIXED_LOCAL","EVIDENCE_INCOMPLETE","BLOCKED","PARTIAL_TRANSFERRED","NEEDS_RECONCILIATION","NEEDS_REVALIDATION","MANUAL_ASSERTION","NOT_RUN"}][:40],
}
OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n")
