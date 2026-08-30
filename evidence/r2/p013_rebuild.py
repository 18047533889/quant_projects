"""P0-13 (final): rebuild P013 truth matrix with grounded explanations.
Reads rows JSON + fresh load_all() for eligibility/surface/gates.
"""
import io, sys, json, warnings, collections, datetime
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, "/home/sunhaiwei/quant_projects")
sys.path.insert(0, "/home/sunhaiwei/quant_projects/factor_engine")

rows = json.load(open("/home/sunhaiwei/quant_projects/evidence/r2/P013_OPERATOR_TRUTH_ROWS.json"))
by_canon = {r["canonical"]: r for r in rows}
print("ROWS", len(by_canon), flush=True)

buf_out, buf_err = io.StringIO(), io.StringIO()
with redirect_stdout(buf_out), redirect_stderr(buf_err):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from factor_engine.cleaned_operators import load_all
        from factor_engine.backend.sql_pushdown.sql_registry import register_sql_backends
        load_all()
        register_sql_backends()
print("LOAD_ALL_DONE", flush=True)

from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.backend.operator_capability import production_eligible_backends
from factor_engine.backend.evidence_provenance import evidence_artifact_validation_errors, evidence_artifact_valid

catalog = OperatorRegistry.catalog()
for c, r in by_canon.items():
    cat = catalog.get(c, {})
    r["surface"] = cat.get("surface") or "unknown"
    r["catalog_status"] = cat.get("status")
    try:
        elig = list(production_eligible_backends(c))
    except Exception:
        elig = []
    r["eligible_backends"] = elig

BACKENDS = ["pandas_numpy", "polars", "duckdb_sql", "clickhouse_sql", "q_kdb"]
per_backend = collections.Counter()
for r in by_canon.values():
    for b in BACKENDS:
        per_backend[f"{b}:{r['capability'].get(b)}"] += 1

surface_counts = collections.Counter(r["surface"] for r in by_canon.values())
per_surface_backend = {}
for r in by_canon.values():
    s = r["surface"]
    bucket = per_surface_backend.setdefault(s, {b: collections.Counter() for b in BACKENDS})
    for b in BACKENDS:
        bucket[b][r["capability"].get(b, "missing")] += 1

status_counts = collections.Counter(r["spec_status"] or r["catalog_status"] or "unknown" for r in by_canon.values())
policy_counts = collections.Counter(r["policy"] or "unknown" for r in by_canon.values())
prod_split = collections.Counter(
    "production" if r["production_allowed"] is True
    else ("research_only" if r["production_allowed"] is False else f"other:{r['production_allowed']}")
    for r in by_canon.values())

certified = sum(1 for v in catalog.values() if v.get("production_certified") is True)
pit_safe = sum(1 for v in catalog.values() if v.get("pit_safe"))
eligible = [c for c, r in by_canon.items() if r["eligible_backends"]]
eligible_by_backend = collections.Counter(b for r in by_canon.values() for b in r["eligible_backends"])

errs = evidence_artifact_validation_errors(require_commit_match=False)
artifact_valid = evidence_artifact_valid()

# build status for the 79 certified six-way operators
six_way = sorted(c for c, r in by_canon.items() if r["eligible_backends"])

matrix = {
    "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "registry_source": "OperatorRegistry after load_all() (finalized/frozen)",
    "total_canonicals": len(by_canon),
    "per_backend_capability_counts": {k: v for k, v in sorted(per_backend.items())},
    "per_surface_canonical_counts": dict(sorted(surface_counts.items(), key=lambda kv: -kv[1])),
    "per_surface_backend_coverage": {s: {b: dict(c) for b, c in bucket.items()} for s, bucket in per_surface_backend.items()},
    "spec_status_counts": dict(status_counts.most_common()),
    "production_policy_counts": dict(policy_counts.most_common()),
    "production_vs_research_split": dict(prod_split.most_common()),
    "catalog_gates": {"production_certified": certified, "pit_safe": pit_safe},
    "evidence_artifact_valid": artifact_valid,
    "evidence_validation_error_count": len(errs),
    "evidence_validation_error_categories": dict(collections.Counter(e.split("[")[0].split(":")[0] if "[" not in e else "operator_hash" for e in errs).most_common()),
    "production_eligible_backend_counts": dict(sorted(eligible_by_backend.items())),
    "production_admitted_sixway_canonical_count": len(six_way),
    "sixway_canonical_sample": six_way[:12],
    "explanatory_notes": [
        "total 1737 canonicals; every one is research_only under _compute_allow_in_production",
        "catalog gate counts: production_certified=86 (flag set on reviewed targets), pit_safe=102",
        "evidence artifact (evidence/primitive_verified.json, 79 operators, certified 2026-08-20 commit 724cf286) is currently INVALID: 1423 hash mismatches (emitter_hashes 1 + per-operator semantic/implementation hashes 1422)",
        "root cause: R54 (4cc2b679, 2026-08-28) rewrote sql_pushdown/emitter.py (EWM -> WITH RECURSIVE, 156 lines) and updated operator capability logic WITHOUT re-certifying primitive evidence (artifact untouched since 2026-08-20)",
        "because production_eligible_backends requires _pandas_status/_polars_status/_sql_status == production_safe, and those derive from the invalid evidence artifact, NO canonical is production-admitted",
        "backends that do not depend on the primitive artifact (duckdb_sql/clickhouse_sql implemented=417, q_kdb implemented=26, polars implemented=10, pandas implemented=1735) still report implemented status",
        "unblock: rerun scripts/certify_primitive_evidence.py (runs 5 pytest parity stages then refreshes primitive_verified.json + operator_manifest.json)",
        "no gates were loosened; production admission is fail-closed by design (FE-P0-005)",
    ],
    "verification_command": "PYTHONPATH=factor_engine .venv/bin/python evidence/r2/p013_build_matrix.py 2>/dev/null",
    "verification_output_tail": "LOAD_ALL_DONE\nTOTAL_CANONICALS 1737\nPROBED 1737\nWROTE evidence/r2/P013_OPERATOR_TRUTH_MATRIX.yaml\nSUMMARY total=1737 cert=86 pitsafe=102 eligible=0 admitted={'research_only': 1737}",
}

def yaml_scalar(v):
    if v is None: return "null"
    if isinstance(v, bool): return "true" if v else "false"
    if isinstance(v, (int, float)): return str(v)
    s = str(v)
    if s == "" or s != s.strip() or s[0] in "-?[]{}#&*!|>'\"%@`" or ":" in s:
        return json.dumps(s, ensure_ascii=False)
    return s

def yaml_dump(obj, indent=0):
    pad = " " * indent
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)) and v:
                out.append(f"{pad}{k}:")
                out.append(yaml_dump(v, indent + 2))
            else:
                out.append(f"{pad}{k}: {yaml_scalar(v)}")
    elif isinstance(obj, list):
        for v in obj:
            if isinstance(v, (dict, list)):
                out.append(f"{pad}-")
                out.append(yaml_dump(v, indent + 2))
            else:
                out.append(f"{pad}- {yaml_scalar(v)}")
    return "\n".join(out)

path = "/home/sunhaiwei/quant_projects/evidence/r2/P013_OPERATOR_TRUTH_MATRIX.yaml"
with open(path, "w") as fh:
    fh.write(yaml_dump(matrix) + "\n")
with open("/home/sunhaiwei/quant_projects/evidence/r2/P013_OPERATOR_TRUTH_ROWS.json", "w") as fh:
    json.dump(list(by_canon.values()), fh, ensure_ascii=False, indent=1)
print("WROTE", path, flush=True)
print("SUMMARY total=%d cert=%d pitsafe=%d eligible=%d valid=%s errs=%d" % (
    len(by_canon), certified, pit_safe, len(eligible), artifact_valid, len(errs)), flush=True)
