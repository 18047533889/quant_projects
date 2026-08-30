"""P0-13: build production truth matrix over the FINAL OperatorRegistry.

Single-process design (fork pool inherits the loaded registry).  Writes
evidence/r2/P013_OPERATOR_TRUTH_MATRIX.yaml and a JSON row snapshot next to it
so partial work survives interruption.  Intentionally does NOT run
scripts/export_operator_manifest.py (its per-entry probe is ~10x slower and
unneeded for coverage stats).
"""
import io, sys, json, warnings, collections, datetime, multiprocessing as mp
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, "/home/sunhaiwei/quant_projects")
sys.path.insert(0, "/home/sunhaiwei/quant_projects/factor_engine")

buf_out, buf_err = io.StringIO(), io.StringIO()
with redirect_stderr(buf_err), redirect_stdout(buf_out):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from factor_engine.cleaned_operators import load_all
        from factor_engine.backend.sql_pushdown.sql_registry import register_sql_backends
        load_all()
        register_sql_backends()
print("LOAD_ALL_DONE", flush=True)

from factor_engine.cleaned_operators.registry import OperatorRegistry

canonicals = OperatorRegistry.list_canonical()
print("TOTAL_CANONICALS", len(canonicals), flush=True)
catalog = OperatorRegistry.catalog()

BACKENDS = ["pandas_numpy", "polars", "duckdb_sql", "clickhouse_sql", "q_kdb"]
EVIDENCE_DIR = "/home/sunhaiwei/quant_projects/evidence/r2"

def probe(canon):
    from factor_engine.backend.operator_capability import backend_status, production_eligible_backends
    from factor_engine.cleaned_operators.operator_spec import build_operator_spec
    caps = {}
    for b in BACKENDS:
        try:
            caps[b] = backend_status(canon, b, production_mode=True)
        except Exception as exc:
            caps[b] = f"error:{type(exc).__name__}"
    try:
        elig = list(production_eligible_backends(canon))
    except Exception:
        elig = []
    try:
        spec = build_operator_spec(canon)
        status = spec.status if spec else None
        policy = spec.production_policy if spec else None
        allowed = bool(spec.allow_in_production) if spec else None
    except Exception as exc:
        status = None; policy = None
        allowed = f"error:{type(exc).__name__}"
    return {"canonical": canon, "capability": caps, "eligible_backends": elig,
            "spec_status": status, "policy": policy, "production_allowed": allowed}

if __name__ == "__main__":
    mp.set_start_method("fork")
    with mp.Pool(28) as pool:
        rows = list(pool.imap_unordered(probe, canonicals, chunksize=8))
    print("PROBED", len(rows), flush=True)

    by_canon = {r["canonical"]: r for r in rows}
    # attach catalog fields
    for c, r in by_canon.items():
        cat = catalog.get(c, {})
        r["surface"] = cat.get("surface") or "unknown"
        r["catalog_status"] = cat.get("status")
        r["registered_backends"] = sorted(set(OperatorRegistry.backends_for(c)))

    # per-backend capability totals
    per_backend = collections.Counter()
    for r in by_canon.values():
        for b in BACKENDS:
            per_backend[f"{b}:{r['capability'].get(b)}"] += 1

    # per-surface canonical counts and per-surface x backend coverage
    surface_counts = collections.Counter(r["surface"] for r in by_canon.values())
    per_surface_backend = {}
    for r in by_canon.values():
        s = r["surface"]
        bucket = per_surface_backend.setdefault(s, {b: collections.Counter() for b in BACKENDS})
        for b in BACKENDS:
            bucket[b][r["capability"].get(b, "missing")] += 1
    per_surface_backend = {s: {b: dict(c) for b, c in bucket.items()}
                           for s, bucket in per_surface_backend.items()}

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

    # verification output tail (this run)
    out_tail = "\n".join([
        "LOAD_ALL_DONE", f"TOTAL_CANONICALS {len(canonicals)}",
        f"PROBED {len(rows)}",
    ])
    for line in (buf_out.getvalue() + buf_err.getvalue()).splitlines()[-3:]:
        out_tail += "\n" + line

    matrix = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "registry_source": "OperatorRegistry after load_all() (finalized/frozen)",
        "total_canonicals": len(by_canon),
        "per_backend_capability_counts": {k: v for k, v in sorted(per_backend.items())},
        "per_surface_canonical_counts": dict(sorted(surface_counts.items(), key=lambda kv: -kv[1])),
        "per_surface_backend_coverage": per_surface_backend,
        "spec_status_counts": dict(status_counts.most_common()),
        "production_policy_counts": dict(policy_counts.most_common()),
        "production_vs_research_split": dict(prod_split.most_common()),
        "catalog_gates": {"production_certified": certified, "pit_safe": pit_safe},
        "evidence_eligible_backend_counts": dict(sorted(eligible_by_backend.items())),
        "production_admitted_canonical_count": len(eligible),
        "verification_command": "PYTHONPATH=factor_engine .venv/bin/python evidence/r2/p013_build_matrix.py 2>/dev/null",
        "verification_output_tail": out_tail,
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

    path = EVIDENCE_DIR + "/P013_OPERATOR_TRUTH_MATRIX.yaml"
    with open(path, "w") as fh:
        fh.write(yaml_dump(matrix) + "\n")
    with open(EVIDENCE_DIR + "/P013_OPERATOR_TRUTH_ROWS.json", "w") as fh:
        json.dump(list(by_canon.values()), fh, ensure_ascii=False, indent=1)
    print("WROTE", path, flush=True)
    print("SUMMARY total=%d cert=%d pitsafe=%d eligible=%d admitted=%s" % (
        len(by_canon), certified, pit_safe, len(eligible), dict(prod_split)), flush=True)
