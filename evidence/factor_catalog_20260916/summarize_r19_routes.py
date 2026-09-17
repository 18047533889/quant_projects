"""Summarize recorded routes; never infer missing GPU or conversion measurements."""
import collections,gzip,json
from pathlib import Path
root=Path(__file__).resolve().parent
with gzip.open(root/"r19-real-smoke-final.jsonl.gz","rt") as f:rows=[json.loads(l) for l in f]
routes=collections.Counter();transfers={};factors=[]
for row in rows:
 route=row.get("backend_path") or {};plan=route.get("physical_plan") or {}
 routes[route.get("actual_backend") or "unreported"]+=1
 for edge in plan.get("transfers",[]):
  key=edge["edge_id"]
  if key in transfers:assert transfers[key]==edge
  transfers[key]=edge
 factors.append({"id":row["id"],"actual_backend":route.get("actual_backend"),"finite_count":row["finite_count"],"value_count":row["value_count"],"region_id":plan.get("region_id"),"estimated_peak_memory_bytes":plan.get("estimated_peak_memory_bytes"),"memory_basis":plan.get("memory_basis"),"resident_reuse_count":plan.get("resident_reuse_count"),"parameter_certification_degradation_reason":route.get("parameter_certification_degradation_reason")})
result={"scope":"One forty-factor research batch. Estimates are not measurements; no GPU speed claim.","actual_backend_factor_counts":dict(routes),"distinct_recorded_transfer_edges":len(transfers),"recorded_transfer_actual_bytes":sum(e.get("actual_bytes",0) for e in transfers.values()),"recorded_transfer_actual_ms":sum(e.get("actual_ms",0) for e in transfers.values()),"transfer_measurement_basis_counts":dict(collections.Counter(e.get("actual_bytes_basis","unreported") for e in transfers.values())),"factor_ledger":factors}
with (root/"r19-performance-conversion-ledger.json").open("x") as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps({k:v for k,v in result.items() if k!="factor_ledger"},ensure_ascii=False))
