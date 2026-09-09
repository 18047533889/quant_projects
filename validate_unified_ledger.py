import json,re,sys
from pathlib import Path

root=Path(__file__).resolve().parent
ledger_path=Path(sys.argv[1]) if len(sys.argv)>1 else root/"evidence/r2/v3_177_unified_ledger.json"
spec_path=Path(sys.argv[2]) if len(sys.argv)>2 else root/"docs/V3_REMEDIATION_SPEC_20260906.md"
spec=spec_path.read_text()
expected={x for x in re.findall(r"^#### ([A-Z]+-\d+) ·",spec,re.M) if not x.startswith("GOLD-")}
data=json.loads(ledger_path.read_text())
rows=data.get("items",[]); ids=[r.get("issue_id") for r in rows]
errors=[]
if len(expected)!=177: errors.append(f"spec inventory expected 177, got {len(expected)}")
if set(ids)!=expected: errors.append(f"missing={sorted(expected-set(ids))} extra={sorted(set(ids)-expected)}")
if len(ids)!=len(set(ids)): errors.append("duplicate issue IDs")
for r in rows:
    for key in ("requirement_code","public_paths","tests","history_evidence","next_local_step"):
        if key not in r: errors.append(f"{r.get('issue_id')}: missing field {key}")
    status = r.get("status", "")
    if status == "CLOSED" or status.endswith("VERIFIED"):
        if not r.get("public_paths") or not r.get("tests") or not r.get("history_evidence"):
            errors.append(f"{r.get('issue_id')}: falsely {status} without complete evidence")
for key in ["GOLD-40",*[f"E2E-{c}" for c in "ABCDEFGH"]]:
    present = key=="GOLD-40" and "gold40" in data or key.startswith("E2E-") and key in data.get("e2e_chains",{})
    if not present: errors.append(f"missing traceability section {key}")
if errors:
    print("\n".join(errors)); sys.exit(1)
print(f"OK: {len(ids)} unique issue IDs; closure evidence guard passed")
