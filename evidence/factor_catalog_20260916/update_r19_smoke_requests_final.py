"""Apply reviewed runtime-derived DSL repairs to the same forty-factor cohort."""
import gzip,hashlib,json
from pathlib import Path
from factor_engine.tools.catalog_r19_operator_recipes import migrate_formula as operator
from factor_engine.tools.catalog_r19_remaining_params_recipes import migrate_formula as params
root=Path(__file__).resolve().parent
source=root/"r19-smoke-requests-r3.jsonl.gz"
assert hashlib.sha256(source.read_bytes()).hexdigest()=="3b5eafb5c9ecbaf41a64bffc310006b585795167a4640e224ea8827c3ff35bab"
with gzip.open(source,"rt") as f:rows=[json.loads(l) for l in f]
changed=0
for row in rows:
 before=row["formula"];notes=[]
 for migrate in (operator,params):
  row["formula"],extra=migrate(row["formula"],row.get("logic") or "")
  notes.extend(extra)
 if before!=row["formula"]:
  changed+=1;assert notes
  row["previous_smoke_formula"]=before;row["migration_changes"]=notes
assert len(rows)==40 and changed>0
output=root/"r19-smoke-requests-final.jsonl.gz"
with gzip.open(output,"xt") as f:
 for row in rows:f.write(json.dumps(row,ensure_ascii=False)+"\n")
print(json.dumps({"changed":changed,"rows":len(rows),"sha256":hashlib.sha256(output.read_bytes()).hexdigest()}))
