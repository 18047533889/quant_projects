"""Apply the reviewed one-lag repair to exactly one frozen smoke request."""
import gzip,hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
source=root/"r19-smoke-requests.jsonl.gz"
assert hashlib.sha256(source.read_bytes()).hexdigest()=="c62faa46ea50a102172af4c3f2e312d313fed7760fc5363afc1811fa0e2bf08d"
from factor_engine.tools.catalog_r19_parameter_recipes import migrate_formula
old="rank(ts_autocorr_decay_half_life(ret, 60, 1, False, 20))"
original="rank(ts_autocorr_decay_half_life(ret, 60, 1, 20, 0.05))"
fixed,notes=migrate_formula(original)
assert fixed=="rank(ts_autocorr_decay_half_life(ret, 60, 10, False, 20))"
rows=[]
with gzip.open(source,"rt") as f:
 for line in f:rows.append(json.loads(line))
changed=0
for row in rows:
 if row["id"]=="cold20_53394749c8b5339b":
  assert row["formula"]==old
  row["formula"]=fixed;row["migration_changes"]=notes;changed+=1
assert changed==1 and len(rows)==40
output=root/"r19-smoke-requests-r3.jsonl.gz"
with gzip.open(output,"xt") as f:
 for row in rows:f.write(json.dumps(row,ensure_ascii=False)+"\n")
print(json.dumps({"output":str(output),"changed":changed,"sha256":hashlib.sha256(output.read_bytes()).hexdigest()}))
