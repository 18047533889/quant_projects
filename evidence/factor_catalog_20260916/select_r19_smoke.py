"""Choose diverse, changed, bound daily formulas from pinned compile evidence."""
import csv,gzip,json,hashlib,collections
from pathlib import Path
root=Path(__file__).resolve().parent
evidence=root/"r19-combined-r3.jsonl.gz"
assert hashlib.sha256(evidence.read_bytes()).hexdigest()=="7555588a66d60cd7cf6ee307f3a401d5ae081c870d8dc28699dd4979bd3bb1c8"
groups=collections.defaultdict(list)
with gzip.open(evidence,"rt") as f:
 for line in f:
  r=json.loads(line)
  if r["status"]!="COMPILED" or r["before_formula"]==r["current_formula"]:continue
  if not r.get("bindings") or any(b["dataset"]!="ashare_stock_daily_adj" for b in r["bindings"]):continue
  family=r["changes"][0].split(":")[0].split("(")[0][:90]
  groups[family].append(r)
chosen=[]
while groups and len(chosen)<40:
 for key in list(groups):
  chosen.append(groups[key].pop(0))
  if not groups[key]:del groups[key]
  if len(chosen)==40:break
selected={(str(r["source_row"]),r["id"]):r for r in chosen}
output=root/"r19-smoke-requests.jsonl.gz"
with gzip.open(root/"factor_catalog_review_r18c.csv.gz","rt",encoding="utf-8-sig",newline="") as f,gzip.open(output,"xt") as dst:
 for old in csv.DictReader(f):
  r=selected.get((old["source_row"],old["id"]))
  if r is None:continue
  assert old["current_formula"]==r["before_formula"]
  result={k:old.get(k,"") for k in ("source_row","id","domain","pit","logic","batch")}
  result.update(formula=r["current_formula"],fields="|".join(sorted({b["table"]+"."+b["column"] for b in r["bindings"]})),tables="StockDailyBarAdj")
  dst.write(json.dumps(result,ensure_ascii=False)+"\n")
print(json.dumps({"selected":len(selected),"output":str(output),"sha256":hashlib.sha256(output.read_bytes()).hexdigest()}))
