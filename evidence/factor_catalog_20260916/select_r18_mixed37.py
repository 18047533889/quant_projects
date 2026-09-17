"""Pinned mixed-source regression subset excluding three invalid window arguments."""
import gzip, hashlib, json
from pathlib import Path
base=Path(__file__).resolve().parent
src=base/"factor_catalog_review_r17b.limit_mixed_source_auto40.input.jsonl.gz"
assert hashlib.sha256(src.read_bytes()).hexdigest()=="d8645d65a622441a5972c72ffdb9e90d9db82b7fb35fb9d5e4395cea07712377"
with gzip.open(src,"rt",encoding="utf-8") as f: rows=[json.loads(x) for x in f]
excluded={"R65X_03049","R65X_03053","R65X_03057"}
assert len(rows)==40 and {r["id"] for r in rows if "hawkes" in r["formula"]}==excluded
kept=[r for r in rows if r["id"] not in excluded]
assert len(kept)==37
out=base/"factor_catalog_review_r17c.limit_valid37.input.jsonl.gz"
with gzip.open(out,"xt",encoding="utf-8") as f:
    for r in kept:f.write(json.dumps(r,ensure_ascii=False)+"\n")
print(json.dumps({"rows":len(kept),"excluded_invalid_ids":sorted(excluded),"output":str(out),"sha256":hashlib.sha256(out.read_bytes()).hexdigest()}))
