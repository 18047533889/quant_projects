"""Select only pinned R18 changed, compiled, adjusted-daily factors for one batch."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def main():
    p=argparse.ArgumentParser()
    for name in ("checkpoint","compile-evidence","output"):
        p.add_argument("--"+name,required=True,type=Path)
    for name in ("checkpoint-sha","compile-sha"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--limit",type=int,default=40)
    args=p.parse_args()
    if not 1<=args.limit<=40: raise ValueError("limit must be 1..40")
    pins=((args.checkpoint,args.checkpoint_sha),(args.compile_evidence,args.compile_sha))
    for path,digest in pins:
        if sha(path)!=digest: raise ValueError("input hash mismatch")
    eligible={}
    with gzip.open(args.compile_evidence,"rt",encoding="utf-8") as f:
        for line in f:
            r=json.loads(line)
            key=(str(r["source_row"]),r["id"])
            if key in eligible: raise ValueError("duplicate changed identity")
            eligible[key]=r
    chosen=[]
    with gzip.open(args.checkpoint,"rt",encoding="utf-8-sig",newline="") as f:
        for r in csv.DictReader(f):
            key=(r["source_row"],r["id"])
            ev=eligible.get(key)
            if ev is None: continue
            if ev["executed_formula"]!=r["current_formula"]: raise ValueError("formula mismatch")
            if ev["formula_sha256"]!=hashlib.sha256(r["current_formula"].encode()).hexdigest():
                raise ValueError("formula digest mismatch")
            if ev["compile_status"]!=r["compile_status"]: raise ValueError("compile mismatch")
            if r["compile_status"]!="COMPILED" or r["static_status"]!="PARSED_FIELDS_BOUND": continue
            bindings=json.loads(r["bindings"])
            if not bindings or any(b["dataset"]!="ashare_stock_daily_adj" for b in bindings): continue
            chosen.append({
                **{k:r.get(k,"") for k in ("source_row","id","domain","pit","logic","batch")},
                "formula":r["current_formula"],"fields":r["current_fields"],"tables":r["current_tables"],
            })
            if len(chosen)==args.limit: break
    if not chosen: raise ValueError("no eligible changed adjusted-daily factors")
    for path,digest in pins:
        if sha(path)!=digest: raise ValueError("input changed during selection")
    with gzip.open(args.output,"xt",encoding="utf-8") as f:
        for row in chosen: f.write(json.dumps(row,ensure_ascii=False)+"\n")
    print(json.dumps({"selected":len(chosen),"output":str(args.output),"sha256":sha(args.output)}))

if __name__=="__main__": main()
