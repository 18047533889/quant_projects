"""Merge explicitly ordered review layers, pin source identity, retain supersession ledger."""
import argparse,collections,csv,gzip,hashlib,json
from pathlib import Path
SOURCE_SHA="190a34ada8b380103e65f59b9cdf503f022c930838a0856e28824ab0b1e6d330"
def digest(p):
    with open(p,"rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def read(p):
    with (gzip.open(p,"rt") if p.suffix==".gz" else p.open(encoding="utf-8")) as f:
        yield from map(json.loads,f)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=Path,required=True)
    ap.add_argument("--layers",type=Path,nargs="+",required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    if digest(args.source)!=SOURCE_SHA:raise ValueError("source identity mismatch")
    merged={};owners={};overrides=[];pins={}
    for path in args.layers:
        pins[str(path)]=digest(path);seen=set()
        for row in read(path):
            row=dict(row)
            row["status"]=row.get("status",row.get("compile_status","UNVERIFIED"))
            row["changes"]=list(row.get("changes") or [])
            if row.get("current_definition"):
                row["changes"].append("R20_CURRENT_DEFINITION: "+row["current_definition"])
            key=(int(row["source_row"]),row["id"])
            if key in seen:raise ValueError(f"duplicate in {path}: {key}")
            seen.add(key)
            if key in merged:
                old=merged[key]
                if old["before_formula"]!=row["before_formula"]:raise ValueError(f"conflicting base {key}")
                overrides.append({"source_row":key[0],"id":key[1],"old_layer":owners[key],"new_layer":str(path),
                                  "formula_changed":old["current_formula"]!=row["current_formula"]})
            if row["current_formula"]!=row["before_formula"] and not row.get("changes"):
                raise ValueError(f"undocumented change {key}")
            merged[key]=row;owners[key]=str(path)
    baseline_failed=set()
    for shard in range(8):
        path=args.source.parent/f"r20-baseline-s{shard}.jsonl.gz"
        for row in read(path):
            if row["status"]=="COMPILE_FAILED":baseline_failed.add((int(row["source_row"]),row["id"]))
    if len(baseline_failed)!=12260 or set(merged)!=baseline_failed:
        raise ValueError(f"baseline failure coverage mismatch missing={len(baseline_failed-set(merged))} extra={len(set(merged)-baseline_failed)}")
    matched=set();all_ids=set()
    with gzip.open(args.source,"rt",encoding="utf-8-sig",newline="") as f:
        for row in csv.DictReader(f):
            if not row["id"]:continue
            if row["id"] in all_ids:raise ValueError("duplicate source id")
            all_ids.add(row["id"]);key=(int(row["source_row"]),row["id"])
            if key in merged:
                if row["current_formula"]!=merged[key]["before_formula"]:raise ValueError(f"source mismatch {key}")
                matched.add(key)
    if matched!=set(merged):raise ValueError("unmatched proposals")
    if len(all_ids)!=113893:raise ValueError("source factor count")
    manifest=Path(str(args.output)+".manifest.json")
    if args.output.exists() or manifest.exists():raise FileExistsError(args.output)
    with gzip.open(args.output,"xt",encoding="utf-8") as f:
        for key in sorted(merged):f.write(json.dumps(merged[key],ensure_ascii=False)+"\n")
    result={"source_sha256":SOURCE_SHA,"proposal_count":len(merged),"output_sha256":digest(args.output),
            "layer_hashes":pins,"supersessions":overrides,"claimed_statuses":dict(collections.Counter(r["status"] for r in merged.values())),
            "scope":"Reviewed proposals only; final full current-code compile required."}
    with manifest.open("x") as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in {"supersessions","layer_hashes"}},ensure_ascii=False))
if __name__=="__main__":main()
