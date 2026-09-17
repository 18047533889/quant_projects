"""Attach one closed smoke batch to the verified R19 checkpoint."""
import argparse,collections,csv,gzip,json
from pathlib import Path
from types import SimpleNamespace
from reconcile_r17_execution import reconcile,sha256
root=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument("--execution",required=True,type=Path);p.add_argument("--summary",required=True,type=Path)
for name in ("checkpoint","request","output-prefix"):p.add_argument("--"+name,required=True,type=Path)
for name in ("checkpoint-sha256","request-sha256"):p.add_argument("--"+name,required=True)
args=p.parse_args()
checkpoint=args.checkpoint
request=args.request
assert sha256(checkpoint)==args.checkpoint_sha256
assert sha256(request)==args.request_sha256
compile_manifest=json.loads(Path(str(checkpoint)+".manifest.json").read_text())
assert compile_manifest["output_sha256"]==args.checkpoint_sha256
with gzip.open(args.execution,"rt") as f: results=[json.loads(l) for l in f]
assert len(results)==40
by_key={(str(r["source_row"]),r["id"]):r for r in results}
assert len(by_key)==40
counts=collections.Counter()
with gzip.open(checkpoint,"rt",encoding="utf-8-sig",newline="") as f:
 for row in csv.DictReader(f):
  ev=by_key.get((row["source_row"],row["id"]))
  if ev:
   assert row["current_formula"]==ev["executed_formula"]
   counts[ev["status"]]+=1
  else:counts[row["execution_status"]]+=1
fmt=lambda d:",".join(k+"="+str(v) for k,v in sorted(d.items()))
payload=reconcile(SimpleNamespace(
 checkpoint=checkpoint,checkpoint_sha256=sha256(checkpoint),
 input=request,input_sha256=sha256(request),execution=args.execution,
 execution_sha256=sha256(args.execution),summary=args.summary,summary_sha256=sha256(args.summary),
 expected_records=40,expected_batch_counts=fmt(collections.Counter(r["status"] for r in results)),
 expected_catalog_counts=fmt(counts),output_prefix=args.output_prefix))
payload["compile_counts_nonempty_ids"]=compile_manifest["compile_counts_nonempty_ids"]
spec=root/"r19_delivery_spec.json"
with spec.open("x") as f:json.dump(payload,f,ensure_ascii=False,indent=2)
print(json.dumps({"checkpoint":payload["output_checkpoint"],"sha":payload["output_checkpoint_sha256"],"counts":payload["output_execution_status_counts"]}))
