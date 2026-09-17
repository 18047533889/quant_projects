"""Select only newly revised, freshly compiled and bound daily formulas."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
base = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint-prefix", default="factor_catalog_review_r13a")
parser.add_argument("--expected-sha", default="cc95e87ed8e874ed72828f45b3ec30172701a9967cb0ddcf0371382cc1384539")
parser.add_argument("--output-name", default="r13a-daily-reviewed.input.jsonl.gz")
args = parser.parse_args()
source = base / (args.checkpoint_prefix + ".csv.gz")
expected = args.expected_sha
assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
changed = set()
with gzip.open(base / (args.checkpoint_prefix + ".decisions.jsonl.gz"), "rt") as f:
    for line in f:
        r = json.loads(line)
        if r.get("status") == "REVIEWED_FORMULA_CHANGED_NOT_EXECUTED":
            changed.add((str(r["source_row"]), r["id"]))
output = base / args.output_name
selected = 0
with gzip.open(source, "rt", encoding="utf-8-sig", newline="") as f, gzip.open(output, "xt") as out:
    for r in csv.DictReader(f):
        if (r["source_row"], r["id"]) not in changed:
            continue
        if r["compile_status"] != "COMPILED" or r["static_status"] != "PARSED_FIELDS_BOUND":
            continue
        bindings = json.loads(r["bindings"])
        if not bindings or any(b["dataset"] != "ashare_stock_daily_adj" for b in bindings):
            continue
        projected = {k: r.get(k, "") for k in ("source_row", "id", "domain", "pit", "logic", "batch")}
        projected.update(formula=r["current_formula"], fields=r["original_fields"], tables=r["original_tables"])
        out.write(json.dumps(projected, ensure_ascii=False) + "\n")
        selected += 1
print(json.dumps({"selected": selected, "source_sha256": expected,
                  "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}))
