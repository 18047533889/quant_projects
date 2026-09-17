"""Select bounded, newly lowered CCI formulas for one auto run."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

base = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint-prefix", default="factor_catalog_review_r16a")
parser.add_argument("--decisions-prefix", default="factor_catalog_review_r16a")
parser.add_argument("--expected-sha", default="269859a6edba076034c9149acadd7a6340192188746cef64e8f12a27115d35ca")
parser.add_argument("--output-name", default="factor_catalog_review_r16a.cci_daily_smoke.input.jsonl.gz")
parser.add_argument("--limit", type=int, default=40)
parser.add_argument("--exclude-execution", type=Path)
args = parser.parse_args()
if not 1 <= args.limit <= 40:
    raise ValueError("limit must be in [1, 40]")
prefix = args.checkpoint_prefix
source = base / f"{prefix}.csv.gz"
expected = args.expected_sha
if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
    raise RuntimeError("R16 checkpoint hash mismatch")

cci = set()
with gzip.open(base / f"{args.decisions_prefix}.decisions.jsonl.gz", "rt") as fh:
    for line in fh:
        decision = json.loads(line)
        if decision.get("status") != "REVIEWED_FORMULA_CHANGED_NOT_EXECUTED":
            continue
        if any(" CCI:" in change for change in decision.get("changes", ())):
            cci.add((str(decision["source_row"]), decision["id"]))

excluded = set()
if args.exclude_execution is not None:
    exclusion = args.exclude_execution if args.exclude_execution.is_absolute() else base / args.exclude_execution
    with gzip.open(exclusion, "rt", encoding="utf-8") as fh:
        for line in fh:
            item = json.loads(line)
            if item.get("status") != "EXECUTED" or not item.get("executed_formula"):
                raise ValueError("exclude execution must contain successful formula evidence")
            key = (str(item["source_row"]), item["id"], item["executed_formula"])
            if key in excluded:
                raise ValueError(f"duplicate excluded execution identity: {key[:2]}")
            excluded.add(key)

output = base / args.output_name
selected = []
with gzip.open(source, "rt", encoding="utf-8-sig", newline="") as fh:
    for row in csv.DictReader(fh):
        if (row["source_row"], row["id"]) not in cci:
            continue
        if row["compile_status"] != "COMPILED" or row["static_status"] != "PARSED_FIELDS_BOUND":
            continue
        if row["execution_status"] == "EXECUTED":
            continue
        if (row["source_row"], row["id"], row["current_formula"]) in excluded:
            continue
        bindings = json.loads(row["bindings"])
        if not bindings or any(b["dataset"] != "ashare_stock_daily_adj" for b in bindings):
            continue
        selected.append({
            **{k: row.get(k, "") for k in ("source_row", "id", "domain", "pit", "logic", "batch")},
            "formula": row["current_formula"],
            "fields": row["original_fields"],
            "tables": row["original_tables"],
        })
        if len(selected) == args.limit:
            break
if not selected:
    raise RuntimeError("no executable CCI daily rows selected")
with gzip.open(output, "xt", encoding="utf-8") as fh:
    for row in selected:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
print(json.dumps({
    "selected": len(selected), "source_sha256": expected,
    "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
}, sort_keys=True))
