"""Publish a reconciled, explicitly historical execution checkpoint."""
import json
import argparse
from pathlib import Path
from export_review import export_review

base = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--audit", default="full-parse-current.jsonl.gz")
parser.add_argument("--output", default="factor_catalog_review_checkpoint_r2.csv.gz")
parser.add_argument("--smoke", action="append", default=[])
args = parser.parse_args()
prior = json.loads((base / "review-checkpoint-export.json").read_text())
paths = [base / name for name in prior["smoke_evidence"]]
for pattern in (
    "resume-daily939-small100-r2-*.jsonl.gz",
    "resume-daily1127-next500-*.jsonl.gz",
    "isolate-daily2186-single-*.jsonl.gz",
    "isolate-daily2186-batch-fixed-*.jsonl.gz",
    "resume-daily2216-next100-*.jsonl.gz",
    "optional-cs-real-row*.jsonl.gz",
    "probe-row1295-fixed.jsonl.gz",
):
    paths.extend(sorted(base.glob(pattern)))
paths.extend(base / name for name in args.smoke)
paths = list(dict.fromkeys(paths))
counts = export_review(
    base / "input.jsonl.gz", base / args.audit,
    base / args.output, paths,
    [base / "compile-full-latest.jsonl.gz"],
)
print(json.dumps({
    "counts": counts, "smoke_evidence": [p.name for p in paths],
    "scope": "checkpoint_not_final_all_factor_execution",
    "execution_scope": "historical_formula_matched_not_current_code_certification",
}, ensure_ascii=False))
