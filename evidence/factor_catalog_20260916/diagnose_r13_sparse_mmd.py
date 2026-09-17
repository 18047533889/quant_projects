"""One bounded DAG of exact subexpressions; diagnostics, not factor rewrites."""
import ast
import gzip
import json
import os
from pathlib import Path

os.environ["ASHARE_PARQUET_ROOT"] = "/home/sunhaiwei/cos_data"
os.environ["DATA_ACCESS_SKIP_COS_MIRROR"] = "1"
os.environ["DATA_ACCESS_RUN_MODE"] = "interactive_research"

from factor_engine.api.dsl_parser import DSLParser
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.cleaned_operators import load_all
from factor_engine.runtime.engine import FactorEngine
from factor_engine.storage.sources.data_access_source import DataAccessSource

base = Path(__file__).resolve().parent
with gzip.open(base / "r13-auto-batch103.jsonl.gz", "rt") as stream:
    records = [json.loads(line) for line in stream]
record = next(r for r in records if r["source_row"] == 60447 or str(r["source_row"]) == "60447")
formula = record["executed_formula"]
nodes = {ast.unparse(n) for n in ast.walk(ast.parse(formula, mode="eval"))
         if isinstance(n, ast.Call)}
load_all()
parser = DSLParser(surface="compat_research")
source = DataAccessSource(
    dataset="ashare_stock_daily_adj",
    fields={"amount": "AdjAmount", "low": "AdjLow", "ret": "Return"},
    start_date="2025-10-01", end_date="2026-04-30",
    instrument_filter=["000001.SZ", "600519.SH"],
    run_mode="interactive_research", production=False, read_auto=False,
)
engine = FactorEngine(build_backend("auto"), source, run_mode="research")
expressions = sorted(nodes, key=lambda x: (len(x), x))
factors = [Factor(name=f"probe_{i}", expr=parser.parse(s), source_expr=s,
                  surface="compat_research") for i, s in enumerate(expressions)]
summaries = {}
def sink(name, value):
    import numpy as np
    arr = np.asarray(value, dtype=float)
    summaries[name] = {"formula": expressions[int(name.split("_")[1])],
                       "size": int(arr.size), "finite": int(np.isfinite(arr).sum()),
                       "unique_finite": int(np.unique(arr[np.isfinite(arr)]).size)}
    return True
engine.run_many(factors, result_policy="sink", sink=sink)
assert len(summaries) == len(factors)
output = base / "r13-sparse-mmd-diagnostic.json"
with output.open("x") as stream:
    json.dump({"source_row": 60447, "window": ["2025-10-01", "2026-04-30"],
               "symbols": ["000001.SZ", "600519.SH"],
               "scope": "subexpression counts, no panels or formula changes",
               "subexpressions": summaries}, stream, indent=2)
print(json.dumps(summaries))
