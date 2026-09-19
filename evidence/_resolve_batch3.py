# -*- coding: utf-8 -*-
"""Batch 3 prep: resolve each candidate operator to its registered polars kernel class
and gather STATIC evidence about whether that kernel is a genuine polars expression.

Nothing here declares anything; it is a pre-filter so that only kernels whose body is
free of pandas round-trip tokens are proposed for a _physical_spec declaration.
"""
from __future__ import annotations

import ast
import inspect
import json
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")
REPO = "/home/sunhaiwei/quant_projects"

from factor_engine.cleaned_operators import load_all  # noqa: E402
from factor_engine.cleaned_operators.registry import OperatorRegistry  # noqa: E402
from factor_engine.backend import polars_backend_kind as pbk  # noqa: E402

CANDS = sys.argv[1:]
if not CANDS:
    raise SystemExit("usage: _resolve_batch3.py op1 op2 ...")

load_all()

BAD_TOKENS = ["to_pandas", "_pl_to_pd", "pd.DataFrame", "pd.Series", "pandas",
              ".to_pandas(", "to_numpy("]
DELEGATE_TOKENS = ["register_polars_udf", "_polars_udf", "delegate", "apply_pandas"]


def tokens(src: str):
    return {
        "pl": len(re.findall(r"\bpl\.", src)),
        "np": len(re.findall(r"\bnp\.", src)),
        "pd": len(re.findall(r"\bpd\.", src)),
        "to_pandas": src.count("to_pandas"),
        "to_numpy": src.count("to_numpy"),
        "map_elements": src.count("map_elements"),
        "map_batches": src.count("map_batches"),
        "over": src.count(".over("),
        "with_columns": src.count("with_columns"),
        "expr_col": src.count("pl.col("),
    }


def body_src(obj):
    cls = obj if isinstance(obj, type) else type(obj)
    try:
        return inspect.getsource(cls), cls.__name__, inspect.getsourcefile(cls)
    except (OSError, TypeError):
        return "", getattr(cls, "__name__", "?"), "?"


rows = []
for canon in CANDS:
    rec = {"operator": canon}
    try:
        pop = OperatorRegistry.get(canon, "pandas_numpy")
        rec["pandas_impl"] = type(pop).__name__
    except Exception as e:
        rec["error_pandas"] = f"{type(e).__name__}: {e}"
    try:
        plop = OperatorRegistry.get(canon, "polars", mode="any")
    except Exception as e:
        rec["error_polars"] = f"{type(e).__name__}: {e}"
        rows.append(rec)
        continue
    if plop is None:
        rec["polars_impl"] = None
        rows.append(rec)
        continue
    src, clsname, srcfile = body_src(plop)
    rec["polars_impl"] = clsname
    rec["source_file"] = srcfile
    rec["source_rel"] = srcfile.split("quant_projects/")[-1]
    rec["module"] = type(plop).__module__
    rec["tok"] = tokens(src) if src else {}
    rec["body_lineno"] = getattr(inspect.getsourcelines(plop if isinstance(plop, type) else type(plop))[1]
                                 if src else None, "real", None)
    try:
        k = pbk.canonical_polars_kind(canon, production_mode=True)
        rec["kind_production"] = getattr(k, "value", str(k))
    except Exception as e:
        rec["kind_production"] = f"ERR {type(e).__name__}: {e}"
    try:
        k = pbk.canonical_polars_kind(canon, production_mode=False)
        rec["kind_research"] = getattr(k, "value", str(k))
    except Exception as e:
        rec["kind_research"] = f"ERR {type(e).__name__}: {e}"
    try:
        rec["has_spec"] = pbk.get_physical_spec(canon) is not None
    except Exception as e:
        rec["has_spec"] = f"ERR {e}"
    rec["has_to_pandas_token"] = any(t in (src or "") for t in BAD_TOKENS)
    rec["has_delegate_token"] = any(t in (src or "") for t in DELEGATE_TOKENS)
    # where in the class hierarchy is the spec attribute looked up?
    rec["spec_attr_where"] = [k for k in dir(plop) if "physical_spec" in k.lower()]
    rows.append(rec)
    print(json.dumps(rec, default=str), flush=True)

out = f"{REPO}/evidence/_resolve_batch3.json"
json.dump(rows, open(out, "w"), indent=1, default=str)
print("\nWROTE", out)
