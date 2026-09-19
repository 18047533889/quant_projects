"""List how the highest-frequency catalog operators are implemented on polars.

The R57 manifest weights every operator by how many catalog rows call it. The
top names are the only ones worth writing a native polars implementation for, so
pair that ranking with each operator's current polars slot classification and its
backend set.
"""
from __future__ import annotations

import collections
import gzip
import json
import sys
from pathlib import Path


def main() -> int:
    manifest_dir = Path(sys.argv[1] if len(sys.argv) > 1 else
        "evidence/factor_catalog_20260916/r57_20260919_full")
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    dep: collections.Counter[str] = collections.Counter()
    for path in sorted(manifest_dir.glob("full-s*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                seen = {op.get("canonical") for op in (rec.get("ops_used") or [])
                        if op.get("canonical")}
                for canon in seen:
                    dep[canon] += 1

    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    load_all()

    print("distinct canonicals used:", len(dep))
    print()
    print("%-32s %9s %-26s %s" % ("canonical", "rows", "polars slot", "backends"))
    print("-" * 118)
    for canon, n in dep.most_common(top_n):
        op = None
        for backend in ("polars", "pandas_numpy", "pandas"):
            try:
                op = OperatorRegistry.get(canon, backend, mode="any")
            except Exception:  # noqa: BLE001
                op = None
            if op is not None:
                break
        spec = getattr(op, "_physical_spec", None)
        if spec is None:
            kind = "NO_SPEC"
        else:
            kind = str(getattr(spec, "kind", None) or getattr(spec, "class_name", None)
                       or type(spec).__name__)
        try:
            backends = sorted(map(str, OperatorRegistry.backends_for(canon) or []))
        except Exception:  # noqa: BLE001
            backends = []
        print("%-32s %9d %-26s %s" % (canon, n, kind, ",".join(backends)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
