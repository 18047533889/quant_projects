"""Measure how many R57 rows call single-input estimators that require two inputs.

The bounded execution smoke found three cross-backend hard failures caused by a
formula/contract arity mismatch rather than a backend difference. If those are
isolated demo templates the priority is low; if they are widespread the catalog
itself is wrong. Count them, and show whether the affected rows are the
"weakop_" demo family or real production factors.
"""
from __future__ import annotations

import ast
import collections
import csv
import gzip
import sys

SUSPECTS = (
    "ts_transfer_entropy",
    "ts_ridge_regression_in_sample_resid",
    "ts_first_passage_hit_probability",
)


def count_args(node: ast.Call) -> tuple[int, int]:
    return len(node.args), len(node.keywords)


def main() -> int:
    path = sys.argv[1]
    calls: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    weakop = collections.Counter()
    total = 0
    parse_fail = 0
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            total += 1
            f = row.get("r57_formula") or ""
            if not any(s + "(" in f for s in SUSPECTS):
                continue
            name = row.get("name") or ""
            is_weak = name.startswith("weakop_")
            try:
                tree = ast.parse(f, mode="eval")
            except SyntaxError:
                parse_fail += 1
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                        and node.func.id in SUSPECTS:
                    na, nk = count_args(node)
                    calls[node.func.id][(na, nk)] += 1
                    if is_weak:
                        weakop[node.func.id] += 1

    print("total rows:", total, "| unparsable rows touching suspects:", parse_fail)
    for name in SUSPECTS:
        print()
        print("===", name, "===")
        if not calls[name]:
            print("   no calls found")
            continue
        for (na, nk), n in sorted(calls[name].items()):
            print("   args=%d keywords=%d -> %d calls" % (na, nk, n))
        print("   weakop_-named rows among them:", weakop[name])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
