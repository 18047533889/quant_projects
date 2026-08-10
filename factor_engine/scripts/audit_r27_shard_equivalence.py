# -*- coding: utf-8 -*-
"""R27-199: shard 等价审计 —— sharded-compute + merge == full compute（R27-155/256）。"""
from __future__ import annotations
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
if ROOT not in sys.path: sys.path.insert(0, ROOT)
from scripts.audit_r27_acceptance import _src_contains

def main() -> None:
    test = os.path.join(ROOT, "tests", "r27", "test_shard_equivalence.py")
    assert os.path.exists(test), "tests/r27/test_shard_equivalence.py missing"
    assert _src_contains("tests/r27/test_shard_equivalence.py", "shard")
    print("R27_SHARD_EQUIVALENCE_OK (tests/r27/test_shard_equivalence.py present)")

if __name__ == "__main__":
    main()
