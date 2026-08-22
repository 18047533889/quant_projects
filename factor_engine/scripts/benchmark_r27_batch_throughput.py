# -*- coding: utf-8 -*-
"""R27-199/124: 大批量因子吞吐 benchmark（BASELINE-1/2/3 vs R27 adaptive）。"""
from __future__ import annotations
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
if ROOT not in sys.path: sys.path.insert(0, ROOT)
from scripts.audit_r27_acceptance import benchmark_light, _head

def main() -> None:
    out = benchmark_light()
    print("HEAD:", _head())
    for k, v in out.items():
        print(f"  {k}: {v}")
    if "error" not in out:
        print("R27_THROUGHPUT_BENCH_OK")

if __name__ == "__main__":
    main()
