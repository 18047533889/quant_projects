# -*- coding: utf-8 -*-
"""R27-199: native multi-root fusion 审计（R27-233）。"""
from __future__ import annotations
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
if ROOT not in sys.path: sys.path.insert(0, ROOT)
from scripts.audit_r27_acceptance import probe_native_fusion

def main() -> None:
    probe = probe_native_fusion()
    print(probe)
    assert probe["can_fuse_same_scope"] is True
    assert len(probe["fusion_groups"]) >= 1
    print("R27_NATIVE_FUSION_OK")

if __name__ == "__main__":
    main()
