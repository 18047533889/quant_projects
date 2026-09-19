# -*- coding: utf-8 -*-
"""Append section 9 to BACKEND_COVERAGE_REPORT_20260919.md via a unified diff.

The report is tracked; per this repo's working rule the tree is only ever modified
through `git apply`, so this script emits a patch rather than writing the file.
"""
from __future__ import annotations

import difflib
import os
import sys

REPO = "/home/sunhaiwei/quant_projects"
REPORT = "/home/sunhaiwei/quant_projects/evidence/BACKEND_COVERAGE_REPORT_20260919.md"
SECTION = os.environ.get("SECTION", "/tmp/section9.md")

old = open(REPORT, encoding="utf-8").read()
add = open(SECTION, encoding="utf-8").read()
if not old.endswith("\n"):
    old += "\n"
new = old + add

diff = "".join(difflib.unified_diff(
    old.splitlines(keepends=True), new.splitlines(keepends=True),
    fromfile="a/evidence/BACKEND_COVERAGE_REPORT_20260919.md",
    tofile="b/evidence/BACKEND_COVERAGE_REPORT_20260919.md", n=3))

out = "/tmp/report9.patch"
open(out, "w", encoding="utf-8").write(diff)
print(f"report: {len(old.splitlines())} -> {len(new.splitlines())} lines")
print(f"EMITTED {out} ({len(diff.splitlines())} lines)")
sys.exit(0)
