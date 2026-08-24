# -*- coding: utf-8 -*-
"""R45 CI hard gate: forbid the historical top-level FactorEngine packages
from reappearing at repo root, and forbid legacy top-level imports.

This is the regression guard for the FactorEngine namespace consolidation.
If any of the 17 packages is recreated at repo root, or any production .py
imports them via the OLD top-level name, this test fails.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_ROOT_PACKAGES = {
    "api", "backend", "cache", "cleaned_operators", "expr", "factor_recipes",
    "fields", "ir", "market", "mining", "planner", "runtime", "security",
    "semantic", "service", "storage", "util", "validation",
}

# A real legacy import statement referencing a forbidden package (old authority).
# Matches only genuine import syntax:
#   from <pkg>[.<sub>] import ...    or    import <pkg>[.<sub>] [as x]
LEGACY_IMPORT_RE = re.compile(
    r"^(from\s+(?:api|backend|cache|cleaned_operators|expr|factor_recipes|"
    r"fields|ir|market|mining|planner|runtime|security|semantic|service|storage|"
    r"util|validation)(?:\.\w[\w.]*)?\s+import"
    r"|import\s+(?:api|backend|cache|cleaned_operators|expr|factor_recipes|"
    r"fields|ir|market|mining|planner|runtime|security|semantic|service|storage|"
    r"util|validation)(?:\.\w[\w.]*)?(?:\s+as\s+\w+)?)"
)

_SKIP_DIRS = {"__pycache__", ".git", "build", "dist", ".venv", "node_modules",
              "cleaned_operators_archived", "vectorbt_qs"}


def _iter_py_files():
    for root, dirs, files in os.walk(_REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for fn in files:
            if fn.endswith(".py"):
                yield Path(root) / fn


def test_no_forbidden_root_packages_reappear():
    present = [p for p in FORBIDDEN_ROOT_PACKAGES if (_REPO_ROOT / p).exists()]
    assert not present, (
        f"R45 namespace consolidation regressed: forbidden root packages reappeared: "
        f"{sorted(present)}. Source authority is factor_engine/* ONLY."
    )


def test_no_legacy_top_level_imports():
    offenders = []
    for path in _iter_py_files():
        rel = path.relative_to(_REPO_ROOT)
        if str(rel).startswith("factor_engine/cleaned_operators_archived/"):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    code = line.split("#", 1)[0]
                    if LEGACY_IMPORT_RE.match(code.strip()):
                        offenders.append(f"{rel}:{i}: {line.strip()[:90]}")
                        break
        except (OSError, UnicodeDecodeError):
            continue
    assert not offenders, (
        f"R45 namespace consolidation regressed: {len(offenders)} legacy top-level "
        f"imports found. All must use factor_engine.<pkg>.:\n" + "\n".join(offenders[:10])
    )
