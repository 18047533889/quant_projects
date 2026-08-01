"""Repository path bootstrap for direct script execution."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "factor_engine", REPO_ROOT / "AutoFactorEvaluation-RECONSTRUCT"):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)
