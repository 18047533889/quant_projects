#!/usr/bin/env python3
"""运行 GTJA-191 独立包全部测试。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


def main() -> int:
    loader = unittest.TestLoader()
    start_dir = str(PACKAGE_ROOT / "tests")
    suite = loader.discover(start_dir, pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
