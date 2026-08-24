"""7×24 连续挖掘入口。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alphaprobe.continuous import main

if __name__ == "__main__":
    main()
