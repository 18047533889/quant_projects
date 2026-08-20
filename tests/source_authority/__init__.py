# -*- coding: utf-8 -*-
"""conftest.py for source_authority tests."""
import sys
from pathlib import Path

# factor_engine/ must be first in sys.path for correct FE imports
FE_ROOT = Path(__file__).resolve().parent.parent.parent / "factor_engine"
ROOT = Path(__file__).resolve().parent.parent.parent

# Add factor_engine first, then root
fe_str = str(FE_ROOT)
root_str = str(ROOT)

# Clear any existing entries and re-add in correct order
sys.path = [p for p in sys.path if p not in (fe_str, root_str)]
sys.path.insert(0, fe_str)
sys.path.insert(1, root_str)

