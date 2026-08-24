# -*- coding: utf-8 -*-
"""Make the INNER factor_optimizer package importable from the repo root.

The real package lives at factor_optimizer/factor_optimizer/ (with __init__.py).
From the repo root, `import factor_optimizer` would otherwise resolve to a
namespace package rooted at the OUTER dir, so submodules like
factor_optimizer.search do not resolve. Inserting this outer dir into
sys.path lets `import factor_optimizer` find the inner package's __init__.py.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
