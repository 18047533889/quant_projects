# -*- coding: utf-8 -*-
"""Make the INNER factor_preprocess package importable from the repo root.

The real package lives at factor_preprocess/factor_preprocess/ (with __init__.py).
From the repo root, `import factor_preprocess` would otherwise resolve to a
namespace package rooted at the OUTER dir, so submodules like
factor_preprocess.transforms do not resolve. Inserting this outer dir into
sys.path lets `import factor_preprocess` find the inner package's __init__.py.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
