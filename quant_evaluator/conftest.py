"""
Helper module for tests: make the pinned source tree importable.

Running ``pytest`` from inside ``quant_evaluator/`` imports the *installed*
wheel copy of ``quant_evaluator`` (whose submodules may be stale).  Tests that
import the source tree (adapters / contracts under ``quant_evaluator/``) must
shadow the wheel.  This plugin inserts the repo root ahead of the wheel.

The pinned source tree is the repodir ``quant_evaluator/`` package (not
``build/lib``) — see ``test_production_serialization.py``.
"""

_REPO_ROOT = "/home/sunhaiwei/quant_projects"
import sys

if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)