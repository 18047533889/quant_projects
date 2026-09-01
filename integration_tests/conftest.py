# -*- coding: utf-8 -*-
"""pytest bootstrap for the cross-package integration harness (Task #107).

Path setup mirrors the root ``tests/conftest.py`` so the integration suite can
be run either as ``pytest integration_tests/`` from the repo root or as its own
leg in a CI tier:

1. Repo root first (CI parity: every other tier runs with the freshly checked
   out repo root importable) — makes ``factor_engine`` / ``data_access`` /
   ``quant_evaluator`` / ``factor_assets`` / ``quant_platform`` resolve to the
   pinned SOURCE tree.
2. The INNER ``factor_preprocess`` / ``factor_optimizer`` package dirs, so the
   (potentially stale) wheels in the venv are shadowed and the tests always
   exercise the source contracts the per-package tasks (A/C/D/E/F/G) are
   landing.

Ordering rule (same as root tests/conftest.py): later ``insert(0)`` wins.  We
insert repo root FIRST and the inner dirs LAST so the inner dirs sit at the
front and their ``factor_preprocess`` / ``factor_optimizer`` top-level packages
(shadowing the namespace dirs) take precedence — and the earlier insert keeps
``factor_engine`` from being shadowed by the stale root copies the root
conftest explicitly guards against.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_QUANT_ROOT = Path(__file__).resolve().parents[1]

#: Inner (real) package dirs for the namespace-shadowed distributions.  From
#: the repo root, ``import factor_preprocess`` would otherwise bind to the
#: OUTER namespace dir whose submodules do not resolve.
_INNER_PKG_DIRS = (
    "factor_preprocess",
    "factor_optimizer",
)

for _path in (
    str(_QUANT_ROOT),
    *[str(_QUANT_ROOT / _dir) for _dir in _INNER_PKG_DIRS],
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

#: Guard: factor_engine must NOT be shadowed by a namespace dir — the real
#: package lives at repo root.  If the resolution was hijacked (e.g. a stray
#: PYTHONPATH entry), fail loudly instead of silently testing the wrong tree.
_FE_PKG = None
try:
    import factor_engine  # noqa: PLC0415  (module-level import after path setup)

    _FE_PKG = factor_engine.__file__
except ImportError:  # pragma: no cover - only on a broken environment
    _FE_PKG = None
if _FE_PKG is None or "site-packages" in _FE_PKG:
    raise RuntimeError(
        "integration_tests: factor_engine resolved outside the repo source "
        f"tree ({_FE_PKG!r}); refusing to run the cross-package harness "
        "against the wrong tree"
    )


def pytest_collection_modifyitems(config, items):
    """Tag everything in this package as ``integration``.

    ``pytest integration_tests/`` runs the harness directly; the ``integration``
    marker lets a CI tier select exactly this suite.  NOTE: running
    ``pytest -m integration`` from the repo root also collects the whole
    ``tests/`` tree (which has 24 PRE-EXISTING collection errors unrelated to
    this harness) — the CI step therefore runs ``pytest integration_tests/``
    with the marker applied, not a bare ``-m integration`` selection.
    """
    marker = config.stash.get("_integration_marker", None)
    if marker is None:
        marker = _build_integration_marker()
        config.stash["_integration_marker"] = marker
    for item in items:
        if item.path.is_relative_to(Path(__file__).parent):
            item.add_marker(marker)


def _build_integration_marker():
    """Build the ``integration`` marker once (lazy, avoids import cycles)."""
    import pytest

    return pytest.mark.integration


#: Optional real-data env vars are NOT required for this harness — every test
#: is hermetic (pure contracts, no parquet / no PG / no network).  Set
#: ``CI_INTEGRATION_ASHARE_DATA`` only if a future A股 real-data shadow leg
#: should also run under this package.
CI_INTEGRATION_ASHARE_DATA = os.environ.get("CI_INTEGRATION_ASHARE_DATA", "0") == "1"
