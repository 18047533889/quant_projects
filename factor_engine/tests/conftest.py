"""pytest 全局：确保 factor_engine 与 quant_projects 根目录在 import 路径中。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_FE_ROOT = Path(__file__).resolve().parents[1]
_QUANT_ROOT = _FE_ROOT.parent

# ``data_access`` is installed as an editable package in the project venv.
# Do not add its source directory directly: that shadows the factor_engine
# ``tests`` package when the monorepo is collected from its root.
for _path in (str(_FE_ROOT), str(_QUANT_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Import-path mismatch guard: the ROOT repo ALSO defines a ``tests/`` package
# (``tests/__init__.py``), so a combined pytest run that collects BOTH
# ``tests/`` and ``factor_engine/tests/`` hits
# ``ImportPathMismatchError('tests.conftest', ...)``.  That clash is inherent
# to the monorepo layout (two distinct ``tests`` packages); it does NOT affect
# runs of either tree alone.  The residency gate is deliberately import-free
# (no ``import tests`` / no root FE import at module scope), so it can be
# collected under EITHER tree.  When pytest starts from the repo ROOT it loads
# the ROOT ``tests/conftest.py``; when it starts inside ``factor_engine/`` it
# loads this submodule conftest.  A run that names BOTH trees in one command
# triggers the mismatch and must be split into two commands (see
# factor_engine/docs/R21_SOURCE_RESIDENCY_AUDIT.md §5).

# The editable-install finder (``__editable__.factor_engine-0.3.1``) maps the
# top-level ``mining`` package to ``<repo>/factor_engine/mining``.  When the
# monorepo root is on sys.path *before* the editable finder is consulted, an
# untracked ``<repo>/mining`` directory wins the mapping and the campaign
# module resolves to the wrong copy.  Re-import the editable ``mining`` first
# so ``tests/r42/test_r21_campaign_snapshot_pin.py`` always exercises the
# factor_engine copy being edited, then restore the root path.
_QUANT_ROOT_STR = str(_QUANT_ROOT)
if _QUANT_ROOT_STR in sys.path:
    _root_pos = sys.path.index(_QUANT_ROOT_STR)
    sys.path.remove(_QUANT_ROOT_STR)
    try:
        import factor_engine.mining  # noqa: F401
    finally:
        sys.path.insert(_root_pos, _QUANT_ROOT_STR)


def pytest_configure(config):
    """R63-P0: make operator-module imports during collection order-independent.

    A test module that calls ``load_all()`` at module scope freezes the registry
    (R40 read-only contract).  A LATER test module that imports an operator
    module (which calls ``register_operator`` at import time) then fails with
    ``operator registry is not writable: frozen``.  That is an import-order
    dependency: the same tree collects cleanly or errors depending on module
    order.

    Fix: install a meta-path finder that thaws the registry right before ANY
    ``factor_engine.cleaned_operators`` submodule is imported.  Every operator
    module import then starts with a writable registry, so collection order never
    matters.  The finder is a no-op for non-operator imports and never touches
    production runtime paths (it only acts when the registry is already
    frozen/finalized, which only happens after a test-session ``load_all()``).
    """
    import importlib.abc
    import sys as _sys

    from factor_engine.cleaned_operators.registry import (
        OperatorRegistry,
        _BOOTSTRAP_TOKEN,
    )

    class _ThawBeforeOperatorImport(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.startswith("factor_engine.cleaned_operators"):
                if OperatorRegistry.lifecycle() in ("frozen", "finalized"):
                    OperatorRegistry.thaw_for_bootstrap(_BOOTSTRAP_TOKEN)
            return None

    _sys.meta_path.insert(0, _ThawBeforeOperatorImport())


def pytest_sessionstart(session):
    """pytest hook：收集开始前重新开放 registry（若已 freeze）。

    各算子测试模块在 import 期注册算子，因此会话级 hook 必须在首个模块
    导入前把 registry 从 frozen/finalized 拉回 BUILDING。若尚未 load（lifecycle
    仍为 building），保持原状。
    """
    # 延迟导入，避免在 conftest 导入时触发副作用
    from factor_engine.cleaned_operators import _test_session_reopen_if_needed

    _test_session_reopen_if_needed()


def pytest_collection_modifyitems(session, config, items):
    """pytest hook：收集结束后、执行前解冻 registry。

    收集期约 135 个测试模块在模块作用域直接调用 ``ensure_cleaned_loaded()``
    或 ``load_all()``，其中任何一个都会把 registry freeze（R40 只读契约）。
    收集完成后统一解冻一次，保证执行期需要注册的算子模块（如不在 load_all
    清单内的 fiscal_logit_score_op / time_semantic_ops / same_clock_lag /
    panel_batch1）可以正常注册。production 语义只约束 load_all 收尾；测试
    会话的注册开放不触碰生产运行时路径。
    """
    from factor_engine.cleaned_operators import _test_session_reopen_if_needed

    _test_session_reopen_if_needed()


@pytest.fixture(autouse=True)
def _reset_production_env_leaks():
    """清理 FactorEngine._sync_production_env 在进程内遗留的生产环境变量。"""
    yield
    os.environ.pop("QUANT_PRODUCTION_MODE", None)
    if os.environ.get("FACTOR_ENGINE_RUN_MODE") == "production":
        os.environ.pop("FACTOR_ENGINE_RUN_MODE", None)
