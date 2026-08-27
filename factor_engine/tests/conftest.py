"""pytest 全局：确保 factor_engine 与 quant_projects 根目录在 import 路径中。

P0-14: 测试生命周期不再以任何方式解冻生产 singleton。

- 移除 R63-P0 的 meta-path thaw finder（``pytest_configure`` 不再安装）。
- 移除 ``_test_session_reopen_if_needed`` 的会话级调用（pytest_sessionstart /
  pytest_collection_modifyitems 不再解冻）。
- 16 个「测试模块作用域导入 + 注册」的算子模块提前在 *legal building window*
  （首个 load_all() 之前）预热进注册表 —— 之后 load_all() freeze 时它们已在
  surface 中，测试期再次模块作用域导入走幂等 no-op，永远不需要 thaw。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_FE_ROOT = Path(__file__).resolve().parents[1]
_QUANT_ROOT = _FE_ROOT.parent

for _path in (str(_FE_ROOT), str(_QUANT_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

_QUANT_ROOT_STR = str(_QUANT_ROOT)
if _QUANT_ROOT_STR in sys.path:
    _root_pos = sys.path.index(_QUANT_ROOT_STR)
    sys.path.remove(_QUANT_ROOT_STR)
    try:
        import factor_engine.mining  # noqa: F401
    finally:
        sys.path.insert(_root_pos, _QUANT_ROOT_STR)

# ---------------------------------------------------------------------------
# P0-14: test-registry staging (registry lifecycle isolation).
#
# The registry is PRODUCTION's — the suite legitimately bootstraps it once
# (module-scope load_all() in test files) and the result is frozen.  The 16
# modules below register operators at import time and are referenced at module
# scope by tests; they must be registered DURING the building window (before any
# load_all freezes) so later module-scope imports are idempotent no-ops.
#
# NO meta-path thaw finder, NO thaw_for_bootstrap on the production class, NO
# _test_session_reopen_if_needed.  Only a NON-mutating lifecycle watcher that
# records whether the production registry was ever observed leaving `building`
# after any finalize/freeze (acceptance #3 assertion data).
# ---------------------------------------------------------------------------

_LATE_SURFACE_MODULES: tuple[str, ...] = (
    "factor_engine.cleaned_operators.panel_batch1",
    "factor_engine.cleaned_operators.time_semantic_gap",
    "factor_engine.cleaned_operators.fundamental.fiscal_logit_score_op",
    "factor_engine.cleaned_operators.common.fiscal_operators",
    "factor_engine.cleaned_operators.common.polars_ts_basic",
    "factor_engine.cleaned_operators.cross_section.panel_batch1",
    "factor_engine.cleaned_operators.intraday.state_space",
    "factor_engine.cleaned_operators.intraday.intra_state_space",
    "factor_engine.cleaned_operators.intraday.smart_money",
    "factor_engine.cleaned_operators.intraday.topology_manifold",
    "factor_engine.cleaned_operators.intraday.true_gap_batch3",
    "factor_engine.cleaned_operators.polars_native.ts_advanced_batch1",
    "factor_engine.cleaned_operators.technical.adaptive_filters",
    "factor_engine.cleaned_operators.fundamental.fiscal_batch2",
    "factor_engine.cleaned_operators.fundamental.fiscal_batch3",
    "factor_engine.cleaned_operators.time_semantic",
)

_STAGED_MODULES: list[str] = []


def pytest_sessionstart(session):
    """Stage late-surface modules into the building registry BEFORE any freeze.

    If load_all() has already frozen (a test module imported earlier during this
    session — the session hook runs before collection, so normally not), skip
    staging and leave the registry frozen.  We never thaw it.
    """
    _install_lifecycle_watcher()
    _stage_late_modules_if_building()


def pytest_collection_modifyitems(session, config, items):
    """Keep the collection-complete hook a no-op (staging already happened at
    sessionstart; the old session-thaw hook is gone)."""
    return None


def _install_lifecycle_watcher() -> None:
    """Non-mutating watcher on the PRODUCTION registry lifecycle.

    Records a consistent observation that the production class ever left
    ``building`` (finalized/frozen), any explicit thaw call on it, and a
    stacked traceback.  The watcher NEVER changes lifecycle; it only observes.
    """
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    if getattr(OperatorRegistry, "_p014_watcher_installed", False):
        return

    _orig_lifecycle = OperatorRegistry.lifecycle
    _orig_thaw = OperatorRegistry.thaw_for_bootstrap
    _orig_freeze = OperatorRegistry.freeze

    OperatorRegistry._p014_production_finalized_once = False
    OperatorRegistry._p014_thaw_calls: list[str] = []

    def _watched_lifecycle(cls):
        res = _orig_lifecycle.__func__(cls)
        if res in ("frozen", "finalized") and not cls.__dict__.get("_p014_production_finalized_once"):
            cls._p014_production_finalized_once = True
        return res

    def _watched_thaw(cls, token):
        cls._p014_thaw_calls = (cls.__dict__.get("_p014_thaw_calls") or [])
        import traceback
        cls._p014_thaw_calls.append("".join(traceback.format_stack()) | "".join(traceback.format_stack(limit=6)))
        return _orig_thaw.__func__(cls, token)

    def _watched_freeze(cls):
        res = _orig_freeze.__func__(cls)
        cls._p014_production_finalized_once = True
        return res

    OperatorRegistry.lifecycle = classmethod(_watched_lifecycle)
    OperatorRegistry.thaw_for_bootstrap = classmethod(_watched_thaw)
    OperatorRegistry.freeze = classmethod(_watched_freeze)
    OperatorRegistry._p014_watcher_installed = True


def _stage_late_modules_if_building() -> None:
    global _STAGED_MODULES
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    # The first explicit load_all in test modules will finalize+freeze.  Stage
    # NOW — while the registry is still building — so their module-scope imports
    # later are no-ops.
    if OperatorRegistry.lifecycle() != "building":
        return
    import importlib
    for mod in _LATE_SURFACE_MODULES:
        try:
            importlib.import_module(mod)
            _STAGED_MODULES.append(mod)
        except Exception as exc:  # noqa: BLE001 - a staging failure must not break the session
            import warnings
            warnings.warn(
                f"P0-14: late-surface module {mod} failed to stage: "
                f"{type(exc).__name__}: {exc}"
            )


# ---------------------------------------------------------------------------
# Acceptance #3: the global (production) registry lifecycle is never observed
# thawed after finalize during a *normal* test session.
# ---------------------------------------------------------------------------

_ACCEPTANCE_EXCLUDED_TRACK = (
    "test_factorengine_hardening",
    "test_r10_registry_identity",
    "test_layer_governance",
    "test_mining_admission",
    "test_r40_registry_governance_208_212",
    "test_r40_bootstrap",
    "r40/conftest",
)


@pytest.fixture(autouse=True)
def _p014_registry_lifecycle_not_thawed(request):
    """Assert the production registry was never observed thawed after finalize.

    The production registry is only ever thawed by the few EXPLICIT governance
    tests that exercise the token contract on purpose (they are recorded in
    ``_ACCEPTANCE_EXCLUDED_TRACK``).  Every other test must see the registry
    never leave the frozen/finalized state.
    """
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    fil = os.path.basename(request.node.fspath)
    is_explicit_governance = any(ex in fil for ex in _ACCEPTANCE_EXCLUDED_TRACK)

    thawed_before = list(OperatorRegistry.__dict__.get("_p014_thaw_calls") or [])
    try:
        yield
    finally:
        thawed_after = list(OperatorRegistry.__dict__.get("_p014_thaw_calls") or [])
        if not is_explicit_governance:
            new = thawed_after[len(thawed_before):]
            assert not new, (
                f"P0-14: production registry was thawed during {request.node.nodeid} "
                f"({len(new)} thaw call(s)); the global singleton must never be "
                "thawed after finalize in a normal test session.  Stacks: "
                + "\n".join(new[:2])
            )


@pytest.fixture(autouse=True)
def _reset_production_env_leaks():
    """清理 FactorEngine._sync_production_env 在进程内遗留的生产环境变量。"""
    yield
    os.environ.pop("QUANT_PRODUCTION_MODE", None)
    if os.environ.get("FACTOR_ENGINE_RUN_MODE") == "production":
        os.environ.pop("FACTOR_ENGINE_RUN_MODE", None)