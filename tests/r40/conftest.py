# -*- coding: utf-8 -*-
"""R40 测试共享 fixtures。

``writable_global_registry``：让依赖全局 ``OperatorRegistry`` 可写的 R40 测试
（operator_spec / bootstrap）在**已被其它测试冻结**的全局 registry 上也能跑。
沿用 ``tests/operators/test_r10_registry_identity.py::mutable_registry`` 已验证的
模式：临时解冻 → yield → 清掉测试新增的 scratch canonical → 重新冻结。

（governance 测试用隔离的 ``_TestRegistry`` 子类，不受此影响；其 seal 兼容性在
测试文件内处理。）
"""
from __future__ import annotations

import pytest

from factor_engine.cleaned_operators.registry import OperatorRegistry, _BOOTSTRAP_TOKEN


@pytest.fixture
def writable_global_registry():
    """临时解冻全局 OperatorRegistry，让测试能注册 scratch 算子。

    - 开始时若已冻结（前序测试 bootstrap 过）→ ``thaw_for_bootstrap`` 恢复可写；
    - yield 后清理本次测试新增的 canonical（snapshot 对比），并恢复 FROZEN
      lifecycle——即使测试体抛异常也恢复。
    """
    was_frozen = OperatorRegistry.lifecycle() == "frozen"
    if was_frozen:
        OperatorRegistry.thaw_for_bootstrap(_BOOTSTRAP_TOKEN)
    # 分别快照：operator canonical 名 vs alias 名（alias 名通常不在 operator 集合里，
    # 用同一集合判断会把既有 alias 误删）。
    before_ops = set(OperatorRegistry._operators)
    before_aliases = set(OperatorRegistry._aliases)
    try:
        yield OperatorRegistry
    finally:
        for name in list(OperatorRegistry._operators):
            if name in before_ops:
                continue
            try:
                OperatorRegistry.unregister(name)
            except Exception:
                pass
        for name in list(OperatorRegistry._catalog):
            if name in before_ops:
                continue
            try:
                OperatorRegistry._catalog.pop(name, None)
            except Exception:
                pass
        for alias in list(OperatorRegistry._aliases):
            if alias in before_aliases:
                continue
            try:
                OperatorRegistry._aliases.pop(alias, None)
            except Exception:
                pass
        if was_frozen:
            if OperatorRegistry.lifecycle() == "building":
                OperatorRegistry.finalize()
            OperatorRegistry.freeze()
