# -*- coding: utf-8 -*-
"""R40 #65/#214/#216/#217 测试：emitter 模板缓存 + emitter identity + 编译三态。"""
from __future__ import annotations

import os

import pytest


def _col(name: str = "close"):
    from planner.logical_plan import PlanNode

    return PlanNode(op="column", attrs={"name": name})


def _lit(value: float):
    from planner.logical_plan import PlanNode

    return PlanNode(op="literal", attrs={"value": value})


def _plan(op: str, *inputs):
    from planner.logical_plan import PlanNode

    return PlanNode(op=op, inputs=tuple(inputs))


# ---------------------------------------------------------------------------
# #65/#217：模板编译 + literal binding 分离的有界缓存
# ---------------------------------------------------------------------------


def test_emitter_caches_compiled_template_not_per_literal():
    from backend.sql_pushdown.emitter import (
        _SQL_TEMPLATE_CACHE,
        compile_plan_to_sql,
        reset_sql_template_cache,
    )

    reset_sql_template_cache()
    for w in (5.0, 10.0, 15.0, 20.0):
        compiled = compile_plan_to_sql(
            _plan("ts_mean", _col(), _lit(w)),
            dataset="d", time_column="ts", instrument_column="inst",
        )
        assert compiled is not None
    info = _SQL_TEMPLATE_CACHE.info()
    # 同一模板形状（不同 literal 值）→ 只保留 1 个条目（不是 per-literal 爆炸）。
    assert info["entries"] == 1


def test_emitter_template_cache():
    from backend.sql_pushdown.emitter import (
        compile_plan_to_sql_template,
        reset_sql_template_cache,
    )

    reset_sql_template_cache()
    p1 = _plan("add", _col(), _lit(5.0))
    tpl = compile_plan_to_sql_template(
        p1, dataset="d", time_column="ts", instrument_column="inst",
    )
    assert tpl is not None
    # 相同 binding → 零重编译（直接复用）。
    bound = tpl.bind(p1, dataset="d", time_column="ts", instrument_column="inst")
    assert bound.query == tpl.compiled.query
    # 不同 literal binding → 诚实重编译（SQL 反映新值，绝不硬凑复用）。
    p2 = _plan("add", _col(), _lit(10.0))
    bound2 = tpl.bind(p2, dataset="d", time_column="ts", instrument_column="inst")
    assert bound2 is not None
    assert "10" in bound2.query
    reset_sql_template_cache()


# ---------------------------------------------------------------------------
# #214：emitter identity 从 build manifest 取；unknown → production hard fail
# ---------------------------------------------------------------------------


def test_emitter_unknown_source_hash_fails_production(tmp_path):
    from backend.sql_pushdown.emitter import (
        SqlCompileError,
        assert_emitter_identity_known,
        emitter_identity,
        generate_emitter_identity_manifest,
    )

    manifest = tmp_path / "emitter_identity.json"
    # 无 manifest → unknown → production hard fail
    with pytest.raises(SqlCompileError):
        assert_emitter_identity_known(production=True, manifest_path=str(manifest))
    # research 不失败（返回空）
    assert assert_emitter_identity_known(production=False, manifest_path=str(manifest)) == ""
    # 生成 manifest → known
    generate_emitter_identity_manifest(out_path=str(manifest))
    ident = emitter_identity(manifest_path=str(manifest))
    assert ident.known is True
    assert ident.source_hash
    got = assert_emitter_identity_known(production=True, manifest_path=str(manifest))
    assert got == ident.source_hash


def test_emitter_unknown_source_hash_fails_production_compile(monkeypatch, tmp_path):
    """production 下 compile_plan_to_sql 也经 emitter identity 门禁（unknown → hard fail）。"""
    from backend.sql_pushdown.emitter import (
        SqlCompileError,
        compile_plan_to_sql,
        generate_emitter_identity_manifest,
    )

    monkeypatch.setenv("FACTOR_ENGINE_RUN_MODE", "production")
    monkeypatch.setenv(
        "FACTOR_ENGINE_EMITTER_IDENTITY_MANIFEST",
        str(tmp_path / "missing_identity.json"),
    )
    with pytest.raises(SqlCompileError):
        compile_plan_to_sql(
            _plan("add", _col(), _lit(1.0)),
            dataset="d", time_column="ts", instrument_column="inst",
        )
    monkeypatch.delenv("FACTOR_ENGINE_RUN_MODE")


# ---------------------------------------------------------------------------
# #216：编译三态——SUPPORTED / SEMANTICALLY_UNSUPPORTED / COMPILER_ERROR
# ---------------------------------------------------------------------------


def test_compile_error_distinguished_from_unsupported():
    from backend.sql_pushdown.emitter import (
        CompileStatus,
        compile_plan_to_sql,
        last_compile_status,
    )

    # 语义不支持（未知算子）→ SEMANTICALLY_UNSUPPORTED
    out = compile_plan_to_sql(
        _plan("not_a_real_op", _col()),
        dataset="d", time_column="ts", instrument_column="inst",
    )
    assert out is None
    assert last_compile_status() == CompileStatus.SEMANTICALLY_UNSUPPORTED

    # 内部编译错误（非有限 literal 触发 _sql_literal raise）→ COMPILER_ERROR
    bad = _plan("add", _col(), _lit(float("inf")))
    out2 = compile_plan_to_sql(
        bad, dataset="d", time_column="ts", instrument_column="inst",
    )
    assert out2 is None
    assert last_compile_status() == CompileStatus.COMPILER_ERROR

    # 成功编译 → SUPPORTED
    out3 = compile_plan_to_sql(
        _plan("add", _col(), _lit(1.0)),
        dataset="d", time_column="ts", instrument_column="inst",
    )
    assert out3 is not None
    assert last_compile_status() == CompileStatus.SUPPORTED


def test_compile_error_hard_fails_production(monkeypatch, tmp_path):
    """production 下 COMPILER_ERROR → hard fail（不静默回退）。"""
    from backend.sql_pushdown.emitter import (
        SqlCompileError,
        compile_plan_to_sql,
        generate_emitter_identity_manifest,
    )

    manifest = tmp_path / "emitter_identity.json"
    generate_emitter_identity_manifest(out_path=str(manifest))
    monkeypatch.setenv("FACTOR_ENGINE_RUN_MODE", "production")
    monkeypatch.setenv("FACTOR_ENGINE_EMITTER_IDENTITY_MANIFEST", str(manifest))
    bad = _plan("add", _col(), _lit(float("inf")))
    with pytest.raises(SqlCompileError):
        compile_plan_to_sql(
            bad, dataset="d", time_column="ts", instrument_column="inst",
        )
    monkeypatch.delenv("FACTOR_ENGINE_RUN_MODE")


def test_emitter_semantically_unsupported_not_hard_fail_in_production(monkeypatch, tmp_path):
    """production 下 SEMANTICALLY_UNSUPPORTED 不 hard fail（允许回退 pandas）。"""
    from backend.sql_pushdown.emitter import (
        CompileStatus,
        SqlCompileError,
        compile_plan_to_sql,
        generate_emitter_identity_manifest,
        last_compile_status,
    )

    manifest = tmp_path / "emitter_identity.json"
    generate_emitter_identity_manifest(out_path=str(manifest))
    monkeypatch.setenv("FACTOR_ENGINE_RUN_MODE", "production")
    monkeypatch.setenv("FACTOR_ENGINE_EMITTER_IDENTITY_MANIFEST", str(manifest))
    try:
        out = compile_plan_to_sql(
            _plan("not_a_real_op", _col()),
            dataset="d", time_column="ts", instrument_column="inst",
        )
        assert out is None
        assert last_compile_status() == CompileStatus.SEMANTICALLY_UNSUPPORTED
    finally:
        monkeypatch.delenv("FACTOR_ENGINE_RUN_MODE")
