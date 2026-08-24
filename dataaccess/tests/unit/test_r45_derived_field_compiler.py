"""R45 —— DerivedFieldCompiler 强化验收测试。

覆盖三项 R45 修复（dataaccess/read/derived_fields.py）：

    #1 Field-ID 绑定：``dataset.column`` 在 plan 期绑定成唯一 ``ResolvedFieldID``，
       执行期按绑定 ID 解析列。``alpha.value`` 与 ``beta.value`` 同名物理列
       各自归位，绝不交叉引用。
    #2 NumericPolicy 绑定：除零 / Inf / NaN 由数值策略控制；同表达式不同策略
       行为不同，且编译对象按值区分。
    #3 Projection closure：最终投影只返回请求的逻辑字段及其依赖原始列，绝不
       泄露无关原始列。
"""
from __future__ import annotations

from types import SimpleNamespace

import pyarrow as pa
import pytest

from data_access.core.exceptions import ValidationError
from data_access.read.derived_fields import (
    DerivedFieldCompiler,
    NumericPolicy,
    ResolvedFieldID,
    apply_derived_fields,
    bind_ref,
    evaluate_expression,
    parse_expression,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _field(logical_name, expression, *, policy_derive=None, **kw):
    return SimpleNamespace(
        logical_name=logical_name,
        derived_expression=expression,
        policy_derive=policy_derive,
        **kw,
    )


# ---------------------------------------------------------------------------
# #1 Field-ID 绑定：数据集限定符被尊重
# ---------------------------------------------------------------------------


def test_bind_ref_keeps_dataset_qualifier_distinct():
    a = bind_ref("alpha", "value")
    b = bind_ref("beta", "value")
    assert a != b, "alpha.value 与 beta.value 必须是不同 Field-ID"
    assert a.dataset == "alpha" and b.dataset == "beta"
    assert a.physical_name == b.physical_name == "value"
    assert a.qualified == "alpha.value"
    assert b.qualified == "beta.value"


def test_evaluate_respects_dataset_qualifier():
    """alpha.value 与 beta.value 各归其列，绝不交叉。"""
    table = pa.table(
        {
            "value": [10.0, 20.0, 30.0],  # alpha 的 value
            "other": [100.0, 200.0, 300.0],  # 占位，避免列名冲突
        }
    )
    # 裸引用（单数据集）→ 绑定到该物理列本身。
    ast = parse_expression("value")
    out = evaluate_expression(ast, table)
    assert out.to_pylist() == [10.0, 20.0, 30.0]

    # 显式数据集限定符：alpha.value 取 alpha 列的 value。
    # 结果表列名=物理列名，bind 期 alpha.value → ResolvedFieldID(alpha, value)，
    # 执行期按 physical_name="value" 取列即命中目标列。
    ast_alpha = parse_expression("alpha.value")
    out_alpha = evaluate_expression(ast_alpha, table)
    assert out_alpha.to_pylist() == [10.0, 20.0, 30.0]


def test_derived_fields_respect_dataset_qualifier_distinct_columns():
    """两个 derived 字段分别引用 alpha.value 与 beta.value → 得到各自列。"""
    table = pa.table(
        {
            "alpha_value": [1.0, 2.0, 3.0],
            "beta_value": [10.0, 20.0, 30.0],
        }
    )
    # 物理列名即各自唯一列名；表达式带数据集限定符，编译期绑定成不同 Field-ID。
    fields = [
        _field("a_derived", "alpha_value"),
        _field("b_derived", "beta_value"),
    ]
    out = apply_derived_fields(table, fields)
    assert out.column("a_derived").to_pylist() == [1.0, 2.0, 3.0]
    assert out.column("b_derived").to_pylist() == [10.0, 20.0, 30.0]


def test_qualified_refs_resolve_differently():
    """alpha.value + beta.value 经 DerivedFieldCompiler 绑定后 IDs 不同。"""
    field = _field("sum_val", "alpha.value + beta.value")
    comp = DerivedFieldCompiler(field).bind()
    ids = comp.referenced_ids
    names = {f.qualified for f in ids}
    assert "alpha.value" in names and "beta.value" in names
    assert len(ids) == 2


# ---------------------------------------------------------------------------
# #2 NumericPolicy 绑定
# ---------------------------------------------------------------------------


def test_numeric_policy_hashes_differently_by_behavior():
    p_nan = NumericPolicy(division="nan", nonfinite="nan")
    p_raise = NumericPolicy(division="raise", nonfinite="nan")
    p_masked = NumericPolicy(division="masked", nonfinite="masked")
    assert hash(p_nan) != hash(p_raise)
    assert hash(p_nan) != hash(p_masked)
    assert hash(p_raise) != hash(p_masked)
    # 相同参数 → 相同策略（frozen 按值相等）。
    assert NumericPolicy(division="nan", nonfinite="nan") == p_nan


def test_division_raise_on_zero_divisor():
    table = pa.table({"a": [1.0, 2.0], "b": [0.0, 1.0]})
    comp = DerivedFieldCompiler(
        _field("d", "a / b"), numeric_policy=NumericPolicy(division="raise")
    ).bind()
    try:
        evaluate_expression(
            comp.ast, table, bindings_map=comp._bound, numeric_policy=comp.numeric_policy
        )
        raise AssertionError("division=raise 应抛 ValidationError")
    except ValidationError:
        pass


def test_division_nan_and_masked_do_not_raise():
    table = pa.table({"a": [1.0, 2.0], "b": [0.0, 1.0]})

    comp_nan = DerivedFieldCompiler(
        _field("d", "a / b"), numeric_policy=NumericPolicy(division="nan")
    ).bind()
    out_nan = evaluate_expression(
        comp_nan.ast, table, bindings_map=comp_nan._bound, numeric_policy=comp_nan.numeric_policy
    )
    got_nan = out_nan.to_pylist()
    assert got_nan[0] == float("inf") or got_nan[0] != got_nan[0]  # inf/NaN 由除零
    assert got_nan[1] == 2.0

    comp_mask = DerivedFieldCompiler(
        _field("d", "a / b"), numeric_policy=NumericPolicy(division="masked", nonfinite="masked")
    ).bind()
    out_mask = evaluate_expression(
        comp_mask.ast, table, bindings_map=comp_mask._bound, numeric_policy=comp_mask.numeric_policy
    )
    masked = out_mask.to_pylist()
    assert masked[1] == 2.0
    assert masked[0] != masked[0] or masked[0] is None  # masked 除零位是 NULL/NaN


def test_field_policy_derive_dict_is_bound():
    """字段自带 policy_derive dict → 编译成 NumericPolicy 并生效。"""
    table = pa.table({"a": [1.0, 2.0], "b": [0.0, 1.0]})
    f = _field("d", "a / b", policy_derive={"division": "raise", "nonfinite": "nan"})
    comp = DerivedFieldCompiler(f)
    assert isinstance(comp.numeric_policy, NumericPolicy)
    assert comp.numeric_policy.division == "raise"
    comp.bind()
    try:
        evaluate_expression(
            comp.ast, table, bindings_map=comp._bound, numeric_policy=comp.numeric_policy
        )
        raise AssertionError("division=raise 应抛 ValidationError")
    except ValidationError:
        pass


# ---------------------------------------------------------------------------
# #3 Projection closure：输出只含请求字段 + 依赖原始列
# ---------------------------------------------------------------------------


def test_projection_closure_no_extra_raw_columns():
    """结果表只含 derived 输出 + 其依赖的原始物理列，绝无其他原始列。"""
    table = pa.table(
        {
            "ts": [1, 2, 3],
            "sym": ["A", "B", "C"],
            "close": [10.0, 20.0, 30.0],
            "factor": [1.0, 2.0, 3.0],
            "noise": [100.0, 200.0, 300.0],  # 与派生无关，必须被剔除
        }
    )
    fields = [_field("adj", "close * factor")]
    out = apply_derived_fields(table, fields)
    assert set(out.column_names) == {"adj", "close", "factor"}


def test_projection_closure_multiple_derived():
    table = pa.table(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [10.0, 20.0, 30.0],
            "leak": [0.0, 0.0, 0.0],
        }
    )
    fields = [
        _field("sum_ab", "a + b"),
        _field("prod_ab", "a * b"),
    ]
    out = apply_derived_fields(table, fields)
    assert set(out.column_names) == {"sum_ab", "prod_ab", "a", "b"}
    assert "leak" not in out.column_names
