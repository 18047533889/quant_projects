# -*- coding: utf-8 -*-
"""identity 包测试：覆盖任务书 §49-§53 全部断言 + 自有断言。

运行：
    cd /home/sunhaiwei/quant_projects/factor_engine
    python3 -m pytest tests/test_identity.py -q

依赖说明：
- 解析依赖 cleaned_operators 注册表（首次导入加载约 10-35s，线程安全）；
- 无 polars/torch 的机器会有注册警告，不影响断言。
"""

from __future__ import annotations

import pytest

from factor_engine.identity import get_factor_identity, FactorIdentityError
from factor_engine.identity import hasher
from factor_engine.identity.subtree import compute_merkle_root, compute_subtree_hashes


# ---------------------------------------------------------------------------
# §49 基本身份字段
# ---------------------------------------------------------------------------

class TestFactorIdentityFields:
    def test_basic_fields_present(self):
        ident = get_factor_identity("ts_mean(close, 20)")
        assert isinstance(ident.canonical_dsl, str)
        assert len(ident.canonical_ast_hash) == 64  # SHA-256 hex full length
        assert isinstance(ident.signal_equivalence_id, str)
        assert len(ident.signal_equivalence_id) == 64
        assert ident.orientation in (1, -1)
        assert isinstance(ident.subtree_hashes, tuple)
        assert ident.subtree_hashes
        assert isinstance(ident.field_signature, str)
        assert isinstance(ident.operator_signature, str)
        assert isinstance(ident.complexity, int) and ident.complexity > 0
        assert isinstance(ident.depth, int) and ident.depth > 0
        assert isinstance(ident.lookback, int) and ident.lookback >= 0
        assert ident.identity_version == "factor_identity_v1"
        assert ident.operator_semantics_version == "v1"

    def test_frozen(self):
        ident = get_factor_identity("ts_mean(close, 20)")
        with pytest.raises(Exception):
            ident.canonical_ast_hash = "x"

    def test_identity_version_present(self):
        ident = get_factor_identity("ts_mean(close, 20)")
        assert ident.identity_version == "factor_identity_v1"

    def test_complexity_depth_lookback(self):
        ident = get_factor_identity("ts_mean(close, 20)")
        # ts_mean + close + 20 = 3 nodes
        assert ident.complexity == 3
        assert ident.depth == 2
        assert ident.lookback == 20


# ---------------------------------------------------------------------------
# §50 canonical DSL 与 AST hash
# ---------------------------------------------------------------------------

class TestCanonicalDsl:
    def test_canonical_dsl_is_json(self):
        import json

        ident = get_factor_identity("ts_mean(close, 20)")
        payload = json.loads(ident.canonical_dsl)
        assert payload["kind"] == "call"
        assert payload["op"] == "ts_mean"

    def test_canonical_ast_hash_is_sha256(self):
        ident = get_factor_identity("ts_mean(close, 20)")
        # 验证 hash = sha256(namespace + canonical_dsl)
        expected = hasher.compute_hash(ident.canonical_dsl)
        assert ident.canonical_ast_hash == expected

    def test_field_canonical_consistency(self):
        """close 与 AdjClose 同身份（field 已解析成规范名）。"""
        i1 = get_factor_identity("close")
        i2 = get_factor_identity("AdjClose")
        assert i1.canonical_ast_hash == i2.canonical_ast_hash
        assert i1.canonical_dsl == i2.canonical_dsl

    def test_operator_alias_canonical_consistency(self):
        """算子别名（ts_mean / TS_MEAN）同身份。"""
        i1 = get_factor_identity("ts_mean(close, 20)")
        i2 = get_factor_identity("TS_MEAN(close, 20)")
        assert i1.canonical_ast_hash == i2.canonical_ast_hash


# ---------------------------------------------------------------------------
# §51 化简规则
# ---------------------------------------------------------------------------

class TestSimplificationRules:
    @pytest.mark.parametrize(
        "given, expected",
        [
            # 括号壳（AST 层天然）
            ("ts_mean(close, 20)", "ts_mean(close, 20)"),
            # unary + 去除
            ("+ts_mean(close, 20)", "ts_mean(close, 20)"),
            # x*1 / 1*x / x/1 / x+0 / 0+x / x-0 → x
            ("ts_mean(close, 20) * 1", "ts_mean(close, 20)"),
            ("1 * ts_mean(close, 20)", "ts_mean(close, 20)"),
            ("ts_mean(close, 20) / 1", "ts_mean(close, 20)"),
            ("ts_mean(close, 20) + 0", "ts_mean(close, 20)"),
            ("0 + ts_mean(close, 20)", "ts_mean(close, 20)"),
            ("ts_mean(close, 20) - 0", "ts_mean(close, 20)"),
            # 0-x → -x
            ("0 - ts_mean(close, 20)", "-ts_mean(close, 20)"),
            # x*(-1) / (-1)*x → -x
            ("ts_mean(close, 20) * (-1)", "-ts_mean(close, 20)"),
            ("(-1) * ts_mean(close, 20)", "-ts_mean(close, 20)"),
            # -(-x) → x
            ("-(-ts_mean(close, 20))", "ts_mean(close, 20)"),
            # pow(x,1) → x
            ("power(close, 1)", "close"),
        ],
    )
    def test_simplification(self, given, expected):
        ig = get_factor_identity(given)
        ie = get_factor_identity(expected)
        assert ig.canonical_ast_hash == ie.canonical_ast_hash
        assert ig.canonical_dsl == ie.canonical_dsl

    def test_dangerous_x_over_x_not_simplified(self):
        """§8：x/x 不得变 1。"""
        i_div = get_factor_identity("close / close")
        i_one = get_factor_identity("close")
        assert i_div.canonical_ast_hash != i_one.canonical_ast_hash

    def test_dangerous_log_exp_not_simplified(self):
        """§8：log(exp(x)) 不得变 x。"""
        i_identity = get_factor_identity("close")
        i_naive = get_factor_identity("log(exp(close))")
        assert i_identity.canonical_ast_hash != i_naive.canonical_ast_hash

    def test_commutative_add(self):
        i1 = get_factor_identity("add(ts_mean(close, 20), ts_mean(high, 20))")
        i2 = get_factor_identity("add(ts_mean(high, 20), ts_mean(close, 20))")
        assert i1.canonical_ast_hash == i2.canonical_ast_hash

    def test_commutative_multiply(self):
        i1 = get_factor_identity("multiply(ts_mean(close, 20), ts_mean(high, 20))")
        i2 = get_factor_identity("multiply(ts_mean(high, 20), ts_mean(close, 20))")
        assert i1.canonical_ast_hash == i2.canonical_ast_hash

    def test_sub_not_commutative(self):
        i1 = get_factor_identity("subtract(ts_mean(close, 20), ts_mean(high, 20))")
        i2 = get_factor_identity("subtract(ts_mean(high, 20), ts_mean(close, 20))")
        assert i1.canonical_ast_hash != i2.canonical_ast_hash

    def test_divide_not_commutative(self):
        i1 = get_factor_identity("divide(ts_mean(close, 20), ts_mean(high, 20))")
        i2 = get_factor_identity("divide(ts_mean(high, 20), ts_mean(close, 20))")
        assert i1.canonical_ast_hash != i2.canonical_ast_hash

    def test_constant_folding(self):
        i = get_factor_identity("add(10, 20)")
        assert i.canonical_dsl == '{"kind":"literal","value":30.0}'

    def test_constant_folding_multiply(self):
        i = get_factor_identity("multiply(2, 3)")
        assert i.canonical_dsl == '{"kind":"literal","value":6.0}'

    def test_idempotent_canonicalization(self):
        """幂等：get_factor_identity(canonical_dsl) == get_factor_identity(原始)。

        canonical_dsl 字段本身保持稳定。
        """
        i1 = get_factor_identity("ts_mean(close, 20) + 0")
        i2 = get_factor_identity(i1.canonical_dsl)
        assert i1.canonical_ast_hash == i2.canonical_ast_hash
        assert i1.canonical_dsl == i2.canonical_dsl


# ---------------------------------------------------------------------------
# §52 sign normalizer
# ---------------------------------------------------------------------------

class TestSignNormalizer:
    @pytest.mark.parametrize(
        "formula",
        [
            "-ts_mean(close, 20)",
            "0 - ts_mean(close, 20)",
            "(-1) * ts_mean(close, 20)",
            "ts_mean(close, 20) * (-1)",
        ],
    )
    def test_sign_forms_same_signal_id(self, formula):
        i_neg = get_factor_identity(formula)
        i_pos = get_factor_identity("ts_mean(close, 20)")
        assert i_neg.orientation == -1
        assert i_pos.orientation == 1
        assert i_neg.signal_equivalence_id == i_pos.signal_equivalence_id

    def test_sign_forms_same_hash(self):
        """四形 neg 写法的 canonical_ast_hash 相同（-f 归一化）。"""
        forms = [
            "-ts_mean(close, 20)",
            "0 - ts_mean(close, 20)",
            "(-1) * ts_mean(close, 20)",
            "ts_mean(close, 20) * (-1)",
        ]
        hashes = {get_factor_identity(f).canonical_ast_hash for f in forms}
        assert len(hashes) == 1

    def test_no_aggressive_algebra(self):
        """§17：A-B 不判等 -(B-A)。"""
        i1 = get_factor_identity("subtract(close, high)")
        i2 = get_factor_identity("neg(subtract(high, close))")
        assert i1.signal_equivalence_id != i2.signal_equivalence_id

    def test_orientation_of_double_neg(self):
        """-(-f) → f，orientation 为 +1。"""
        i = get_factor_identity("-(-ts_mean(close, 20))")
        assert i.orientation == 1


# ---------------------------------------------------------------------------
# §53 parameter family
# ---------------------------------------------------------------------------

class TestParameterFamily:
    def test_same_family_different_hash(self):
        i19 = get_factor_identity("ts_mean(close, 19)")
        i20 = get_factor_identity("ts_mean(close, 20)")
        i21 = get_factor_identity("ts_mean(close, 21)")
        assert i19.parameter_family_id == i20.parameter_family_id == i21.parameter_family_id
        assert i19.canonical_ast_hash != i20.canonical_ast_hash
        assert i20.canonical_ast_hash != i21.canonical_ast_hash

    def test_different_family_same_role(self):
        i1 = get_factor_identity("ts_mean(close, 20)")
        i2 = get_factor_identity("ts_std(close, 20)")
        assert i1.parameter_family_id != i2.parameter_family_id

    def test_power_not_familied(self):
        """POWER 保守不 family 化。"""
        i1 = get_factor_identity("power(close, 2)")
        i2 = get_factor_identity("power(close, 3)")
        assert i1.parameter_family_id != i2.parameter_family_id


# ---------------------------------------------------------------------------
# subtree / merkle
# ---------------------------------------------------------------------------

class TestSubtree:
    def test_merkle_root_equals_canonical_ast_hash(self):
        ident = get_factor_identity("ts_mean(close, 20)")
        from factor_engine.api.dsl_parser import parse_expr
        from factor_engine.identity.canonicalizer import canonicalize

        ast = canonicalize(parse_expr("ts_mean(close, 20)"))
        root = compute_merkle_root(ast)
        assert root == ident.canonical_ast_hash

    def test_subtree_hashes_include_root_last(self):
        ident = get_factor_identity("ts_mean(close, 20)")
        assert ident.subtree_hashes[-1] == ident.canonical_ast_hash

    def test_subtree_hashes_count(self):
        """叶子(2) + 根(1) = 3 个去重子树哈希。"""
        ident = get_factor_identity("ts_mean(close, 20)")
        # close、20、ts_mean 三个节点（去重后可能 3 个）
        assert len(ident.subtree_hashes) == 3

    def test_subtree_hashes_change_with_formula(self):
        i1 = get_factor_identity("ts_mean(close, 20)")
        i2 = get_factor_identity("ts_std(close, 20)")
        assert i1.subtree_hashes != i2.subtree_hashes


# ---------------------------------------------------------------------------
# hasher / namespace
# ---------------------------------------------------------------------------

class TestHasher:
    def test_namespace_changes_hash(self):
        h1 = hasher.compute_hash("ts_mean(close, 20)")
        h2 = hasher.compute_hash("other_ns:ts_mean(close, 20)")
        assert h1 != h2

    def test_hash_length(self):
        assert len(hasher.compute_hash("x")) == 64

    def test_factor_id_derivation(self):
        h = hasher.compute_hash("ts_mean(close, 20)")
        fid = hasher.derive_factor_id(h)
        assert fid.startswith("F_")
        # "F_" + 11 base32 字符 = 13
        assert len(fid) == 13
    def test_custom_namespace(self):
        h1 = hasher.compute_namespaced_hash("ns1", "data")
        h2 = hasher.compute_namespaced_hash("ns2", "data")
        assert h1 != h2


# ---------------------------------------------------------------------------
# 解析失败
# ---------------------------------------------------------------------------

class TestErrors:
    def test_parse_failure_raises_factor_identity_error(self):
        with pytest.raises(FactorIdentityError):
            get_factor_identity("ts_mean(close, ")  # 语法错误

    def test_unknown_operator_raises(self):
        with pytest.raises(FactorIdentityError):
            get_factor_identity("no_such_operator(close, 20)")

    def test_error_has_formula_and_reason(self):
        try:
            get_factor_identity("bad syntax(")
        except FactorIdentityError as e:
            assert "bad syntax(" in str(e.formula)
            assert e.reason
        else:
            pytest.fail("should have raised")


# ---------------------------------------------------------------------------
# 追加：其他
# ---------------------------------------------------------------------------

class TestExtra:
    def test_formula_whitespace_insensitive(self):
        i1 = get_factor_identity("ts_mean(close,20)")
        i2 = get_factor_identity("ts_mean( close , 20 )")
        assert i1.canonical_ast_hash == i2.canonical_ast_hash

    def test_parameter_family_threshold_config(self):
        """THRESHOLD 可配置：family_threshold=False 时参数保留。"""
        from factor_engine.identity.api import get_factor_identity as _gf
        from factor_engine.identity.parameter_family import (
            ParameterFamilyConfig,
            parameter_family_id,
        )
        from factor_engine.api.dsl_parser import parse_expr
        from factor_engine.identity.canonicalizer import canonicalize

        ast = canonicalize(parse_expr("ts_mean(close, 20)"))
        fid_on = parameter_family_id(
            ast, config=ParameterFamilyConfig(family_threshold=True)
        )
        fid_off = parameter_family_id(
            ast, config=ParameterFamilyConfig(family_threshold=False)
        )
        assert fid_on == fid_off  # ts_mean 的 20 是 WINDOW 参数，不受 THRESHOLD 配置影响

    def test_subtree_hashes_deduped(self):
        """去重：重复子树只出现一次。"""
        ident = get_factor_identity("ts_mean(close, 20)")
        assert len(ident.subtree_hashes) == len(set(ident.subtree_hashes))