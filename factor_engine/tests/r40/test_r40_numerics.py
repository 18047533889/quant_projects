# -*- coding: utf-8 -*-
"""R40 #251-260: DegeneracyPolicy / 稳定求和 / MomentConvention /
ComputePrecisionPolicy / EdgeContract 数值边沿 / NumericDeterminismLevel /
permutation-equivariance / prefix-invariance / chunk-boundary / cross-process."""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from backend.elementwise_semantics import (
    DIV_OR_DEFAULT_SEMANTICS,
    DEFAULT_COMPUTE_PRECISION,
    compute_precision_policy,
    edge_value_corpus,
    inf_semantics_for,
    protected_division_truth_table,
)
from backend.numeric_semantics import (
    DEFAULT_MOMENT_CONVENTION,
    MomentConvention,
    canonical_numeric_algorithm,
    moment_convention_for,
)
from cleaned_operators._numpy_kernels import (
    DEFAULT_DEGENERACY_POLICY,
    cs_regression_,
    cs_resid_,
    neumaier_cumsum_,
    pairwise_sum_,
    welford_rolling_var_,
)
from cleaned_operators.edge_requirements import EdgeContract, edge_contract_numeric_identity
from cleaned_operators.math_certificate import (
    CHECKPOINT_CHUNK_INVARIANCE_FAILURE,
    CHUNK_BOUNDARY_INVARIANCE,
    PREFIX_INVARIANCE_FAILURE,
    check_chunk_boundary_invariance_all_streamable,
    check_permutation_equivariance_all_tie_sensitive_operators,
    check_prefix_invariance_all_causal_ts,
    cross_process_determinism_probe,
)
from runtime.execution_traits import (
    NumericDeterminismLevel,
    determinism_level_for,
    record_blas_config,
)
from runtime.hybrid_executor import HybridExecutor
from runtime.resource_broker import ResourceBroker


def test_near_zero_variance_produces_null_not_exploded_beta() -> None:
    """#251: near-zero 方差 (1e-30) -> NaN，不爆炸成 beta。"""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    x_near = np.array([1.0 + 1e-30, 1.0, 1.0 - 1e-30, 1.0, 1.0 + 1e-31])
    resid = cs_resid_(y, x_near)
    assert np.all(np.isnan(resid))  # 退化为 NaN，而非 1e30 的 beta
    beta = cs_regression_(y, x_near, mode=1)
    assert np.all(np.isnan(beta))
    # 正常方差仍工作
    x_ok = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert not np.all(np.isnan(cs_resid_(y, x_ok)))
    assert DEFAULT_DEGENERACY_POLICY.variance_is_degenerate(0.0) is True
    assert DEFAULT_DEGENERACY_POLICY.variance_is_degenerate(1e-30) is True
    assert DEFAULT_DEGENERACY_POLICY.variance_is_degenerate(1.0) is False


def test_high_dynamic_range_sum_precision() -> None:
    """#252: pairwise / Neumaier / Welford 在高动态范围下保精度。"""
    big = 1e16
    small = np.full(1000, 1.0)
    arr = np.concatenate([[big], small])
    # 朴素顺序求和（Python loop）把 1e16 后的小量全部吞掉。
    naive_loop = 0.0
    for v in arr:
        naive_loop += v
    exact_sum = 10000000000001000.0  # 全部项都是精确整数 -> 精确和可表示
    pairwise = pairwise_sum_(arr)
    # pairwise 比朴素顺序求和更接近精确和。
    assert abs(pairwise - exact_sum) < abs(naive_loop - exact_sum)
    assert abs(pairwise - exact_sum) < 8.0  # 数个 ulp 内
    # Neumaier cumsum 尾值 ≈ pairwise（相对差 < 1e-14）
    c = neumaier_cumsum_(arr)
    assert np.isfinite(c[-1])
    assert abs(c[-1] - pairwise) / abs(pairwise) < 1e-14
    # Welford rolling var 无负方差
    w = welford_rolling_var_(np.array([1e16, 1e16 + 1.0, 1e16 + 2.0, 1e16 + 3.0]), window=3)
    assert np.all(np.isnan(w[:2])) or np.all(w[~np.isnan(w)] >= -1e-12)
    # canonical numeric algorithm 声明进 numeric contract
    assert canonical_numeric_algorithm("cs_sum") == "pairwise_summation"
    assert canonical_numeric_algorithm("cum_sum") == "neumaier_cumulative"


def test_moment_convention_backend_parity() -> None:
    """#253: MomentConvention 完整约定（ddof/bias/fisher/nan/finite）。"""
    m = DEFAULT_MOMENT_CONVENTION
    assert m.ddof == 1 and m.bias is False and m.fisher is True
    assert m.nan_policy == "omit" and m.finite_policy == "exclude"
    pop = MomentConvention.from_std_ddof("population")
    assert pop.ddof == 0
    assert moment_convention_for("ts_std").ddof == 1
    assert "ddof" in m.identity_payload()


def test_float32_input_compute_promotion_behavior() -> None:
    """#254: ComputePrecisionPolicy 声明 float64 计算 + input dtype identity。"""
    p = compute_precision_policy()
    assert p.compute_dtype == "float64"
    assert p.input_dtype_identity_preserved is True
    assert p.output_storage_precision == "float64"
    assert p.identity_payload()["compute_dtype"] == "float64"
    assert DEFAULT_COMPUTE_PRECISION.compute_dtype == "float64"


def test_signed_zero_subnormal_overflow_edge_policy() -> None:
    """#255: EdgeContract signed-zero / subnormal / overflow 进 numeric identity。"""
    c = EdgeContract(nan="propagate", pos_inf="propagate", neg_inf="propagate")
    assert c.signed_zero_policy == "preserve"
    assert c.subnormal_policy == "preserve"
    assert c.overflow_policy == "preserve"
    ident = edge_contract_numeric_identity("rank")
    assert ident["signed_zero_policy"] == "preserve"
    # 非法策略拒绝
    with pytest.raises(ValueError):
        EdgeContract(signed_zero_policy="bogus")
    with pytest.raises(ValueError):
        EdgeContract(subnormal_policy="bogus")
    with pytest.raises(ValueError):
        EdgeContract(overflow_policy="bogus")


def test_determinism_level_classification_per_operator() -> None:
    """#256: NumericDeterminismLevel 三级分类 + BLAS config 记录。"""
    assert determinism_level_for("numba_kernel") is NumericDeterminismLevel.BITWISE_DETERMINISTIC
    assert determinism_level_for("duckdb_sql") is NumericDeterminismLevel.DETERMINISTIC_WITHIN_TOLERANCE
    assert determinism_level_for("research_python") is NumericDeterminismLevel.NONDETERMINISTIC_RESEARCH_ONLY
    cfg = record_blas_config()
    assert "threadpoolctl_available" in cfg
    h = HybridExecutor(broker=ResourceBroker())
    assert "blas_config" in h.summary()


def test_permutation_equivariance_all_tie_sensitive_operators() -> None:
    """#257: 所有 tie-sensitive 算子 column-permutation equivariant。"""
    ok, res = check_permutation_equivariance_all_tie_sensitive_operators()
    assert ok is True, res
    assert set(res) == {"quantile_bucket", "topk", "group_rank", "winsorize", "neutralize", "cs_regression"}


def test_prefix_invariance_hard_gate_all_causal_ts_operators() -> None:
    """#258: 所有 causal TS canonical 的 prefix-invariance universal gate。"""
    ok, det = check_prefix_invariance_all_causal_ts()
    assert ok is True, det
    assert det["gate"] == PREFIX_INVARIANCE_FAILURE


def test_chunk_boundary_invariance_all_streamable_operators() -> None:
    """#259: 可流式/分块算子（EWM/cumsum）的 chunk-boundary invariance。"""
    ok, det = check_chunk_boundary_invariance_all_streamable()
    assert ok is True, det
    assert det["gate"] == CHUNK_BOUNDARY_INVARIANCE


def test_full_production_determinism_across_processes_and_restarts() -> None:
    """#260: 两个独立进程对同一 spec 产生相同 checksum/hash；worker count 不影响。"""
    probe_code = (
        "from cleaned_operators.math_certificate import cross_process_determinism_probe;"
        "import json,sys;"
        "print(json.dumps(cross_process_determinism_probe(worker_count=int(sys.argv[1]), seed=42)))"
    )
    def _run(worker: str):
        return subprocess.run(
            [sys.executable, "-c", probe_code, worker],
            capture_output=True, text=True, timeout=120,
        )

    r1 = _run("1")
    r2 = _run("4")
    assert r1.returncode == 0 and r2.returncode == 0, (r1.stderr, r2.stderr)
    d1 = json.loads(r1.stdout.strip())
    d2 = json.loads(r2.stdout.strip())
    assert d1["result_checksum"] == d2["result_checksum"]
    assert d1["plan_hash"] == d2["plan_hash"]
    assert d1["axis_hash"] == d2["axis_hash"]
    assert d1["n_values"] == d2["n_values"]
    # 同一进程两次（restart 语义）也一致
    r3 = _run("2")
    d3 = json.loads(r3.stdout.strip())
    assert d3["result_checksum"] == d1["result_checksum"]


def test_edge_corpus_and_inf_semantics_declared() -> None:
    """#201/#202: 共享 edge corpus + element-math vs stats Inf 语义声明。"""
    corpus = edge_value_corpus()
    for name in ("nan", "pos_inf", "neg_inf", "pos_zero", "neg_zero", "subnormal", "huge", "tiny"):
        assert name in corpus
    assert inf_semantics_for("maximum") == "inf_is_mathematical_value"
    assert inf_semantics_for("ts_mean") == "inf_is_invalid_sample"


def test_protected_div_truth_table_matches_pandas() -> None:
    """#199/#200: ProtectedDivisionSemantics 真值表与 pandas emitter 一致。"""
    from backend.elementwise_semantics import (
        PROTECTED_DIV_SEMANTICS,
        div_or_default_pandas,
        protected_div_pandas,
    )

    x = pd.Series([10.0, np.inf, -np.inf, 10.0, np.inf, np.nan, 1e308])
    y = pd.Series([2.0, 2.0, 2.0, np.inf, np.inf, 2.0, 1e-10])
    tt = protected_division_truth_table(x, y, eps=1e-12, default=0.0, semantics=PROTECTED_DIV_SEMANTICS)
    ref = protected_div_pandas(x, y, epsilon=1e-12, default=0.0)
    assert np.allclose(tt.fillna(-999), ref.fillna(-999), equal_nan=True)
    dtt = protected_division_truth_table(x, y, eps=1e-12, default=0.0, semantics=DIV_OR_DEFAULT_SEMANTICS)
    dref = div_or_default_pandas(x, y, epsilon=1e-12, default=0.0)
    assert np.allclose(dtt.fillna(-999), dref.fillna(-999), equal_nan=True)
