"""Q Backend Unit Tests.

测试 Q backend 各组件的基本功能。
"""

from __future__ import annotations

import pytest

from backend.q_backend.q_capability import (
    QBackendCapability,
    QCapabilityLevel,
    get_q_capability,
)
from backend.q_backend.q_compiler import QCompiler, get_q_compiler
from backend.q_backend.q_process_manager import (
    QAvailabilityStatus,
    check_q_availability,
    is_q_available,
)


class TestQCapability:
    """测试 Q capability 查询。"""

    def test_get_capability_singleton(self):
        """测试单例获取。"""
        cap1 = get_q_capability()
        cap2 = get_q_capability()
        assert cap1 is cap2

    def test_native_operators(self):
        """测试原生算子支持。"""
        cap = get_q_capability()

        # Phase 1 native ops
        assert cap.supports_native("add")
        assert cap.supports_native("ts_mean")
        assert cap.supports_native("lag")
        assert cap.supports_native("cs_rank")

    def test_unsupported_operators(self):
        """测试不支持的算子。"""
        cap = get_q_capability()

        # Deferred ops
        assert not cap.supports_native("garch")
        assert not cap.supports_native("kalman_filter")
        assert not cap.supports_native("network_centrality")

    def test_streaming_safe_operators(self):
        """测试流式安全算子。"""
        cap = get_q_capability()

        # Streaming safe
        assert cap.is_streaming_safe("ts_mean")
        assert cap.is_streaming_safe("lag")
        assert cap.is_streaming_safe("add")

        # Not streaming safe
        assert not cap.is_streaming_safe("cs_rank")

    def test_requires_full_group(self):
        """测试需要完整分组的算子。"""
        cap = get_q_capability()

        assert cap.requires_full_group("cs_rank")
        assert cap.requires_full_group("cs_zscore")
        assert cap.requires_full_group("group_mean")

        assert not cap.requires_full_group("add")
        assert not cap.requires_full_group("ts_mean")

    def test_compute_native_fraction(self):
        """测试原生支持比例计算。"""
        cap = get_q_capability()

        ops = ["add", "ts_mean", "cs_rank", "garch"]
        fraction, native, unsupported = cap.compute_native_fraction(ops)

        assert len(native) == 3  # add, ts_mean, cs_rank
        assert len(unsupported) == 1  # garch
        assert fraction == 0.75

    def test_get_capability_levels(self):
        """测试不同能力级别查询。"""
        cap = get_q_capability()

        # Native
        assert cap.get_capability("add") == QCapabilityLevel.NATIVE

        # Unsupported
        assert cap.get_capability("garch") == QCapabilityLevel.UNSUPPORTED

        # Unknown
        assert cap.get_capability("unknown_op_xyz") == QCapabilityLevel.UNSUPPORTED


class TestQCompiler:
    """测试 Q 编译器。"""

    def test_get_compiler_singleton(self):
        """测试单例获取。"""
        compiler1 = get_q_compiler()
        compiler2 = get_q_compiler()
        assert compiler1 is compiler2

    def test_can_compile_operator(self):
        """测试算子编译能力查询。"""
        compiler = get_q_compiler()

        assert compiler.can_compile_operator("add")
        assert compiler.can_compile_operator("ts_mean")
        assert not compiler.can_compile_operator("garch")

    def test_compile_arithmetic_operator(self):
        """测试算术算子编译。"""
        compiler = get_q_compiler()

        # Binary ops
        assert compiler.compile_operator("add", ["x", "y"]) == "x + y"
        assert compiler.compile_operator("subtract", ["x", "y"]) == "x - y"
        assert compiler.compile_operator("multiply", ["x", "y"]) == "x * y"
        assert compiler.compile_operator("divide", ["x", "y"]) == "x % y"

        # Unary ops
        assert compiler.compile_operator("negate", ["x"]) == "neg x"
        assert compiler.compile_operator("abs", ["x"]) == "abs x"
        assert compiler.compile_operator("sqrt", ["x"]) == "sqrt x"

    def test_compile_lag_operator(self):
        """测试 lag 算子编译。"""
        compiler = get_q_compiler()

        # Default lag (1 period)
        result = compiler.compile_operator("lag", ["x"], {"periods": 1})
        assert result == "prev x"

        # Multi-period lag
        result = compiler.compile_operator("lag", ["x"], {"periods": 5})
        assert result == "5 prev\\x"

    def test_compile_rolling_operator(self):
        """测试滚动窗口算子编译。"""
        compiler = get_q_compiler()

        # ts_mean with window
        result = compiler.compile_operator("ts_mean", ["x"], {"window": 20})
        assert result == "20 mavg x"

        # ts_std
        result = compiler.compile_operator("ts_std", ["x"], {"window": 10})
        assert result == "10 mdev x"

    def test_compile_correlation_operator(self):
        """测试相关性算子编译。"""
        compiler = get_q_compiler()

        result = compiler.compile_operator("ts_corr", ["x", "y"], {"window": 20})
        assert result == "20 cor[x;y]"

        result = compiler.compile_operator("ts_cov", ["x", "y"], {"window": 15})
        assert result == "15 cov[x;y]"

    def test_compile_rank_operator(self):
        """测试排名算子编译。"""
        compiler = get_q_compiler()

        result = compiler.compile_operator("rank", ["x"])
        assert result == "rank x"

        result = compiler.compile_operator("cs_rank", ["x"])
        # cs_rank uses lambda form which gets applied in the context
        assert result == "{iasc iasc x}"

    def test_compile_aggregation_operator(self):
        """测试聚合算子编译。"""
        compiler = get_q_compiler()

        assert compiler.compile_operator("mean", ["x"]) == "avg x"
        assert compiler.compile_operator("sum", ["x"]) == "sum x"
        assert compiler.compile_operator("std", ["x"]) == "dev x"
        assert compiler.compile_operator("min", ["x"]) == "min x"
        assert compiler.compile_operator("max", ["x"]) == "max x"

    def test_validate_region(self):
        """测试 region 验证。"""
        compiler = get_q_compiler()

        # Valid region
        nodes = [
            {"operator": "add", "id": "n1"},
            {"operator": "ts_mean", "id": "n2"},
        ]
        is_valid, unsupported = compiler.validate_region(nodes)
        assert is_valid
        assert len(unsupported) == 0

        # Invalid region
        nodes = [
            {"operator": "add", "id": "n1"},
            {"operator": "garch", "id": "n2"},
        ]
        is_valid, unsupported = compiler.validate_region(nodes)
        assert not is_valid
        assert "garch" in unsupported

    def test_compile_region(self):
        """测试 region 编译。"""
        compiler = get_q_compiler()

        nodes = [
            {
                "id": "n1",
                "operator": "add",
                "inputs": ["input1", "input2"],
                "params": {},
            },
            {
                "id": "n2",
                "operator": "ts_mean",
                "inputs": ["n1"],
                "params": {"window": 20},
            },
        ]

        plan = compiler.compile_region(
            region_id="test_region",
            nodes=nodes,
            input_tables=["input1", "input2"],
            output_name="result",
        )

        assert plan.region_id == "test_region"
        assert plan.node_ids == ("n1", "n2")
        assert "n1: input1 + input2;" in plan.q_code
        assert "n2: 20 mavg n1;" in plan.q_code
        assert "result: n2" in plan.q_code


class TestQProcessManager:
    """测试 Q 进程管理器。"""

    def test_check_availability(self):
        """测试可用性检查。"""
        info = check_q_availability()

        assert info.status in {
            QAvailabilityStatus.AVAILABLE,
            QAvailabilityStatus.UNAVAILABLE,
            QAvailabilityStatus.LICENSE_MISSING,
            QAvailabilityStatus.PROCESS_FAILED,
        }

        if info.status == QAvailabilityStatus.AVAILABLE:
            assert info.version is not None
            assert info.pykx_version is not None
            print(f"\nQ Available: version={info.version}, pykx={info.pykx_version}")
        else:
            print(f"\nQ Unavailable: {info.status.value} - {info.error_message}")

    def test_is_available_function(self):
        """测试便捷可用性函数。"""
        available = is_q_available()
        assert isinstance(available, bool)

        info = check_q_availability()
        assert available == (info.status == QAvailabilityStatus.AVAILABLE)


class TestQBackendIntegration:
    """测试 Q backend 集成。"""

    def test_import_q_backend(self):
        """测试 Q backend 导入。"""
        from backend.q_backend import QBackend, get_q_backend

        backend = get_q_backend()
        assert isinstance(backend, QBackend)

    def test_q_backend_initialization(self):
        """测试 Q backend 初始化。"""
        from backend.q_backend import QBackend

        backend = QBackend(fallback_to_pandas=True, production_mode=False)
        assert backend.runtime_backend_label == "q_kdb"
        assert not backend.prefers_native_scan
        assert not backend.supports_lazy_shared

    def test_q_backend_stats(self):
        """测试 Q backend 统计。"""
        from backend.q_backend import QBackend

        backend = QBackend()
        stats = backend.get_stats()

        assert "regions_compiled" in stats
        assert "regions_executed" in stats
        assert "regions_failed" in stats
        assert "fallback_count" in stats
        assert "total_rows_processed" in stats
        assert "total_execution_time_ms" in stats

        # Initial stats should be zero
        assert stats["regions_compiled"] == 0
        assert stats["regions_executed"] == 0

        # Test reset
        backend.reset_stats()
        stats = backend.get_stats()
        assert stats["regions_compiled"] == 0


class TestCapabilityRegistryIntegration:
    """测试 capability registry 与 Q backend 集成。"""

    def test_q_backend_in_registry(self):
        """测试 Q backend 在 registry 中注册。"""
        from backend.capability_registry import BackendKind

        assert BackendKind.Q_KDB in BackendKind

    def test_query_q_capability(self):
        """测试通过 registry 查询 Q capability。"""
        from backend.capability_registry import BackendCapabilityRegistry, BackendKind

        # Query Q backend capability for a native operator
        result = BackendCapabilityRegistry.query(
            "add",
            BackendKind.Q_KDB,
            mode="production",
        )

        if is_q_available():
            # If Q is available, native ops should be supported
            assert result.supported or result.reason
        else:
            # If Q is not available, expect appropriate message
            print(f"Q capability query: {result.reason}")


class TestQFinancialOperators:
    """测试金融特定算子。"""

    def test_fin_lag_compilation(self):
        """测试 fin_lag 编译。"""
        compiler = get_q_compiler()

        result = compiler.compile_operator("fin_lag", ["price"], {"periods": 1})
        assert "prev" in result

    def test_fin_delta_compilation(self):
        """测试 fin_delta 编译。"""
        compiler = get_q_compiler()

        result = compiler.compile_operator("fin_delta", ["price"])
        assert "deltas" in result

    def test_financial_operators_coverage(self):
        """测试金融算子覆盖。"""
        cap = get_q_capability()

        financial_ops = ["fin_lag", "fin_delta", "fin_returns"]
        for op in financial_ops:
            assert cap.supports_native(op), f"{op} should be native"


class TestQCrossSectionOperators:
    """测试截面算子。"""

    def test_cs_operators_compilation(self):
        """测试截面算子编译。"""
        compiler = get_q_compiler()

        # cs_zscore
        result = compiler.compile_operator("cs_zscore", ["x"])
        assert "avg" in result or "dev" in result

        # cs_demean
        result = compiler.compile_operator("cs_demean", ["x"])
        assert "avg" in result

    def test_cs_operators_coverage(self):
        """测试截面算子覆盖。"""
        cap = get_q_capability()

        cs_ops = ["cs_rank", "cs_zscore", "cs_demean", "cs_winsorize", "cs_quantile"]
        for op in cs_ops:
            assert cap.supports_native(op), f"{op} should be native"

    def test_cs_operators_require_full_group(self):
        """测试截面算子需要完整分组。"""
        cap = get_q_capability()

        cs_ops = ["cs_rank", "cs_zscore", "cs_demean"]
        for op in cs_ops:
            assert cap.requires_full_group(op), f"{op} should require full group"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
