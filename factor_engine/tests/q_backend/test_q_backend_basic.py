"""q/K Backend 基础功能测试。

测试 QBackend 核心执行路径：
- 基础算子执行
- Region 编译
- 数据转换
- 错误处理
"""

import pytest
import pandas as pd
import numpy as np

from factor_engine.backend.q_backend.q_backend import QBackend
from factor_engine.backend.q_backend.q_process_manager import (
    QAvailabilityStatus,
    get_q_process_manager,
)
from factor_engine.backend.context import ExecutionContext
from factor_engine.planner.logical_plan import PlanNode


@pytest.fixture
def q_available():
    """检查 q 是否可用。"""
    manager = get_q_process_manager()
    info = manager.check_availability()
    return info.status == QAvailabilityStatus.AVAILABLE


@pytest.fixture
def sample_data():
    """生成测试数据。"""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    instruments = ["AAPL", "GOOGL", "MSFT"]

    data = []
    for inst in instruments:
        for date in dates:
            data.append({
                "timestamp": date,
                "instrument": inst,
                "value": np.random.randn(),
            })

    df = pd.DataFrame(data)
    return df.set_index(["timestamp", "instrument"])


class TestQBackendBasic:
    """基础执行测试。"""

    def test_backend_initialization_fails_closed_without_q(self):
        """测试后端初始化。无真实 q 运行时，生产模式构造必须 fail-closed。"""
        from factor_engine.backend.operator_capability import BackendUnavailableError

        with pytest.raises(BackendUnavailableError):
            QBackend(fallback_to_pandas=False, production_mode=True)

    def test_stats_tracking(self):
        """测试统计信息跟踪。"""
        backend = QBackend()
        stats = backend.get_stats()

        assert "regions_compiled" in stats
        assert "regions_executed" in stats
        assert "regions_failed" in stats
        assert stats["regions_compiled"] == 0

        backend.reset_stats()
        assert backend.get_stats()["regions_compiled"] == 0

    @pytest.mark.skipif(not pytest.importorskip("pykx", reason="PyKX not available"), reason="q not available")
    def test_simple_execution_with_fallback(self, sample_data):
        """测试简单执行（允许 fallback）。"""
        backend = QBackend(fallback_to_pandas=True, production_mode=False)

        # 创建简单计划节点（只是测试框架）
        plan = PlanNode(
            node_id="test_node",
            operator="ts_mean",
            children=[],
            params={"window": 5},
        )

        ctx = ExecutionContext()
        ctx.base_data = sample_data.reset_index()

        # 执行（如果 q 不可用会 fallback）
        try:
            result = backend.execute(plan, ctx)
            assert result is not None
        except Exception as e:
            # 无 q 时 planning-time fallback（QPlanningFallbackAllowed）是显式的
            # 规划期决策，不是运行时静默降级 —— 有真实 q 时则正常执行。
            from factor_engine.backend.q_backend.q_errors import QPlanningFallbackAllowed

            if not isinstance(e, QPlanningFallbackAllowed):
                pytest.fail(f"Execution failed even with fallback: {e}")


class TestQBackendCapabilities:
    """能力查询测试。"""

    def test_capability_query(self):
        """测试能力查询。"""
        from factor_engine.backend.q_backend.q_capability import (
            QCapabilityLevel,
            get_q_capability,
        )

        cap = get_q_capability()

        # 检查 Phase 1 算子（research 模式有 lowering）
        assert cap.get_capability("ts_mean", mode="research") == QCapabilityLevel.NATIVE
        assert cap.get_capability("ts_sum", mode="research") == QCapabilityLevel.NATIVE
        assert cap.get_capability("cs_rank", mode="research") == QCapabilityLevel.NATIVE

        # 检查 deferred / 无 lowering 算子（fail-closed，绝不虚报 native）
        assert cap.get_capability("fin_lag", mode="research") == QCapabilityLevel.UNSUPPORTED
        assert cap.get_capability("garch", mode="research") == QCapabilityLevel.UNSUPPORTED
        assert cap.get_capability("kalman_filter", mode="research") == QCapabilityLevel.UNSUPPORTED

    def test_streaming_safe_detection(self):
        """测试 streaming-safe 检测。"""
        from factor_engine.backend.q_backend.q_capability import get_q_capability

        cap = get_q_capability()

        # Streaming-safe
        assert cap.is_streaming_safe("ts_mean") is True
        assert cap.is_streaming_safe("ts_sum") is True
        assert cap.is_streaming_safe("lag") is True
        assert cap.is_streaming_safe("add") is False

        # Non-streaming
        assert cap.is_streaming_safe("cs_rank") is False
        assert cap.is_streaming_safe("cs_zscore") is False


class TestQBackendIntegration:
    """集成测试。"""

    def test_factory_integration_fails_closed_without_q(self):
        """测试 factory 集成。无真实 q 时 build_backend("q_kdb") 必须 fail-closed，
        绝不返回一个伪装成功的 QBackend。"""
        from factor_engine.backend.operator_capability import BackendUnavailableError
        from factor_engine.backend.factory import build_backend

        with pytest.raises(BackendUnavailableError):
            build_backend("q_kdb")
        with pytest.raises(BackendUnavailableError):
            build_backend("q")

    def test_capability_registry_integration(self):
        """测试 capability registry 集成。"""
        from factor_engine.backend.capability_registry import BackendKind

        assert hasattr(BackendKind, "Q_KDB")
        assert BackendKind.Q_KDB == "q_kdb"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
