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
from factor_engine.backend.q_backend.q_process_manager import get_q_process_manager
from factor_engine.backend.context import ExecutionContext
from factor_engine.planner.logical_plan import PlanNode


@pytest.fixture
def q_available():
    """检查 q 是否可用。"""
    manager = get_q_process_manager()
    info = manager.check_availability()
    return info.is_available


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

    def test_backend_initialization(self):
        """测试后端初始化。"""
        backend = QBackend(fallback_to_pandas=False, production_mode=True)
        assert backend is not None
        assert backend.runtime_backend_label == "q_kdb"

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
            # Fallback 应该防止失败
            pytest.fail(f"Execution failed even with fallback: {e}")


class TestQBackendCapabilities:
    """能力查询测试。"""

    def test_capability_query(self):
        """测试能力查询。"""
        from factor_engine.backend.q_backend.q_capability import get_q_capability

        cap = get_q_capability()

        # 检查 Phase 1 算子
        assert cap.can_execute("ts_mean")
        assert cap.can_execute("ts_sum")
        assert cap.can_execute("cs_rank")
        assert cap.can_execute("fin_lag")

        # 检查 deferred 算子
        assert not cap.can_execute("garch")
        assert not cap.can_execute("kalman_filter")

    def test_streaming_safe_detection(self):
        """测试 streaming-safe 检测。"""
        from factor_engine.backend.q_backend.q_capability import get_q_capability

        cap = get_q_capability()

        # Streaming-safe
        ts_mean_cap = cap.get_capability("ts_mean")
        assert ts_mean_cap.streaming_safe

        # Non-streaming (需要完整组)
        cs_rank_cap = cap.get_capability("cs_rank")
        assert not cs_rank_cap.streaming_safe
        assert cs_rank_cap.requires_full_group or cs_rank_cap.requires_global_sort


class TestQBackendIntegration:
    """集成测试。"""

    def test_factory_integration(self):
        """测试 factory 集成。"""
        from factor_engine.backend.factory import build_backend

        backend = build_backend("q_kdb")
        assert isinstance(backend, QBackend)

        # 别名
        backend2 = build_backend("q")
        assert isinstance(backend2, QBackend)

    def test_capability_registry_integration(self):
        """测试 capability registry 集成。"""
        from factor_engine.backend.capability_registry import BackendKind

        assert hasattr(BackendKind, "Q_KDB")
        assert BackendKind.Q_KDB == "q_kdb"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
