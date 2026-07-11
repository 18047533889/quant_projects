"""
gateway_core.py 的单元测试
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from gateway.config import GatewayConfig
from gateway.gateway_core import GatewayCore, quick_process, check_candidate
from gateway.models import GatewayLabel


def create_test_candidate(
        expr: str = "MA(close, 5)",
        candidate_id: str = None,
        config_dict: dict = None
) -> dict:
    """创建测试用的候选因子字典"""
    if candidate_id is None:
        candidate_id = f"cand_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if config_dict is None:
        config_dict = {"window": 5}

    return {
        "schema_version": "disk.v1",
        "candidate_id": candidate_id,
        "Expr": expr,
        "Config": config_dict,
        "BornTimestamp": datetime.now(timezone.utc).isoformat(),
        "BasicInfo": {
            "source": "test",
            "creator": "unit_test"
        }
    }


def test_pass_result():
    """测试 Pass 结果"""
    config = GatewayConfig()
    gateway = GatewayCore(config)

    # 创建一个合法的候选因子
    candidate = create_test_candidate("MA(close, 5)")

    result = gateway.process(candidate)

    print(f"\nPass 测试结果: {result.label.value}")
    print(f"  run_id: {result.run_id}")
    print(f"  checked_at: {result.checked_at}")
    print(f"  complexity_score: {result.complexity_score}")

    assert result.label == GatewayLabel.PASS
    assert result.reason == ""
    assert result.is_legal is True
    assert result.has_future is False
    assert result.is_within_budget is True

    print("✅ Pass 结果测试通过")


def test_rejected_invalid_operator():
    """测试非法算子导致拒绝"""
    config = GatewayConfig()
    gateway = GatewayCore(config)

    # 创建包含非法算子的候选因子
    candidate = create_test_candidate("INVALID_FUNC(close, 5)")

    result = gateway.process(candidate)

    print(f"\n非法算子测试结果: {result.label.value}")
    print(f"  reason: {result.reason[:80]}...")

    assert result.label == GatewayLabel.REJECTED
    assert "Operator" in result.reason or "whitelist" in result.reason.lower()

    print("✅ 非法算子拒绝测试通过")


def test_rejected_future_function():
    """测试未来函数导致拒绝"""
    config = GatewayConfig()
    gateway = GatewayCore(config)

    # 创建包含未来函数的候选因子
    candidate = create_test_candidate("REFX(close, 1)")

    result = gateway.process(candidate)

    print(f"\n未来函数测试结果: {result.label.value}")
    print(f"  reason: {result.reason[:80]}...")

    assert result.label == GatewayLabel.REJECTED
    assert "Future" in result.reason or "REFX" in result.reason

    print("✅ 未来函数拒绝测试通过")


def test_rejected_complexity():
    """测试复杂度过高导致拒绝"""
    config = GatewayConfig()
    # 临时降低阈值以触发拒绝
    config.max_complexity = 1.0
    gateway = GatewayCore(config)

    # 创建复杂度较高的候选因子
    candidate = create_test_candidate("CORR(MA(close, 5), STD(close, 20))")

    result = gateway.process(candidate)

    print(f"\n复杂度超限测试结果: {result.label.value}")
    print(f"  reason: {result.reason[:80]}...")
    print(f"  complexity_score: {result.complexity_score}")

    assert result.label == GatewayLabel.REJECTED
    assert "Complexity" in result.reason or "budget" in result.reason.lower()

    # 恢复阈值
    config.max_complexity = 20.0
    print("✅ 复杂度超限拒绝测试通过")


def test_duplicated_duplicate():
    """测试重复因子导致 Duplicated"""
    config = GatewayConfig()
    gateway = GatewayCore(config)

    expr = "MA(close, 10)"

    # 第一次处理 - 应该 Pass
    candidate1 = create_test_candidate(expr, "cand_001")
    result1 = gateway.process(candidate1)
    assert result1.label == GatewayLabel.PASS

    # 第二次处理相同表达式 - 应该 Duplicated（重复）
    candidate2 = create_test_candidate(expr, "cand_002")
    result2 = gateway.process(candidate2)

    print(f"\n重复因子测试结果: {result2.label.value}")
    print(f"  reason: {result2.reason[:80]}...")
    print(f"  is_duplicate: {result2.is_duplicate}")
    print(f"  match_type: {getattr(result2, 'historical_report_id', 'N/A')}")

    assert result2.label == GatewayLabel.DUPLICATED
    assert result2.is_duplicate is True

    print("✅ 重复因子 Duplicated 测试通过")


def test_batch_processing():
    """测试批量处理"""
    config = GatewayConfig()
    gateway = GatewayCore(config)

    # 创建多个候选因子
    candidates = [
        create_test_candidate("MA(close, 5)", "cand_001"),
        create_test_candidate("STD(close, 20)", "cand_002"),
        create_test_candidate("INVALID_FUNC(close)", "cand_003"),
        create_test_candidate("REFX(close, 1)", "cand_004"),
        create_test_candidate("MA(close, 5)", "cand_005"),  # 重复
    ]

    results = gateway.process_batch(candidates)

    print(f"\n批量处理结果:")
    print(f"  总数: {len(results)}")
    print(f"  Pass: {sum(1 for r in results if r.label == GatewayLabel.PASS)}")
    print(f"  Duplicated: {sum(1 for r in results if r.label == GatewayLabel.DUPLICATED)}")
    print(f"  Rejected: {sum(1 for r in results if r.label == GatewayLabel.REJECTED)}")

    assert len(results) == 5
    assert sum(1 for r in results if r.label == GatewayLabel.PASS) >= 2
    assert sum(1 for r in results if r.label == GatewayLabel.DUPLICATED) >= 1
    assert sum(1 for r in results if r.label == GatewayLabel.REJECTED) >= 2

    print("✅ 批量处理测试通过")


def test_stats():
    """测试统计功能"""
    config = GatewayConfig()
    gateway = GatewayCore(config)

    # 处理几个候选因子
    candidates = [
        create_test_candidate("MA(close, 5)", "cand_001"),
        create_test_candidate("INVALID_FUNC(close)", "cand_002"),
        create_test_candidate("MA(close, 5)", "cand_003"),  # 重复
    ]

    for candidate in candidates:
        gateway.process(candidate)

    stats = gateway.get_stats()

    print(f"\n统计信息:")
    print(f"  total_processed: {stats['total_processed']}")
    print(f"  pass_count: {stats['pass_count']}")
    print(f"  duplicated_count: {stats['duplicated_count']}")
    print(f"  rejected_count: {stats['rejected_count']}")

    assert stats['total_processed'] == 3
    assert stats['pass_count'] >= 1
    assert stats['duplicated_count'] >= 1
    assert stats['rejected_count'] >= 1

    # 测试重置
    gateway.reset_stats()
    stats_reset = gateway.get_stats()
    assert stats_reset['total_processed'] == 0

    print("✅ 统计功能测试通过")


def test_convenience_functions():
    """测试便捷函数"""
    candidate = create_test_candidate("MA(close, 5)")

    # 测试 quick_process
    result = quick_process(candidate)
    print(f"\nquick_process 结果: {result.label.value}")
    assert isinstance(result.label, GatewayLabel)

    # 测试 check_candidate
    can_pass, reason = check_candidate(candidate)
    print(f"check_candidate: can_pass={can_pass}, reason='{reason}'")
    assert can_pass is True
    assert reason == ""

    # 测试非法因子
    bad_candidate = create_test_candidate("INVALID_FUNC(close)")
    can_pass, reason = check_candidate(bad_candidate)
    print(f"check_candidate (非法): can_pass={can_pass}, reason='{reason[:50]}...'")
    assert can_pass is False

    print("✅ 便捷函数测试通过")


if __name__ == "__main__":
    test_pass_result()
    test_rejected_invalid_operator()
    test_rejected_future_function()
    test_rejected_complexity()
    test_duplicated_duplicate()
    test_batch_processing()
    test_stats()
    test_convenience_functions()

    print("\n" + "=" * 50)
    print("🎉 所有网关核心测试通过！")
    print("=" * 50)