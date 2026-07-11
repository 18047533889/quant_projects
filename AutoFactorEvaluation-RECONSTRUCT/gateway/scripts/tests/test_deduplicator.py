"""
deduplicator.py 的单元测试
"""

import sys
import tempfile
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from gateway.config import GatewayConfig
from gateway.deduplicator import (
    ExpressionNormalizer,
    HashDeduplicator,
    SemanticDeduplicator,
    DuplicateRecord,
    normalize_expression,
    compute_expression_hash,
    compute_candidate_hash
)


def test_expression_normalizer():
    """测试表达式标准化器"""
    normalizer = ExpressionNormalizer()

    # 测试空格移除和大小写转换
    expr1 = "MA(close, 5)"
    expr2 = "ma(close , 5)"
    expr3 = "MA( close, 5 )"

    norm1 = normalizer.normalize(expr1)
    norm2 = normalizer.normalize(expr2)
    norm3 = normalizer.normalize(expr3)

    print(f"\n原表达式: '{expr1}'")
    print(f"标准化后: '{norm1}'")

    assert norm1 == "ma(close,5)"
    assert norm1 == norm2
    assert norm1 == norm3

    # 测试带配置的标准化
    normalized_with_config = normalizer.normalize_with_config(expr1, {"window": 5})
    print(f"带配置标准化: '{normalized_with_config}'")

    assert "ma(close,5)" in normalized_with_config
    assert "window" in normalized_with_config

    print("✅ 表达式标准化器测试通过")


def test_hash_deduplicator():
    """测试哈希去重器"""
    dedup = HashDeduplicator()

    # 计算哈希
    expr1 = "MA(close, 5)"
    expr2 = "MA(close, 10)"
    expr3 = "MA(close, 5)"  # 与 expr1 相同

    hash1 = dedup.compute_expr_hash(expr1)
    hash2 = dedup.compute_expr_hash(expr2)
    hash3 = dedup.compute_expr_hash(expr3)

    print(f"\n'{expr1}' 哈希: {hash1[:16]}...")
    print(f"'{expr2}' 哈希: {hash2[:16]}...")

    # 相同表达式应产生相同哈希
    assert hash1 == hash3
    # 不同表达式应产生不同哈希
    assert hash1 != hash2

    # 测试候选哈希（包含配置）
    candidate_hash1 = dedup.compute_candidate_hash(expr1, {"window": 5})
    candidate_hash2 = dedup.compute_candidate_hash(expr1, {"window": 10})
    candidate_hash3 = dedup.compute_candidate_hash(expr1, {"window": 5})

    print(f"候选哈希 (config window=5): {candidate_hash1[:16]}...")
    print(f"候选哈希 (config window=10): {candidate_hash2[:16]}...")

    assert candidate_hash1 != candidate_hash2
    assert candidate_hash1 == candidate_hash3

    # 测试添加和检查记录
    record = DuplicateRecord(
        candidate_id="test_001",
        expr_hash=hash1,
        candidate_hash=candidate_hash1,
        checked_at="2026-01-01T00:00:00Z",
        historical_report_id="report_001"
    )
    dedup.add_record(record)

    # 检查应该找到
    found = dedup.check_expr_hash(hash1)
    assert found is not None
    assert found.candidate_id == "test_001"

    found = dedup.check_candidate_hash(candidate_hash1)
    assert found is not None

    # 检查不存在的
    not_found = dedup.check_expr_hash("nonexistent")
    assert not_found is None

    print("✅ 哈希去重器测试通过")


def test_semantic_deduplicator():
    """测试语义去重器"""
    config = GatewayConfig()
    dedup = SemanticDeduplicator(config, use_vector_db=False)

    expr1 = "MA(close, 5)"
    expr2 = "MA(close, 5)"  # 相同表达式
    expr3 = "STD(close, 20)"  # 不同表达式

    config1 = {"window": 5}
    config2 = {"window": 5}  # 相同配置

    # 第一次检查，应该不重复
    result1 = dedup.is_duplicate(expr1, config1, "cand_001")
    print(f"\n第一次检查 '{expr1}': is_duplicate={result1.is_duplicate}")
    assert result1.is_duplicate is False
    assert result1.match_type == "none"

    # 注册因子
    dedup.register_factor("cand_001", expr1, config1, "report_001")

    # 第二次检查相同因子，应该重复
    result2 = dedup.is_duplicate(expr2, config2, "cand_002")
    print(f"第二次检查相同因子: is_duplicate={result2.is_duplicate}, match_type={result2.match_type}")
    assert result2.is_duplicate is True

    # 检查不同表达式，应该不重复
    result3 = dedup.is_duplicate(expr3, config1, "cand_003")
    print(f"检查不同表达式: is_duplicate={result3.is_duplicate}")
    assert result3.is_duplicate is False

    print("✅ 语义去重器测试通过")


def test_persist():
    """测试持久化功能"""
    dedup = HashDeduplicator()

    # 创建测试记录
    record = DuplicateRecord(
        candidate_id="test_persist",
        expr_hash="hash123",
        candidate_hash="cand_hash123",
        checked_at="2026-01-01T00:00:00Z",
        historical_report_id="report_persist"
    )
    dedup.add_record(record)
    assert dedup.size() == 1

    # 保存到临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = Path(f.name)

    dedup.save_to_persist(temp_path)

    # 创建新的去重器并加载
    new_dedup = HashDeduplicator()
    new_dedup.load_from_persist(temp_path)

    # 验证记录被加载
    found = new_dedup.check_expr_hash("hash123")
    assert found is not None
    assert found.candidate_id == "test_persist"

    # 清理临时文件
    temp_path.unlink()

    print("✅ 持久化测试通过")


def test_convenience_functions():
    """测试便捷函数"""
    expr = "MA(close, 5)"
    config = {"window": 5}

    # 测试标准化
    normalized = normalize_expression(expr)
    print(f"\nnormalize_expression('{expr}') = '{normalized}'")
    assert normalized == "ma(close,5)"

    # 测试表达式哈希
    expr_hash = compute_expression_hash(expr)
    print(f"compute_expression_hash() = {expr_hash[:16]}...")
    assert len(expr_hash) == 64

    # 测试候选哈希
    candidate_hash = compute_candidate_hash(expr, config)
    print(f"compute_candidate_hash() = {candidate_hash[:16]}...")
    assert len(candidate_hash) == 64

    print("✅ 便捷函数测试通过")


def test_cache_lru():
    """测试 LRU 缓存机制"""
    dedup = HashDeduplicator(cache_size=3)

    # 添加 4 条记录
    for i in range(4):
        record = DuplicateRecord(
            candidate_id=f"cand_{i}",
            expr_hash=f"hash_{i}",
            candidate_hash=f"cand_hash_{i}",
            checked_at="2026-01-01T00:00:00Z",
            historical_report_id=f"report_{i}"
        )
        dedup.add_record(record)

    # 缓存大小应为 3（最多 3 条）
    assert dedup.size() == 3

    # 最早添加的 hash_0 应该被移除
    assert dedup.check_expr_hash("hash_0") is None
    # 最新的 hash_3 应该存在
    assert dedup.check_expr_hash("hash_3") is not None

    print("✅ LRU 缓存测试通过")


if __name__ == "__main__":
    test_expression_normalizer()
    test_hash_deduplicator()
    test_semantic_deduplicator()
    test_persist()
    test_convenience_functions()
    test_cache_lru()

    print("\n" + "=" * 50)
    print("🎉 所有去重测试通过！")
    print("=" * 50)