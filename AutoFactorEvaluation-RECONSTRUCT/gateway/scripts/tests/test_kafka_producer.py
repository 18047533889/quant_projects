"""
kafka_producer.py 的单元测试
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from gateway.config import GatewayConfig
from gateway.models import Candidate, GatewayResult, GatewayLabel
from gateway.kafka_producer import (
    KafkaProducer,
    KafkaMessageBuilder,
    SendStatus,
    create_producer,
    send_factor_message
)
from datetime import datetime, timezone


def create_test_candidate() -> Candidate:
    """创建测试用的 Candidate 对象"""
    return Candidate(
        schema_version="disk.v1",
        candidate_id="test_cand_001",
        expr="MA(close, 5)",
        config={"window": 5},
        born_timestamp=datetime.now(timezone.utc).isoformat(),
        basic_info={"source": "test"}
    )


def create_test_result() -> GatewayResult:
    """创建测试用的 GatewayResult 对象"""
    return GatewayResult(
        label=GatewayLabel.PASS,
        reason="",
        run_id="gateway_test_001",
        checked_at=datetime.now(timezone.utc).isoformat(),
        complexity_score=1.0,
        is_within_budget=True,
        is_legal=True,
        has_future=False,
        is_duplicate=False
    )


def test_message_builder():
    """测试消息构建器"""
    candidate = create_test_candidate()
    result = create_test_result()

    # 测试 Pass 消息
    pass_msg = KafkaMessageBuilder.build_pass_message(candidate, result)
    print(f"\nPass 消息结构: {list(pass_msg.keys())}")
    assert pass_msg["message_type"] == "factor_candidate_passed"
    assert pass_msg["candidate"]["candidate_id"] == "test_cand_001"
    assert pass_msg["candidate"]["expr"] == "MA(close, 5)"
    assert pass_msg["gateway"]["label"] == "Pass"

    # 测试 Temp 消息
    temp_msg = KafkaMessageBuilder.build_temp_message(candidate, result, "hist_001")
    print(f"Temp 消息结构: {list(temp_msg.keys())}")
    assert temp_msg["message_type"] == "factor_candidate_temp"
    assert temp_msg["historical_report_id"] == "hist_001"

    # 测试 Rejected 消息
    rejected_msg = KafkaMessageBuilder.build_rejected_message(candidate, result)
    print(f"Rejected 消息结构: {list(rejected_msg.keys())}")
    assert rejected_msg["message_type"] == "factor_candidate_rejected"

    print("✅ 消息构建器测试通过")


def test_kafka_producer_disabled():
    """测试 Kafka 禁用时的行为"""
    config = GatewayConfig()
    config.kafka.enabled = False

    producer = KafkaProducer(config)
    candidate = create_test_candidate()
    result = create_test_result()

    send_result = producer.send_pass(candidate, result)

    print(f"\n禁用状态发送结果: {send_result.status.value}")
    assert send_result.status == SendStatus.SKIPPED
    assert "disabled" in send_result.message.lower()

    producer.close()
    print("✅ 禁用状态测试通过")


@patch('gateway.kafka_producer.KafkaProducerClient')
def test_kafka_producer_sync_send(mock_kafka_client):
    """测试 Kafka 同步发送（Mock）"""
    config = GatewayConfig()
    config.kafka.enabled = True
    config.kafka.bootstrap_servers = "localhost:9092"

    # Mock 生产者
    mock_producer = MagicMock()
    mock_future = MagicMock()
    mock_future.get.return_value = MagicMock(
        topic="test_topic",
        partition=0,
        offset=123
    )
    mock_producer.send.return_value = mock_future
    mock_kafka_client.return_value = mock_producer

    producer = KafkaProducer(config, enable_async=False)
    producer._initialized = True
    producer._producer = mock_producer

    candidate = create_test_candidate()
    result = create_test_result()

    send_result = producer.send_pass(candidate, result)

    print(f"\n同步发送结果: {send_result.status.value}")
    print(f"  partition: {send_result.partition}")
    print(f"  offset: {send_result.offset}")

    assert send_result.status == SendStatus.SUCCESS
    assert send_result.partition == 0
    assert send_result.offset == 123

    producer.close()
    print("✅ 同步发送测试通过")


@patch('gateway.kafka_producer.KafkaProducerClient')
def test_kafka_producer_async_send(mock_kafka_client):
    """测试 Kafka 异步发送（Mock）"""
    config = GatewayConfig()
    config.kafka.enabled = True
    config.kafka.bootstrap_servers = "localhost:9092"

    # Mock 生产者
    mock_producer = MagicMock()
    mock_future = MagicMock()
    mock_producer.send.return_value = mock_future
    mock_kafka_client.return_value = mock_producer

    producer = KafkaProducer(config, enable_async=True)
    producer._initialized = True
    producer._producer = mock_producer

    candidate = create_test_candidate()
    result = create_test_result()

    send_result = producer.send_pass(candidate, result)

    print(f"\n异步发送结果: {send_result.status.value}")
    print(f"  message: {send_result.message}")

    assert send_result.status == SendStatus.PENDING
    assert "initiated" in send_result.message.lower()

    producer.close()
    print("✅ 异步发送测试通过")


def test_stats():
    """测试统计功能"""
    config = GatewayConfig()
    config.kafka.enabled = False  # 禁用避免真实连接

    producer = KafkaProducer(config)

    # 初始统计
    stats = producer.get_stats()
    print(f"\n初始统计: {stats}")

    # 发送几条消息
    candidate = create_test_candidate()
    result = create_test_result()

    for _ in range(5):
        producer.send_pass(candidate, result)

    stats = producer.get_stats()
    print(f"发送后统计: {stats}")
    assert stats["sent_skipped"] >= 5

    # 重置统计
    producer.reset_stats()
    stats = producer.get_stats()
    print(f"重置后统计: {stats}")
    assert stats["sent_success"] == 0

    print("✅ 统计功能测试通过")


def test_convenience_functions():
    """测试便捷函数"""
    candidate = create_test_candidate()
    result = create_test_result()
    config = GatewayConfig()
    config.kafka.enabled = False  # 禁用避免真实连接

    # 测试 create_producer
    producer = create_producer(config)
    assert isinstance(producer, KafkaProducer)

    # 测试 send_factor_message（只发送 Pass 消息）
    send_result = send_factor_message(candidate, result, config)
    print(f"\nsend_factor_message 结果: {send_result.status.value}")
    # 由于 Kafka 禁用，应该跳过
    assert send_result.status == SendStatus.SKIPPED

    producer.close()
    print("✅ 便捷函数测试通过")


if __name__ == "__main__":
    test_message_builder()
    test_kafka_producer_disabled()
    test_kafka_producer_sync_send()
    test_kafka_producer_async_send()
    test_stats()
    test_convenience_functions()

    print("\n" + "=" * 50)
    print("🎉 所有 Kafka 生产者测试通过！")
    print("=" * 50)