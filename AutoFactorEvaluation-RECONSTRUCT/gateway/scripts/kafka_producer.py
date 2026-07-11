"""
Kafka 消息生产者模块

负责:
- 将通过网关（Pass）的候选因子发送到 Kafka
- 消息格式标准化（包含候选 ID、表达式、配置、网关结果）
- 异步/同步发送模式
- 错误处理和重试机制
- 消息发送确认和日志记录

使用场景:
- 下游资产化模块订阅 topic: factor_candidates_compute
- 实时接收通过网关的候选因子进行进一步处理
"""

import json
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

try:
    from kafka import KafkaProducer as KafkaProducerClient
    from kafka.errors import KafkaError, KafkaTimeoutError

    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False


    # 创建假的异常类用于类型提示
    class KafkaError(Exception):
        pass


    class KafkaTimeoutError(Exception):
        pass

from .config import GatewayConfig
from .models import Candidate, GatewayResult, GatewayLabel


class SendStatus(Enum):
    """发送状态"""
    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"
    SKIPPED = "skipped"


@dataclass
class SendResult:
    """发送结果"""
    status: SendStatus
    message: str
    topic: str
    partition: Optional[int] = None
    offset: Optional[int] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "message": self.message,
            "topic": self.topic,
            "partition": self.partition,
            "offset": self.offset,
            "timestamp": self.timestamp,
        }


class KafkaMessageBuilder:
    """
    Kafka 消息构建器

    构建符合下游消费规范的消息格式
    """

    MESSAGE_VERSION = "v1.0"

    @classmethod
    def build_pass_message(
            cls,
            candidate: Candidate,
            result: GatewayResult,
            factor_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        构建通过网关的因子消息

        Args:
            candidate: 候选因子对象
            result: 网关结果
            factor_id: 因子 ID（如果已生成）

        Returns:
            消息字典
        """
        message = {
            "schema_version": cls.MESSAGE_VERSION,
            "message_type": "factor_candidate_passed",
            "created_at": datetime.now(timezone.utc).isoformat(),

            # 候选信息
            "candidate": {
                "candidate_id": candidate.candidate_id,
                "expr": candidate.expr,
                "config": candidate.config,
                "basic_info": candidate.basic_info,
                "born_timestamp": candidate.born_timestamp,
            },

            # 网关信息
            "gateway": {
                "label": result.label.value,
                "run_id": result.run_id,
                "checked_at": result.checked_at,
                "complexity_score": result.complexity_score,
                "is_within_budget": result.is_within_budget,
            },

            # 可选字段
            "factor_id": factor_id,
        }

        return message

    @classmethod
    def build_temp_message(
            cls,
            candidate: Candidate,
            result: GatewayResult,
            historical_report_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        构建临时观察因子消息

        Args:
            candidate: 候选因子对象
            result: 网关结果
            historical_report_id: 历史报告 ID（如果是重复因子）

        Returns:
            消息字典
        """
        message = {
            "schema_version": cls.MESSAGE_VERSION,
            "message_type": "factor_candidate_temp",
            "created_at": datetime.now(timezone.utc).isoformat(),

            "candidate": {
                "candidate_id": candidate.candidate_id,
                "expr": candidate.expr,
                "config": candidate.config,
            },

            "gateway": {
                "label": result.label.value,
                "run_id": result.run_id,
                "checked_at": result.checked_at,
                "reason": result.reason,
                "is_duplicate": result.is_duplicate,
            },

            "historical_report_id": historical_report_id,
        }

        return message

    @classmethod
    def build_rejected_message(
            cls,
            candidate: Candidate,
            result: GatewayResult
    ) -> Dict[str, Any]:
        """
        构建被拒绝因子的消息（用于日志/监控）

        Args:
            candidate: 候选因子对象
            result: 网关结果

        Returns:
            消息字典
        """
        message = {
            "schema_version": cls.MESSAGE_VERSION,
            "message_type": "factor_candidate_rejected",
            "created_at": datetime.now(timezone.utc).isoformat(),

            "candidate": {
                "candidate_id": candidate.candidate_id,
                "expr": candidate.expr[:200],  # 限制长度
            },

            "gateway": {
                "label": result.label.value,
                "run_id": result.run_id,
                "checked_at": result.checked_at,
                "reason": result.reason,
                "error_message": result.error_message,
            },
        }

        return message


class KafkaProducer:
    """
    Kafka 消息生产者

    负责将网关结果发送到 Kafka 供下游消费
    """

    # 默认配置
    DEFAULT_RETRY_COUNT = 3
    DEFAULT_RETRY_BACKOFF_MS = 100
    DEFAULT_REQUEST_TIMEOUT_MS = 5000
    DEFAULT_MAX_BLOCK_MS = 10000

    def __init__(
            self,
            config: GatewayConfig,
            topic: Optional[str] = None,
            bootstrap_servers: Optional[str] = None,
            retry_count: int = DEFAULT_RETRY_COUNT,
            enable_async: bool = True
    ):
        """
        初始化 Kafka 生产者

        Args:
            config: 网关配置对象
            topic: Kafka topic（可选，默认使用配置中的）
            bootstrap_servers: Kafka 服务器地址（可选，默认使用配置中的）
            retry_count: 重试次数
            enable_async: 是否使用异步发送（默认 True）
        """
        self.config = config
        self.logger = logging.getLogger("gateway.kafka")

        # 获取 Kafka 配置
        self.bootstrap_servers = bootstrap_servers or config.kafka.bootstrap_servers
        self.topic = topic or config.kafka.topic
        self.enabled = config.kafka.enabled
        self.retry_count = retry_count
        self.enable_async = enable_async

        # 生产者实例
        self._producer = None
        self._initialized = False

        # 统计信息
        self.stats = {
            "sent_success": 0,
            "sent_failed": 0,
            "sent_skipped": 0,
        }

        # 回调函数列表
        self._callbacks: List[Callable[[SendResult], None]] = []

        # 如果启用 Kafka，初始化生产者
        if self.enabled:
            self._init_producer()

    def _init_producer(self) -> None:
        """初始化 Kafka 生产者"""
        if not KAFKA_AVAILABLE:
            self.logger.warning(
                "kafka-python not installed. Kafka producer disabled. "
                "Install with: pip install kafka-python"
            )
            self.enabled = False
            return

        try:
            self._producer = KafkaProducerClient(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode('utf-8'),
                retries=self.retry_count,
                retry_backoff_ms=self.DEFAULT_RETRY_BACKOFF_MS,
                request_timeout_ms=self.DEFAULT_REQUEST_TIMEOUT_MS,
                max_block_ms=self.DEFAULT_MAX_BLOCK_MS,
                acks='all',  # 等待所有副本确认
                compression_type='gzip',  # 压缩消息
            )
            self._initialized = True
            self.logger.info(
                f"Kafka producer initialized: {self.bootstrap_servers}, topic: {self.topic}"
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize Kafka producer: {e}")
            self.enabled = False
            raise

    def send_pass(
            self,
            candidate: Candidate,
            result: GatewayResult,
            factor_id: Optional[str] = None
    ) -> SendResult:
        """
        发送通过网关的因子消息

        Args:
            candidate: 候选因子对象
            result: 网关结果
            factor_id: 因子 ID（如果已生成）

        Returns:
            SendResult 对象
        """
        if not self.enabled:
            return self._skip_result("Kafka producer disabled")

        if not self._initialized:
            return self._fail_result("Kafka producer not initialized")

        message = KafkaMessageBuilder.build_pass_message(candidate, result, factor_id)
        return self._send(message, candidate.candidate_id)

    def send_temp(
            self,
            candidate: Candidate,
            result: GatewayResult,
            historical_report_id: Optional[str] = None
    ) -> SendResult:
        """
        发送临时观察因子消息

        Args:
            candidate: 候选因子对象
            result: 网关结果
            historical_report_id: 历史报告 ID

        Returns:
            SendResult 对象
        """
        if not self.enabled:
            return self._skip_result("Kafka producer disabled")

        if not self._initialized:
            return self._fail_result("Kafka producer not initialized")

        message = KafkaMessageBuilder.build_temp_message(candidate, result, historical_report_id)
        return self._send(message, candidate.candidate_id)

    def send_rejected(
            self,
            candidate: Candidate,
            result: GatewayResult
    ) -> SendResult:
        """
        发送被拒绝因子消息（用于监控）

        Args:
            candidate: 候选因子对象
            result: 网关结果

        Returns:
            SendResult 对象
        """
        if not self.enabled:
            return self._skip_result("Kafka producer disabled")

        if not self._initialized:
            return self._fail_result("Kafka producer not initialized")

        message = KafkaMessageBuilder.build_rejected_message(candidate, result)
        return self._send(message, candidate.candidate_id)

    def _send(self, message: Dict[str, Any], key: str) -> SendResult:
        """
        发送消息到 Kafka

        Args:
            message: 消息字典
            key: 消息 key（用于分区）

        Returns:
            SendResult 对象
        """
        try:
            key_bytes = key.encode('utf-8') if key else None

            if self.enable_async:
                # 异步发送
                future = self._producer.send(self.topic, key=key_bytes, value=message)

                # 添加回调
                future.add_callback(
                    lambda metadata: self._on_send_success(metadata, key, message)
                )
                future.add_errback(
                    lambda error: self._on_send_error(error, key, message)
                )

                # 立即返回 pending 状态
                self.stats["sent_success"] += 1  # 暂计，实际成功在回调中确认
                return SendResult(
                    status=SendStatus.PENDING,
                    message=f"Async send initiated for {key}",
                    topic=self.topic,
                )
            else:
                # 同步发送
                future = self._producer.send(self.topic, key=key_bytes, value=message)
                metadata = future.get(timeout=10)

                self.stats["sent_success"] += 1
                return SendResult(
                    status=SendStatus.SUCCESS,
                    message="Message sent successfully",
                    topic=metadata.topic,
                    partition=metadata.partition,
                    offset=metadata.offset,
                )

        except KafkaTimeoutError as e:
            self.stats["sent_failed"] += 1
            self.logger.error(f"Kafka timeout for {key}: {e}")
            return self._fail_result(f"Timeout: {e}")

        except KafkaError as e:
            self.stats["sent_failed"] += 1
            self.logger.error(f"Kafka error for {key}: {e}")
            return self._fail_result(f"Kafka error: {e}")

        except Exception as e:
            self.stats["sent_failed"] += 1
            self.logger.error(f"Unexpected error for {key}: {e}")
            return self._fail_result(f"Unexpected error: {e}")

    def _on_send_success(self, metadata, key: str, message: Dict[str, Any]) -> None:
        """异步发送成功回调"""
        self.logger.debug(
            f"Kafka send success: key={key}, topic={metadata.topic}, "
            f"partition={metadata.partition}, offset={metadata.offset}"
        )

        result = SendResult(
            status=SendStatus.SUCCESS,
            message="Message sent successfully",
            topic=metadata.topic,
            partition=metadata.partition,
            offset=metadata.offset,
        )
        self._notify_callbacks(result)

    def _on_send_error(self, error, key: str, message: Dict[str, Any]) -> None:
        """异步发送失败回调"""
        self.stats["sent_failed"] += 1
        # 回调前已统计为成功，需要修正
        self.stats["sent_success"] = max(0, self.stats["sent_success"] - 1)

        self.logger.error(f"Kafka send error for {key}: {error}")

        result = self._fail_result(f"Async send error: {error}")
        self._notify_callbacks(result)

    def _skip_result(self, reason: str) -> SendResult:
        """生成跳过结果"""
        self.stats["sent_skipped"] += 1
        return SendResult(
            status=SendStatus.SKIPPED,
            message=f"Message not sent: {reason}",
            topic=self.topic,
        )

    def _fail_result(self, reason: str) -> SendResult:
        """生成失败结果"""
        return SendResult(
            status=SendStatus.FAILED,
            message=reason,
            topic=self.topic,
        )

    def add_callback(self, callback: Callable[[SendResult], None]) -> None:
        """
        添加发送结果回调函数

        Args:
            callback: 回调函数，接收 SendResult 参数
        """
        self._callbacks.append(callback)

    def _notify_callbacks(self, result: SendResult) -> None:
        """通知所有回调"""
        for callback in self._callbacks:
            try:
                callback(result)
            except Exception as e:
                self.logger.error(f"Callback error: {e}")

    def flush(self, timeout: float = 10.0) -> None:
        """
        等待所有待发送消息完成

        Args:
            timeout: 超时时间（秒）
        """
        if self._producer and self._initialized:
            self._producer.flush(timeout=timeout)
            self.logger.info("Kafka producer flushed")

    def close(self) -> None:
        """关闭 Kafka 生产者"""
        if self._producer and self._initialized:
            self.flush()
            self._producer.close()
            self._initialized = False
            self.logger.info("Kafka producer closed")

    def get_stats(self) -> Dict[str, int]:
        """获取发送统计信息"""
        return self.stats.copy()

    def reset_stats(self) -> None:
        """重置统计信息"""
        self.stats = {
            "sent_success": 0,
            "sent_failed": 0,
            "sent_skipped": 0,
        }

    @property
    def is_available(self) -> bool:
        """检查 Kafka 是否可用"""
        return self.enabled and self._initialized and KAFKA_AVAILABLE


# 便捷函数
def create_producer(
        config: Optional[GatewayConfig] = None,
        topic: Optional[str] = None,
        bootstrap_servers: Optional[str] = None
) -> KafkaProducer:
    """
    创建 Kafka 生产者实例

    Args:
        config: 网关配置对象
        topic: Kafka topic
        bootstrap_servers: Kafka 服务器地址

    Returns:
        KafkaProducer 实例
    """
    if config is None:
        config = GatewayConfig()

    return KafkaProducer(config, topic, bootstrap_servers)


def send_factor_message(
        candidate: Candidate,
        result: GatewayResult,
        config: Optional[GatewayConfig] = None,
        factor_id: Optional[str] = None
) -> SendResult:
    """
    快速发送因子消息（仅发送通过的消息）

    Args:
        candidate: 候选因子对象
        result: 网关结果
        config: 网关配置对象
        factor_id: 因子 ID

    Returns:
        SendResult 对象
    """
    if result.label != GatewayLabel.PASS:
        return SendResult(
            status=SendStatus.SKIPPED,
            message=f"Not sending non-pass message (label: {result.label.value})",
            topic="",
        )

    if config is None:
        config = GatewayConfig()

    producer = KafkaProducer(config)
    try:
        return producer.send_pass(candidate, result, factor_id)
    finally:
        producer.close()