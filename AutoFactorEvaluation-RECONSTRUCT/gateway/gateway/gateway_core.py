"""
网关核心路由引擎

负责:
- 整合所有子模块（Validator、FutureScanner、ComplexityEvaluator、Deduplicator、DataQualityChecker）
- 按顺序执行检查：Schema → 算子白名单 → 未来函数 → 复杂度 → 去重 → 数据质量
- 生成路由决策：Pass / Duplicated / Rejected
- 返回 GatewayResult 供外部使用

路由规则:
- 所有检查通过 → Pass
- 语义重复 → Duplicated（引用历史报告）
- 任何关键检查失败（非法算子、未来函数、超预算）→ Rejected
- 数据质量不通过 → 仍标记 Pass 但附带 DataQuality 警告信息
"""

import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

from .models import (
    Candidate,
    GatewayResult,
    GatewayLabel,
    GatewaySegment,
)
from .config import GatewayConfig
from .validator import StaticValidator
from .future_scanner import FutureFunctionScanner
from .complexity import ComplexityEvaluator
from .deduplicator import SemanticDeduplicator, DeduplicationResult
from .exceptions import (
    GatewayError,
    SchemaValidationError,
    OperatorNotAllowedError,
    FutureFunctionError,
    ComplexityError
)
from .data_quality import DataQualityRunner, QualityReport

logger = logging.getLogger("gateway.gateway_core")


class GatewayCore:
    """
    网关核心路由引擎

    使用示例:
        config = GatewayConfig()
        gateway = GatewayCore(config)
        result = gateway.process(candidate_dict)
    """

    def __init__(
        self,
        config: GatewayConfig,
        use_ast_parser: bool = False,
        market_data_path: Optional[str] = None,
    ):
        """
        初始化网关核心引擎

        Args:
            config: 网关配置对象
            use_ast_parser: 是否使用 AST 解析器（更精确的校验）
            market_data_path: 可选的市场数据路径，用于数据质量检测。
                提供时自动启用数据质量检测；为 None 时跳过。
        """
        self.config = config
        self.use_ast_parser = use_ast_parser

        # 初始化各子模块
        self.validator = StaticValidator(config, use_ast_parser)
        self.future_scanner = FutureFunctionScanner(config, use_ast_parser)
        self.complexity_evaluator = ComplexityEvaluator(config, use_ast_parser)
        self.deduplicator = SemanticDeduplicator(config, use_vector_db=False)

        # 数据质量检测（可选）
        self.market_data_path = market_data_path
        self.dq_runner: Optional[DataQualityRunner] = None
        if market_data_path is not None:
            self.dq_runner = DataQualityRunner()
            logger.info("Data quality checker enabled, market_data_path=%s", market_data_path)

        # 统计信息
        self.stats = {
            "total_processed": 0,
            "pass_count": 0,
            "duplicated_count": 0,
            "rejected_count": 0,
        }

    def process(self, candidate_dict: Dict[str, Any]) -> GatewayResult:
        """
        处理单个候选因子，返回网关结果

        Args:
            candidate_dict: 候选因子的原始字典（从 candidate.json 读取）

        Returns:
            GatewayResult 对象
        """
        # 生成运行 ID 和时间戳
        run_id = self._generate_run_id()
        checked_at = datetime.now(timezone.utc).isoformat()

        # 更新统计
        self.stats["total_processed"] += 1

        # 1. 解析为 Candidate 对象
        try:
            candidate = Candidate.from_dict(candidate_dict)
        except Exception as e:
            self.stats["rejected_count"] += 1
            return self._create_rejected_result(
                run_id, checked_at,
                f"Failed to parse candidate: {str(e)}",
                error_message=str(e)
            )

        # 2. 执行所有校验
        validation_passed, validation_error, validation_warnings = self.validator.validate_all(candidate_dict)
        if not validation_passed:
            self.stats["rejected_count"] += 1
            return self._create_rejected_result(
                run_id, checked_at,
                f"Validation failed: {validation_error}",
                is_legal=False,
                error_message=validation_error
            )

        # 3. 检查未来函数
        has_future, future_detail = self.future_scanner.scan(candidate.expr)
        if has_future:
            self.stats["rejected_count"] += 1
            return self._create_rejected_result(
                run_id, checked_at,
                f"Future function detected: {future_detail}",
                has_future=True,
                error_message=future_detail
            )

        # 4. 评估复杂度
        complexity_score, is_within_budget, complexity_report = self.complexity_evaluator.evaluate(candidate.expr)
        if not is_within_budget:
            self.stats["rejected_count"] += 1
            return self._create_rejected_result(
                run_id, checked_at,
                f"Complexity budget exceeded: {complexity_score:.2f} > {self.config.max_complexity}",
                complexity_score=complexity_score,
                is_within_budget=False,
                error_message=f"Complexity score {complexity_score:.2f} exceeds threshold {self.config.max_complexity}"
            )

        # 5. 检查重复
        dedup_result = self.deduplicator.is_duplicate(
            candidate.expr,
            candidate.config,
            candidate.candidate_id
        )

        if dedup_result.is_duplicate:
            # 重复因子 -> Duplicated（引用历史报告）
            self.stats["duplicated_count"] += 1
            return self._create_duplicated_result(
                run_id, checked_at,
                f"Duplicate factor detected (match type: {dedup_result.match_type})",
                is_duplicate=True,
                historical_report_id=dedup_result.historical_report_id,
                similarity_score=dedup_result.similarity_score
            )

        # 6. 数据质量检测（可选）
        dq_label: Optional[str] = None
        dq_score: Optional[float] = None
        dq_reason: Optional[str] = None
        if self.dq_runner is not None and self.market_data_path is not None:
            try:
                dq_result = self.dq_runner.check(self.market_data_path)
                dq_label = dq_result.label
                dq_score = dq_result.overall_score
                dq_reason = dq_result.reason
                if dq_label != "Pass":
                    logger.warning(
                        "Data quality warning for candidate %s: label=%s score=%.3f reason=%s",
                        candidate.candidate_id, dq_label, dq_score, dq_reason,
                    )
            except Exception as e:
                logger.warning("Data quality check failed (non-fatal): %s", e)
                dq_label = "Warning"
                dq_reason = f"Data quality check error: {e}"

        # 7. 所有检查通过 -> Pass
        self.stats["pass_count"] += 1

        # 注册因子到去重缓存（供后续使用）
        duplicated_report_id = f"pending_{candidate.candidate_id}"
        self.deduplicator.register_factor(
            candidate.candidate_id,
            candidate.expr,
            candidate.config,
            duplicated_report_id
        )

        return self._create_pass_result(
            run_id, checked_at,
            is_legal=True,
            has_future=False,
            complexity_score=complexity_score,
            is_within_budget=True,
            is_duplicate=False,
            data_quality_label=dq_label,
            data_quality_score=dq_score,
            data_quality_reason=dq_reason,
        )

    def _create_pass_result(
            self,
            run_id: str,
            checked_at: str,
            is_legal: bool = True,
            has_future: bool = False,
            complexity_score: float = 0.0,
            is_within_budget: bool = True,
            is_duplicate: bool = False,
            data_quality_label: Optional[str] = None,
            data_quality_score: Optional[float] = None,
            data_quality_reason: Optional[str] = None,
    ) -> GatewayResult:
        """创建 Pass 结果"""
        return GatewayResult(
            label=GatewayLabel.PASS,
            reason="",
            run_id=run_id,
            checked_at=checked_at,
            is_legal=is_legal,
            has_future=has_future,
            complexity_score=complexity_score,
            is_within_budget=is_within_budget,
            is_duplicate=is_duplicate,
            data_quality_label=data_quality_label,
            data_quality_score=data_quality_score,
            data_quality_reason=data_quality_reason,
        )

    def _create_duplicated_result(
            self,
            run_id: str,
            checked_at: str,
            reason: str,
            is_duplicate: bool = True,
            historical_report_id: Optional[str] = None,
            similarity_score: float = 1.0
    ) -> GatewayResult:
        """创建 duplicated 结果"""
        return GatewayResult(
            label=GatewayLabel.DUPLICATED,
            reason=reason,
            run_id=run_id,
            checked_at=checked_at,
            is_legal=True,
            has_future=False,
            is_within_budget=True,
            is_duplicate=is_duplicate,
            historical_report_id=historical_report_id
        )

    def _create_rejected_result(
            self,
            run_id: str,
            checked_at: str,
            reason: str,
            is_legal: bool = False,
            has_future: bool = False,
            complexity_score: float = 0.0,
            is_within_budget: bool = False,
            is_duplicate: bool = False,
            error_message: str = ""
    ) -> GatewayResult:
        """创建 Rejected 结果"""
        return GatewayResult(
            label=GatewayLabel.REJECTED,
            reason=reason,
            run_id=run_id,
            checked_at=checked_at,
            is_legal=is_legal,
            has_future=has_future,
            complexity_score=complexity_score,
            is_within_budget=is_within_budget,
            is_duplicate=is_duplicate,
            error_message=error_message or reason
        )

    def _generate_run_id(self) -> str:
        """生成唯一的运行 ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = uuid.uuid4().hex[:8]
        return f"gateway_{timestamp}_{random_suffix}"

    def process_batch(self, candidate_dicts: List[Dict[str, Any]]) -> List[GatewayResult]:
        """
        批量处理多个候选因子

        Args:
            candidate_dicts: 候选因子字典列表

        Returns:
            GatewayResult 对象列表
        """
        results = []
        for candidate_dict in candidate_dicts:
            try:
                result = self.process(candidate_dict)
                results.append(result)
            except Exception as e:
                # 单个失败不影响其他
                results.append(self._create_rejected_result(
                    self._generate_run_id(),
                    datetime.now(timezone.utc).isoformat(),
                    f"Unexpected error: {str(e)}",
                    error_message=str(e)
                ))
        return results

    def get_stats(self) -> Dict[str, int]:
        """获取处理统计信息"""
        return self.stats.copy()

    def reset_stats(self) -> None:
        """重置统计信息"""
        self.stats = {
            "total_processed": 0,
            "pass_count": 0,
            "duplicated_count": 0,
            "rejected_count": 0,
        }

    def load_history(self, history_path: str) -> None:
        """
        加载历史去重记录

        Args:
            history_path: 历史记录文件路径
        """
        from pathlib import Path
        self.deduplicator.load_history(Path(history_path))

    def save_history(self, history_path: str) -> None:
        """
        保存去重记录到文件

        Args:
            history_path: 历史记录文件路径
        """
        from pathlib import Path
        self.deduplicator.save_history(Path(history_path))


# 便捷函数
def quick_process(candidate_dict: Dict[str, Any], config: Optional[GatewayConfig] = None) -> GatewayResult:
    """
    快速处理单个候选因子

    Args:
        candidate_dict: 候选因子字典
        config: 配置对象（可选）

    Returns:
        GatewayResult 对象
    """
    if config is None:
        config = GatewayConfig()

    gateway = GatewayCore(config)
    return gateway.process(candidate_dict)


def check_candidate(candidate_dict: Dict[str, Any]) -> Tuple[bool, str]:
    """
    快速检查候选因子是否可以通过网关

    Args:
        candidate_dict: 候选因子字典

    Returns:
        (can_pass, reason) 元组
    """
    config = GatewayConfig()
    gateway = GatewayCore(config)
    result = gateway.process(candidate_dict)

    can_pass = result.label == GatewayLabel.PASS
    reason = result.reason if not can_pass else ""

    return can_pass, reason