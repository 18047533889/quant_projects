"""
磁盘 I/O 工具模块 - 严格按照 Disk Contracts 规范

负责:
- 从 CandidatePool 读取 candidate.json
- 将网关结果写入对应的输出库（GatewayPassBase / DuplicatedBase / AntiSampleBase）
- 生成并写入 manifest.json
- 写入缓存报告到 GatewayReportCache
- 所有操作遵循 Fail-fast 原则
"""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from uuid import uuid4

from .models import (
    Candidate,
    GatewayResult,
    GatewayLabel,
    Manifest,
    ValidationStatus,
    GatewaySegment
)
from .config import GatewayConfig, ConfigError


class IOManager:
    """
    磁盘 I/O 管理器

    负责所有与磁盘静态库的交互，符合 Disk Contracts 规范
    """

    def __init__(self, config: GatewayConfig):
        """
        初始化 I/O 管理器

        Args:
            config: 网关配置对象
        """
        self.config = config

        # 确保输出目录存在
        self._ensure_output_dirs()

    def _ensure_output_dirs(self) -> None:
        """确保所有输出目录存在"""
        dirs = [
            self.config.paths.gateway_pass_base,
            self.config.paths.duplicated_base,
            self.config.paths.anti_sample_base,
            self.config.paths.report_cache,
        ]

        for dir_path in dirs:
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)  # kept for runtime per-factor subdirs

    def read_candidate(self, candidate_id: str) -> Tuple[Dict[str, Any], Candidate]:
        """
        从 CandidatePool 读取候选因子 JSON

        符合 Disk Contracts:
        - 路径: {candidate_pool}/{candidate_id}/candidate.json
        - Fail fast: 文件缺失或格式错误立即抛出异常

        Args:
            candidate_id: 候选因子 ID

        Returns:
            (原始字典, Candidate 对象)

        Raises:
            FileNotFoundError: candidate.json 不存在
            json.JSONDecodeError: JSON 格式错误
            ConfigError: schema 验证失败
        """
        pool_dir = Path(self.config.paths.candidate_pool) / candidate_id
        candidate_file = pool_dir / "candidate.json"

        if not candidate_file.exists():
            raise FileNotFoundError(
                f"Candidate file not found: {candidate_file}\n"
                f"Expected path: {pool_dir}/candidate.json"
            )

        try:
            with open(candidate_file, "r", encoding="utf-8") as f:
                candidate_dict = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in candidate file: {e}",
                e.doc,
                e.pos
            )

        # 验证 schema_version
        schema_version = candidate_dict.get("schema_version")
        if schema_version != "disk.v1":
            raise ConfigError(
                f"Unsupported schema_version: {schema_version}. "
                f"Expected: disk.v1"
            )

        # 验证 candidate_id 匹配
        file_candidate_id = candidate_dict.get("candidate_id")
        if file_candidate_id != candidate_id:
            raise ConfigError(
                f"candidate_id mismatch: "
                f"file has '{file_candidate_id}', "
                f"expected '{candidate_id}'"
            )

        # 转换为 Candidate 对象
        candidate = Candidate.from_dict(candidate_dict)

        return candidate_dict, candidate

    def write_output(
            self,
            candidate_id: str,
            candidate_dict: Dict[str, Any],
            result: GatewayResult
    ) -> Path:
        """
        根据网关结果写入对应的输出库

        符合 Disk Contracts:
        - Pass → {gateway_pass_base}/{candidate_id}/
        - Duplicated → {duplicated_base}/{candidate_id}/
        - Rejected → {anti_sample_base}/gateway/{candidate_id}/

        每个输出目录包含:
        - candidate.json (已追加 Gateway 段落)
        - manifest.json

        Args:
            candidate_id: 候选因子 ID
            candidate_dict: 已更新 Gateway 段落的原始字典
            result: 网关结果

        Returns:
            输出目录路径
        """
        # 确定输出根目录
        if result.label == GatewayLabel.PASS:
            root_dir = self.config.paths.gateway_pass_base
        elif result.label == GatewayLabel.DUPLICATED:
            root_dir = self.config.paths.duplicated_base
        else:  # REJECTED
            root_dir = Path(self.config.paths.anti_sample_base) / "gateway"

        output_dir = Path(root_dir) / candidate_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # 写入 candidate.json
        candidate_file = output_dir / "candidate.json"
        with open(candidate_file, "w", encoding="utf-8") as f:
            json.dump(candidate_dict, f, indent=2, ensure_ascii=False)

        # 生成并写入 manifest.json
        validation_status = (
            ValidationStatus.PASSED
            if result.label != GatewayLabel.REJECTED
            else ValidationStatus.FAILED
        )

        manifest = Manifest.for_gateway_output(
            candidate_id=candidate_id,
            library=output_dir.parent.name + "/" + output_dir.name,
            run_id=result.run_id,
            validation_status=validation_status,
            errors=[result.reason] if result.reason else []
        )

        manifest_file = output_dir / "manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2, ensure_ascii=False)

        return output_dir

    def write_cache_report(
            self,
            candidate_hash: str,
            expr_hash: str,
            result: GatewayResult,
            report_md: str
    ) -> Path:
        """
        写入缓存报告到 GatewayReportCache

        符合 Disk Contracts:
        - 路径: {report_cache}/{candidate_hash}/
        - 包含: gateway_report.md, manifest.json

        Args:
            candidate_hash: 候选因子的哈希值（用作目录名）
            expr_hash: 表达式的哈希值
            result: 网关结果
            report_md: Markdown 格式的报告内容

        Returns:
            缓存目录路径
        """
        cache_dir = Path(self.config.paths.report_cache) / candidate_hash
        cache_dir.mkdir(parents=True, exist_ok=True)

        # 写入 gateway_report.md
        report_file = cache_dir / "gateway_report.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_md)

        # 生成并写入 manifest.json
        manifest = Manifest.for_cache_report(
            candidate_hash=candidate_hash,
            expr_hash=expr_hash,
            run_id=result.run_id,
            cache_key_version="v1"
        )

        manifest_file = cache_dir / "manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2, ensure_ascii=False)

        return cache_dir

    def write_rejected_log(
            self,
            candidate_id: str,
            candidate_dict: Dict[str, Any],
            result: GatewayResult
    ) -> Path:
        """
        写入拒绝日志到反样本库（额外的审计日志）

        这是 SOP 中要求的 Tier4 日志记录

        Args:
            candidate_id: 候选因子 ID
            candidate_dict: 候选因子字典
            result: 网关结果

        Returns:
            日志文件路径
        """
        # 反样本库根目录
        anti_sample_root = Path(self.config.paths.anti_sample_base)

        # 创建 tier4 日志目录
        tier4_dir = anti_sample_root / "gateway" / "tier4_logs"
        tier4_dir.mkdir(parents=True, exist_ok=True)

        # 日志文件名包含时间戳
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        log_file = tier4_dir / f"{timestamp}_{candidate_id}.json"

        # 构建日志内容
        log_entry = {
            "candidate_id": candidate_id,
            "run_id": result.run_id,
            "checked_at": result.checked_at,
            "label": result.label.value,
            "reason": result.reason,
            "candidate": candidate_dict,
            "diagnostics": {
                "is_legal": result.is_legal,
                "has_future": result.has_future,
                "complexity_score": result.complexity_score,
                "is_within_budget": result.is_within_budget,
                "is_duplicate": result.is_duplicate,
                "error_message": result.error_message,
            }
        }

        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False)

        return log_file

    def write_afv(
        self,
        factor_dir: Path,
        result: GatewayResult,
        manifest: dict,
    ) -> Path:
        """在因子目录下写入 afv.json（Alpha Factor Verification）。

        afv.json 替代旧的 candidate.json，包含网关审查标记。

        Args:
            factor_dir: 待写入的因子目录。
            result: 网关审查结果。
            manifest: 因子 manifest 元信息。

        Returns:
            afv.json 路径。
        """
        checks = {
            "schema_valid": result.is_legal,
            "operator_allowed": result.is_legal,
            "has_future": result.has_future,
            "complexity_score": result.complexity_score,
            "is_within_budget": result.is_within_budget,
            "is_duplicate": result.is_duplicate,
        }
        if result.data_quality_label is not None:
            checks["data_quality"] = {
                "label": result.data_quality_label,
                "score": result.data_quality_score,
                "reason": result.data_quality_reason,
            }

        afv = {
            "schema_version": "disk.v1",
            "candidate_id": manifest.get("candidate_id", factor_dir.name),
            "campaign_id": manifest.get("campaign_id", ""),
            "formula": manifest.get("formula", ""),
            "gateway_version": "1.0.0",
            "Gateway": {
                "Label": result.label.value,
                "Reason": result.reason,
                "run_id": result.run_id,
                "checked_at": result.checked_at,
                "recommendation": result.label.value.lower(),
                "checks": checks,
            },
        }
        if result.historical_report_id:
            afv["Gateway"]["historical_report_id"] = result.historical_report_id
        if result.error_message:
            afv["Gateway"]["error_message"] = result.error_message

        afv_path = factor_dir / "afv.json"
        afv_path.write_text(json.dumps(afv, indent=2, ensure_ascii=False), encoding="utf-8")
        return afv_path

    def move_to_temp(self, factor_dir: Path, temp_base: Optional[Path] = None) -> Path:
        """将因子目录整体转移到 tier0/gateway_temp/。

        Args:
            factor_dir: 源因子目录路径。
            temp_base: 临时库根目录，默认由 config 路径构造。

        Returns:
            转移后的目标目录路径。
        """
        if temp_base is None:
            from config_manager import ConfigManager
            try:
                temp_base = ConfigManager().path("gateway_temp")
            except Exception:
                raise ValueError(
                    "Gateway temp_base 必须从外部配置传入"
                    "（ConfigManager.path('gateway_temp')）"
                )
        target = temp_base / factor_dir.name
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(factor_dir), str(target))
        return target

    @staticmethod
    def route_factor(temp_dir: Path, target_base: Path) -> Path:
        """根据 afv.json 将因子从临时库路由到目标库（直接转移路径）。

        Args:
            temp_dir: tier0/gateway_temp 中的因子目录。
            target_base: 目标库根目录（如 tier1/gateway_pass_base）。

        Returns:
            路由后的目标目录路径。
        """
        target_dir = target_base / temp_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)
        # 直接转移整个目录
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.move(str(temp_dir), str(target_dir))
        return target_dir

    def validate_output(self, candidate_id: str, output_dir: Path) -> bool:
        """
        验证输出目录的完整性（用于测试/调试）

        检查:
        - candidate.json 存在且格式正确
        - manifest.json 存在且格式正确
        - candidate_id 匹配

        Args:
            candidate_id: 候选因子 ID
            output_dir: 输出目录路径

        Returns:
            True 表示验证通过
        """
        candidate_file = output_dir / "candidate.json"
        manifest_file = output_dir / "manifest.json"

        # 检查文件存在
        if not candidate_file.exists():
            return False
        if not manifest_file.exists():
            return False

        # 验证 candidate.json
        try:
            with open(candidate_file, "r", encoding="utf-8") as f:
                candidate_data = json.load(f)

            # 检查 candidate_id
            if candidate_data.get("candidate_id") != candidate_id:
                return False

            # 检查 Gateway 段落是否存在
            if "Gateway" not in candidate_data:
                return False

        except (json.JSONDecodeError, KeyError):
            return False

        # 验证 manifest.json
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)

            # 检查必需字段
            required_fields = ["schema_version", "library", "candidate_id", "producer", "run_id"]
            for field in required_fields:
                if field not in manifest_data:
                    return False

        except json.JSONDecodeError:
            return False

        return True


def generate_candidate_hash(expr: str, config: Dict[str, Any]) -> str:
    """
    生成候选因子的哈希值（用于缓存目录）

    基于表达式和配置内容的哈希

    Args:
        expr: 因子表达式
        config: 因子配置

    Returns:
        哈希字符串（十六进制）
    """
    import hashlib

    # 规范化表达式（去除空格、统一大小写）
    normalized_expr = expr.strip().lower().replace(" ", "")

    # 规范化配置（排序后转为字符串）
    config_str = json.dumps(config, sort_keys=True)

    # 组合并计算哈希
    combined = f"{normalized_expr}|{config_str}"
    return hashlib.sha256(combined.encode()).hexdigest()


def generate_expr_hash(expr: str) -> str:
    """
    生成表达式的哈希值

    Args:
        expr: 因子表达式

    Returns:
        哈希字符串（十六进制）
    """
    import hashlib
    normalized_expr = expr.strip().lower().replace(" ", "")
    return hashlib.sha256(normalized_expr.encode()).hexdigest()


def generate_run_id() -> str:
    """
    生成唯一的运行 ID

    格式: gateway_YYYYMMDD_HHMMSS_xxxxxxxx
    其中 xxxxxxxx 是随机 8 位十六进制

    Returns:
        运行 ID 字符串
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    random_suffix = uuid4().hex[:8]
    return f"gateway_{timestamp}_{random_suffix}"


def generate_report_md(
        candidate_id: str,
        result: GatewayResult,
        candidate_dict: Dict[str, Any]
) -> str:
    """
    生成 Markdown 格式的缓存报告

    Args:
        candidate_id: 候选因子 ID
        result: 网关结果
        candidate_dict: 候选因子字典

    Returns:
        Markdown 格式的报告内容
    """
    expr = candidate_dict.get("Expr", "N/A")

    lines = [
        f"# Gateway Report for {candidate_id}",
        "",
        "## Summary",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| **Run ID** | `{result.run_id}` |",
        f"| **Checked At** | {result.checked_at} |",
        f"| **Label** | `{result.label.value}` |",
        f"| **Reason** | {result.reason or 'N/A'} |",
        "",
        "## Expression",
        "```python",
        expr[:500] + ("..." if len(expr) > 500 else ""),
        "```",
        "",
        "## Check Results",
        "| Check | Status | Details |",
        "|-------|--------|---------|",
    ]

    # Schema 校验
    schema_status = "✅ Pass" if result.is_legal is not False else "❌ Fail"
    lines.append(f"| Schema | {schema_status} | - |")

    # 算子白名单
    op_status = "✅ Pass" if result.is_legal is not False else "❌ Fail"
    lines.append(f"| Operators | {op_status} | - |")

    # 未来函数
    future_status = "✅ Pass" if not result.has_future else "❌ Fail"
    future_detail = f"Found: {result.error_message}" if result.has_future else "-"
    lines.append(f"| Future Function | {future_status} | {future_detail} |")

    # 复杂度
    if result.complexity_score is not None:
        budget_status = "✅ Pass" if result.is_within_budget else "❌ Fail"
        lines.append(f"| Complexity | {budget_status} | Score: {result.complexity_score:.2f} |")
    else:
        lines.append(f"| Complexity | ⏭️ Skipped | - |")

    # 去重
    if result.is_duplicate is not None:
        dup_status = "⚠️ Duplicate" if result.is_duplicate else "✅ New"
        dup_detail = f"Historical ID: {result.historical_report_id}" if result.historical_report_id else "-"
        lines.append(f"| Deduplication | {dup_status} | {dup_detail} |")
    else:
        lines.append(f"| Deduplication | ⏭️ Skipped | - |")

    lines.extend([
        "",
        "## Full Candidate Info",
        "```json",
        json.dumps(candidate_dict, indent=2, ensure_ascii=False)[:2000],
        ("..." if len(json.dumps(candidate_dict)) > 2000 else ""),
        "```",
        "",
        "---",
        f"*Report generated by Gateway | {result.run_id}*",
    ])

    return "\n".join(lines)