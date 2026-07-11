"""
网关命令行入口

功能:
- 单个候选因子处理
- 批量处理（按 campaign）
- 支持配置文件自定义
- 支持历史去重记录加载/保存
- 输出详细的处理日志和统计信息

使用示例:
    # 处理单个候选
    python -m gateway.main --candidate-id cand_20260528_a1b2c3d4

    # 批量处理某个 campaign
    python -m gateway.main --campaign-id campaign_alpha_001

    # 指定配置文件
    python -m gateway.main --candidate-id xxx --config /path/to/config.yaml

    # 加载历史去重记录
    python -m gateway.main --candidate-id xxx --history-file ./dedup_history.json

    # 显示帮助
    python -m gateway.main --help
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from gateway.config import GatewayConfig, ConfigError
from gateway.gateway_core import GatewayCore
from gateway.io_utils import (
    IOManager,
    generate_run_id,
    generate_report_md,
    generate_candidate_hash,
    generate_expr_hash
)
from gateway.models import GatewayLabel, ValidationStatus
from gateway.exceptions import GatewayError


# 配置日志
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """设置日志配置"""
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding='utf-8'))

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def get_logger(name: str) -> logging.Logger:
    """获取日志记录器"""
    return logging.getLogger(name)


class GatewayCLI:
    """网关命令行处理器"""

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化 CLI

        Args:
            config_path: 配置文件路径（可选）
        """
        self.start_time = None
        self.logger = get_logger("gateway_cli")

        try:
            if config_path:
                config_dir = Path(config_path).parent
                self.config = GatewayConfig(config_dir)
                self.logger.info(f"Loaded config from: {config_path}")
            else:
                self.config = GatewayConfig()
                self.logger.info("Loaded default config")
        except ConfigError as e:
            self.logger.error(f"Config error: {e}")
            sys.exit(1)

        self.gateway = GatewayCore(self.config)
        self.io = IOManager(self.config)
        self.stats = {
            "processed": 0,
            "passed": 0,
            "duplicated": 0,
            "rejected": 0,
            "errors": 0
        }

    def run(self, args: argparse.Namespace) -> int:
        """
        运行 CLI

        Args:
            args: 命令行参数

        Returns:
            退出码（0 表示成功）
        """
        self.start_time = time.time()

        # 设置日志
        setup_logging(args.log_level, args.log_file)

        self.logger.info("=" * 60)
        self.logger.info("Gateway Service Starting")
        self.logger.info("=" * 60)

        # 加载历史去重记录
        if args.history_file:
            self._load_history(args.history_file)

        # 执行处理
        exit_code = 0
        try:
            if args.candidate_id:
                # 单个候选处理
                self._process_single(args.candidate_id)
            elif args.campaign_id:
                # 批量处理
                self._process_campaign(args.campaign_id)
            else:
                self.logger.error("Please specify --candidate-id or --campaign-id")
                return 1

            # 保存历史去重记录
            if args.save_history:
                self._save_history(args.save_history)

            # 打印统计信息
            self._print_summary()

        except KeyboardInterrupt:
            self.logger.warning("Interrupted by user")
            exit_code = 1
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}", exc_info=True)
            exit_code = 1

        return exit_code

    def _process_single(self, candidate_id: str) -> None:
        """处理单个候选因子"""
        self.logger.info(f"Processing single candidate: {candidate_id}")

        try:
            # 1. 读取候选因子
            candidate_dict, candidate = self.io.read_candidate(candidate_id)
            self.logger.info(f"  Loaded candidate: {candidate.expr[:80]}...")

            # 2. 执行网关处理
            result = self.gateway.process(candidate_dict)

            # 3. 更新 candidate.json 中的 Gateway 段落
            candidate_dict["Gateway"] = {
                "Label": result.label.value,
                "Reason": result.reason,
                "run_id": result.run_id,
                "checked_at": result.checked_at
            }

            # 4. 写入输出
            output_dir = self.io.write_output(candidate_id, candidate_dict, result)
            self.logger.info(f"  Output written to: {output_dir}")

            # 5. 生成并写入缓存报告
            expr_hash = generate_expr_hash(candidate.expr)
            candidate_hash = generate_candidate_hash(candidate.expr, candidate.config)
            report_md = generate_report_md(candidate_id, result, candidate_dict)
            cache_dir = self.io.write_cache_report(candidate_hash, expr_hash, result, report_md)
            self.logger.info(f"  Cache report written to: {cache_dir}")

            # 6. 如果是 Rejected，写入额外日志
            if result.label == GatewayLabel.REJECTED:
                log_file = self.io.write_rejected_log(candidate_id, candidate_dict, result)
                self.logger.info(f"  Rejected log written to: {log_file}")

            # 7. 更新统计
            self._update_stats(result)

            # 8. 打印结果
            self._print_result(candidate_id, result)

        except FileNotFoundError as e:
            self.logger.error(f"Candidate not found: {e}")
            self.stats["errors"] += 1
            raise
        except Exception as e:
            self.logger.error(f"Failed to process {candidate_id}: {e}")
            self.stats["errors"] += 1
            raise

        #     # 如果网关 Pass，执行数据质量检测
        # if result.label == GatewayLabel.PASS:
        #     from gateway.data_quality import run_data_quality
        #
        #     dq_result = run_data_quality(
        #         str(output_dir / "candidate.json"),
        #         self.market_data_path
        #     )
        #
        #     self.logger.info(f"  Data Quality: {dq_result['DataQuality']['Label']}")

    def _process_campaign(self, campaign_id: str) -> None:
        """批量处理 campaign 下的所有候选因子"""
        self.logger.info(f"Processing campaign: {campaign_id}")

        # 构建候选池路径
        candidate_pool_dir = Path(self.config.paths.candidate_pool)
        if not candidate_pool_dir.exists():
            self.logger.error(f"Candidate pool directory not found: {candidate_pool_dir}")
            return

        # 查找 campaign 下的所有候选
        # 注意：这里假设候选目录直接在 candidate_pool 下
        # 实际结构可能是 {candidate_pool}/{campaign_id}/{candidate_id}/
        # 根据你的实际结构调整
        candidate_dirs = []

        # 方式1：campaign 直接包含候选目录
        campaign_dir = candidate_pool_dir / campaign_id
        if campaign_dir.exists() and campaign_dir.is_dir():
            for subdir in campaign_dir.iterdir():
                if subdir.is_dir() and (subdir / "candidate.json").exists():
                    candidate_dirs.append(subdir.name)
        else:
            # 方式2：候选目录直接以 campaign_id 为前缀
            for subdir in candidate_pool_dir.iterdir():
                if subdir.is_dir() and subdir.name.startswith(campaign_id):
                    if (subdir / "candidate.json").exists():
                        candidate_dirs.append(subdir.name)

        if not candidate_dirs:
            self.logger.warning(f"No candidates found for campaign: {campaign_id}")
            return

        self.logger.info(f"Found {len(candidate_dirs)} candidates")

        # 批量处理
        for i, candidate_id in enumerate(candidate_dirs, 1):
            self.logger.info(f"\n[{i}/{len(candidate_dirs)}] Processing: {candidate_id}")
            try:
                self._process_single(candidate_id)
            except Exception as e:
                self.logger.error(f"Failed to process {candidate_id}: {e}")
                continue

        self.logger.info(f"\nCampaign {campaign_id} processing completed")

    def _update_stats(self, result) -> None:
        """更新统计信息"""
        self.stats["processed"] += 1
        if result.label == GatewayLabel.PASS:
            self.stats["passed"] += 1
        elif result.label == GatewayLabel.DUPLICATED:
            self.stats["duplicated"] += 1
        elif result.label == GatewayLabel.REJECTED:
            self.stats["rejected"] += 1

    def _print_result(self, candidate_id: str, result) -> None:
        """打印处理结果"""
        label_icon = {
            GatewayLabel.PASS: "✅",
            GatewayLabel.DUPLICATED: "⚠️",
            GatewayLabel.REJECTED: "❌"
        }.get(result.label, "❓")

        self.logger.info(f"  {label_icon} Result: {result.label.value}")
        if result.reason:
            self.logger.info(f"    Reason: {result.reason[:100]}")

        # 打印诊断信息
        if result.complexity_score is not None:
            self.logger.info(f"    Complexity: {result.complexity_score:.2f}")
        if result.is_duplicate:
            self.logger.info(f"    Duplicate: Yes (historical: {result.historical_report_id})")

    def _print_summary(self) -> None:
        """打印处理总结"""
        elapsed = time.time() - self.start_time

        self.logger.info("\n" + "=" * 60)
        self.logger.info("Processing Summary")
        self.logger.info("=" * 60)
        self.logger.info(f"  Total processed: {self.stats['processed']}")
        self.logger.info(f"  ✅ Passed:       {self.stats['passed']}")
        self.logger.info(f"  ⚠️ Duplicated:   {self.stats['duplicated']}")
        self.logger.info(f"  ❌ Rejected:     {self.stats['rejected']}")
        self.logger.info(f"  🔴 Errors:       {self.stats['errors']}")
        self.logger.info(f"  ⏱️  Time:         {elapsed:.2f}s")

        if self.stats['processed'] > 0:
            gateway_stats = self.gateway.get_stats()
            self.logger.info("\n  Gateway Internal Stats:")
            self.logger.info(f"    Total: {gateway_stats['total_processed']}")
            self.logger.info(f"    Pass:  {gateway_stats['pass_count']}")
            self.logger.info(f"    Duplicated:  {gateway_stats['duplicated_count']}")
            self.logger.info(f"    Rej:   {gateway_stats['rejected_count']}")

        self.logger.info("=" * 60)

    def _load_history(self, history_file: str) -> None:
        """加载历史去重记录"""
        history_path = Path(history_file)
        if history_path.exists():
            self.logger.info(f"Loading history from: {history_file}")
            self.gateway.load_history(history_file)
            cache_size = self.gateway.deduplicator.get_cache_size()
            self.logger.info(f"  Loaded {cache_size} historical records")
        else:
            self.logger.warning(f"History file not found: {history_file}")

    def _save_history(self, history_file: str) -> None:
        """保存历史去重记录"""
        self.logger.info(f"Saving history to: {history_file}")
        self.gateway.save_history(history_file)
        cache_size = self.gateway.deduplicator.get_cache_size()
        self.logger.info(f"  Saved {cache_size} records")


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="Factor Gateway - 因子极速网关",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single candidate
  python -m gateway.main --candidate-id cand_20260528_a1b2c3d4

  # Process campaign
  python -m gateway.main --campaign-id campaign_alpha_001

  # With custom config
  python -m gateway.main --candidate-id xxx --config ./my_config.yaml

  # With history deduplication
  python -m gateway.main --candidate-id xxx --history-file ./history.json --save-history ./history.json
        """
    )

    # 输入参数（二选一）
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--candidate-id",
        type=str,
        help="单个候选因子 ID"
    )
    input_group.add_argument(
        "--campaign-id",
        type=str,
        help="批量处理 campaign ID"
    )

    # 配置文件
    parser.add_argument(
        "--config",
        type=str,
        help="配置文件路径 (默认: gateway/configs/gateway_config.yaml)"
    )

    # 历史去重
    parser.add_argument(
        "--history-file",
        type=str,
        help="历史去重记录文件路径（加载）"
    )
    parser.add_argument(
        "--save-history",
        type=str,
        help="保存历史去重记录到文件"
    )

    # 日志
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别 (默认: INFO)"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="日志输出文件路径（可选）"
    )

    # 其他
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="静默模式，减少输出"
    )

    return parser


def main():
    """主入口函数"""
    parser = create_parser()
    args = parser.parse_args()

    # 静默模式调整日志级别
    if args.quiet:
        args.log_level = "WARNING"

    # 创建 CLI 并运行
    cli = GatewayCLI(config_path=args.config)
    exit_code = cli.run(args)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()