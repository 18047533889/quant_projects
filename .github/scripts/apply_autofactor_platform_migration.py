from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "AutoFactorEvaluation-RECONSTRUCT"


def replace_once(text: str, pattern: str, replacement: str, *, label: str) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.S)
    if count != 1:
        raise RuntimeError(f"{label}: expected one replacement, got {count}")
    return updated


def patch_gateway_core() -> None:
    path = PROJECT / "gateway" / "scripts" / "gateway_core.py"
    text = path.read_text(encoding="utf-8")
    if "import json\n" not in text:
        text = text.replace("import hashlib\n", "import hashlib\nimport json\n", 1)

    step2 = '''def _step2_build_yaml(manifest: dict) -> StepResult:
    try:
        from integrations.quant_platform import build_factor_engine_config

        config = build_factor_engine_config(
            manifest,
            backend=str(manifest.get("backend", "pandas")),
            run_mode="research",
        )
        if not config["factor"]["expr"]:
            return StepResult("Step2_YamlConfig", "REJECTED", "因子公式为空")
        return StepResult(
            "Step2_YamlConfig",
            "PASS",
            detail={"engine_config": config, "data_source_type": "data_access"},
        )
    except Exception as exc:
        return StepResult("Step2_YamlConfig", "REJECTED", f"平台配置构建失败: {exc}")
'''
    text = replace_once(
        text,
        r"def _step2_build_yaml\(manifest: dict\) -> StepResult:\n.*?(?=\n# =+\n# Step 3)",
        step2,
        label="gateway step2",
    )

    step4 = '''def _step4_tiny_run(manifest: dict) -> StepResult:
    try:
        import os

        from integrations.quant_platform import (
            build_factor_engine_config,
            execute_factor_formula,
            validate_factor_formula,
        )

        config = build_factor_engine_config(
            manifest,
            backend=str(manifest.get("backend", "pandas")),
            run_mode="research",
        )
        factor_cfg = config["factor"]
        data_cfg = config["data_source"]
        start_date = data_cfg.get("start_date") or os.environ.get("AUTOFACTOR_TINY_START")
        end_date = data_cfg.get("end_date") or os.environ.get("AUTOFACTOR_TINY_END")

        if not start_date or not end_date:
            _, plan, analysis = validate_factor_formula(
                factor_cfg["expr"],
                name=factor_cfg["name"],
                freq=factor_cfg["freq"],
                universe=factor_cfg.get("universe"),
                backend=config["backend"]["type"],
                run_mode="research",
            )
            return StepResult(
                "Step4_TinyRun",
                "PASS",
                detail={
                    "compile_only": True,
                    "reason": "未配置 AUTOFACTOR_TINY_START/END，已完成真实 FactorEngine 编译",
                    "lookback": getattr(analysis, "lookback", None),
                    "root_op": getattr(plan, "op", None),
                },
            )

        execution = execute_factor_formula(
            factor_cfg["expr"],
            factor_name=factor_cfg["name"],
            market=str(manifest.get("market") or "ashare"),
            dataset=data_cfg["dataset"],
            fields=data_cfg.get("fields"),
            start_date=start_date,
            end_date=end_date,
            instrument_filter=data_cfg.get("instrument_filter"),
            params=data_cfg.get("params"),
            backend=config["backend"]["type"],
            run_mode="research",
            freq=factor_cfg["freq"],
            universe=factor_cfg.get("universe"),
            description=factor_cfg.get("description"),
        )
        series = execution.result
        return StepResult(
            "Step4_TinyRun",
            "PASS",
            detail={
                "compile_only": False,
                "rows": len(series),
                "non_null": int(series.notna().sum()),
                "data_snapshot_id": execution.snapshot_id,
                "dataset": data_cfg["dataset"],
            },
        )
    except Exception as exc:
        import traceback

        return StepResult(
            "Step4_TinyRun",
            "REJECTED",
            f"当前 FactorEngine/DataAccess 试运行失败: {exc}",
            detail={"traceback": traceback.format_exc()},
        )
'''
    text = replace_once(
        text,
        r"def _step4_tiny_run\(manifest: dict\) -> StepResult:\n.*?(?=\n# =+\n# Step 5)",
        step4,
        label="gateway step4",
    )
    path.write_text(text, encoding="utf-8")


def patch_data_quality() -> None:
    path = PROJECT / "gateway" / "scripts" / "data_quality.py"
    text = path.read_text(encoding="utf-8")
    replacement = '''    def _load_data(self, source: str) -> Optional[pd.DataFrame]:
        """从 DataAccess 登记数据集读取质量检测样本，不再直接扫描 Parquet。"""
        try:
            import os

            from integrations.quant_platform import load_market_frame

            known = {"ashare_stock_daily", "us_stock_daily", "us_stocks_sip_day_aggs"}
            dataset = source if source in known else os.environ.get("AUTOFACTOR_MARKET_DATASET")
            market = "us" if str(dataset or source).startswith("us_") else os.environ.get(
                "AUTOFACTOR_MARKET", "ashare"
            )
            frame, snapshot = load_market_frame(
                market=market,
                dataset=dataset,
                fields=["close", "high", "low", "volume", "vwap"],
                start_date=os.environ.get("AUTOFACTOR_DQ_START"),
                end_date=os.environ.get("AUTOFACTOR_DQ_END"),
            )
            frame["timestamp"] = frame["datetime"]
            self.logger.info(
                "DataAccess DQ input dataset=%s rows=%d snapshot=%s",
                dataset,
                len(frame),
                snapshot.snapshot_id,
            )
            return frame
        except Exception as exc:
            self.logger.error("DataAccess load failed: %s", exc)
            return None
'''
    text = replace_once(
        text,
        r"    def _load_data\(self, path: str\) -> Optional\[pd\.DataFrame\]:\n.*?(?=\n    def _calc_score)",
        replacement,
        label="data quality loader",
    )
    path.write_text(text, encoding="utf-8")


def patch_config_example() -> None:
    path = PROJECT / "config.yaml.example"
    text = path.read_text(encoding="utf-8")
    block = '''market_data:
  market: ashare
  dataset: ashare_stock_daily
    # 正式行情由 data_access/config/datasets.yaml 管理；禁止业务代码直接扫描路径
  start_date: null
  end_date: null
  forward_return_field: vwap
  adjustment_field: Factor

industry_data:
  dataset: ashare_stock_industry
'''
    text = replace_once(
        text,
        r"market_data:\n.*?(?=  source_field:)",
        block,
        label="config market data",
    )
    path.write_text(text, encoding="utf-8")


def patch_requirements() -> None:
    path = PROJECT / "requirements.txt"
    lines = path.read_text(encoding="utf-8").splitlines()
    output: list[str] = []
    for line in lines:
        if line.strip().startswith("fastparquet"):
            continue
        if line.strip().startswith("vectorbt>=1.0.0"):
            output.append("vectorbt>=0.26.0")
        else:
            output.append(line)
    joined = "\n".join(output).rstrip() + "\n"
    additions: list[str] = []
    if "duckdb>=" not in joined:
        additions.append("duckdb>=1.0.0")
    if "polars>=" not in joined:
        additions.append("polars>=1.0.0")
    if additions:
        marker = "# --- 配置 ---"
        addition_block = "# --- 统一计算/数据平台 ---\n" + "\n".join(additions) + "\n\n"
        if marker in joined:
            joined = joined.replace(marker, addition_block + marker, 1)
        else:
            joined += "\n" + addition_block
    path.write_text(joined, encoding="utf-8")


def add_gateway_compatibility() -> None:
    outer = PROJECT / "gateway"
    (outer / "__init__.py").write_text(
        '"""Gateway package。正式编排入口位于 gateway.scripts。"""\n',
        encoding="utf-8",
    )
    inner = outer / "gateway"
    for name in (
        "config",
        "complexity",
        "deduplicator",
        "future_scanner",
        "io_utils",
        "kafka_producer",
        "validator",
    ):
        target = inner / f"{name}.py"
        target.write_text(
            f'"""兼容旧 gateway.{name} 导入。"""\n'
            f'try:\n    from gateway.scripts.{name} import *  # noqa: F401,F403\n'
            f'except ImportError:\n    from scripts.{name} import *  # type: ignore # noqa: F401,F403\n',
            encoding="utf-8",
        )


def add_documentation() -> None:
    path = PROJECT / "docs" / "quant_platform_integration.md"
    content = """# Quant platform integration

`AutoFactorEvaluation-RECONSTRUCT` 不再携带私有 FactorEngine 副本。

- DSL parser、IR、planner、算子和执行后端：仓库根 `factor_engine/`。
- 行情、快照、schema、查询预算和因子 staging：仓库根 `data_access/`。
- 唯一适配入口：`integrations/quant_platform.py`。
- A 股默认数据集：`ashare_stock_daily`；美股默认：`us_stock_daily`。
- 因子产物先写 `factor_lake_staging`，只有显式 `publish=True` 才晋升。
- `expression_type=python/code` 在生产接入中 fail-closed，必须转换为 DSL。

运行时需令仓库根和 `factor_engine/` 可导入。根 CI 已设置对应 `PYTHONPATH`；
服务器脚本可从仓库根启动，或设置 `QUANT_PROJECTS_ROOT`。
"""
    path.write_text(content, encoding="utf-8")


def cleanup() -> None:
    nested = PROJECT / "factor_engine"
    if nested.exists():
        shutil.rmtree(nested)
    for backup in PROJECT.rglob("*.bak"):
        backup.unlink()


def main() -> None:
    if not PROJECT.is_dir():
        raise RuntimeError(f"missing project: {PROJECT}")
    patch_gateway_core()
    patch_data_quality()
    patch_config_example()
    patch_requirements()
    add_gateway_compatibility()
    add_documentation()
    cleanup()
    print("AutoFactorEvaluation platform migration applied")


if __name__ == "__main__":
    main()
