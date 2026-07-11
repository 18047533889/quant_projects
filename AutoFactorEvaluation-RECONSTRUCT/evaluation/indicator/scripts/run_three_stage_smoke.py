"""可选的三段链路 smoke 验证脚本。

所属模块:
    evaluation/indicator

对应文档:
    docs/FID_stage4_2_indicator_metrics.md

文件职责:
    当 repo 根目录同时存在 `restructure_ts_1/` 和 `yurui/` 时，临时生成上游
    所需输入，运行上游阶段 4-1、本模块阶段 4-2，以及下游 Yurui/Dev4
    tagging/routing，用于验证三段源码是否能串联。当前脚本不属于核心计算入口。
"""

from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path.cwd()
UPSTREAM_ROOT = ROOT / "restructure_ts_1"
DOWNSTREAM_ROOT = ROOT / "yurui"

if not UPSTREAM_ROOT.exists() or not DOWNSTREAM_ROOT.exists():
    missing = [str(p.name) for p in (UPSTREAM_ROOT, DOWNSTREAM_ROOT) if not p.exists()]
    raise SystemExit(
        "Missing upstream/downstream project folder(s): "
        + ", ".join(missing)
        + ". Put them under the workspace root before running this optional smoke test."
    )

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(UPSTREAM_ROOT))
sys.path.insert(0, str(DOWNSTREAM_ROOT))

from cs_performance_series.config import EvaluationConfig
from cs_performance_series.constants import (
    COL_ASSET,
    COL_DATETIME,
    COL_FACTOR_ID,
    COL_FACTOR_VALUE,
    COL_FORWARD_RETURN,
    COL_HORIZON,
    COL_IS_ACTIVE,
    COL_IS_TRADABLE,
    COL_KNOWLEDGE_TS,
)
from cs_performance_series.service import CrossSectionalPerformanceSeriesService
from dev4.routing import AdmissionRouter
from dev4.schemas import normalize_evaluation_summary, normalize_factor_meta
from dev4.storage import MockStorage
from dev4.tagging import TaggingEngine
from main_code.calculator import run_metric_calculation


FACTOR_ID = "F_EQ_D_PV_001"
EVAL_RUN_ID = "ER_PIPELINE_SMOKE_001"


def main() -> None:
    """运行可选三段 smoke；缺少上下游源码目录时在导入前退出。"""

    smoke_root = ROOT / "outputs" / "three_stage_smoke"
    if smoke_root.exists():
        shutil.rmtree(smoke_root)
    smoke_root.mkdir(parents=True, exist_ok=True)

    upstream_input_root = smoke_root / "stage41_input"
    upstream_output_root = smoke_root / "stage41_output"
    our_output_root = smoke_root / "stage42_output"
    downstream_output_root = smoke_root / "yurui_output"
    downstream_config_dir = _write_yurui_smoke_configs(smoke_root / "yurui_configs", downstream_output_root)

    upstream_dir = _write_upstream_gate_files(upstream_input_root / "upstream")
    fe, fr, universe = _synthetic_panel(n_days=90, n_assets=80)
    pure_factor_dir = _write_pure_factor_dir(upstream_input_root, FACTOR_ID, fe)

    stage41_result = CrossSectionalPerformanceSeriesService.run(
        base_output_dir=upstream_output_root,
        factor_id=FACTOR_ID,
        eval_run_id=EVAL_RUN_ID,
        pure_factor_dir=pure_factor_dir,
        upstream_dir=upstream_dir,
        forward_return=fr,
        universe=universe,
        evaluation_config=EvaluationConfig(
            horizons=[5],
            min_assets=30,
            n_quantiles=5,
            bundle_export_horizon=5,
        ),
    )

    stage42_result = run_metric_calculation(
        stage41_result.stage4_2_primary_slice_dir,
        our_output_root,
        ROOT / "database" / "log" / "evaluation" / "indicator",
    )

    eval_summary = normalize_evaluation_summary(
        json.loads((our_output_root / "dev4_evaluation_summary.json").read_text(encoding="utf-8"))
    )
    factor_meta = normalize_factor_meta(_factor_meta())

    storage = MockStorage(base_dir=downstream_output_root, config_dir=downstream_config_dir)
    storage.upsert_factor_registry(factor_meta)
    tagging_engine = TaggingEngine(storage=storage, config_dir=downstream_config_dir)
    tags_result = tagging_engine.generate_tags(
        _tag_payload(eval_summary, factor_meta),
        eval_summary,
        factor_meta,
    )
    router = AdmissionRouter(config_dir=downstream_config_dir, storage=storage)
    admission_decision = router.decide(eval_summary, tags_result["tag_package"])
    route_record = router.execute(admission_decision, factor_meta["factor_id"])

    summary = {
        "stage41_primary_slice": str(stage41_result.stage4_2_primary_slice_dir),
        "stage42_output": str(our_output_root),
        "yurui_output": str(downstream_output_root),
        "stage42_metric_rows": len(stage42_result["metrics_matrix"]),
        "admission_decision": admission_decision,
        "route_record": route_record,
    }
    (smoke_root / "three_stage_smoke_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _write_upstream_gate_files(upstream_dir: Path) -> Path:
    """写出上游服务启动所需的最小网关/协议 JSON 文件。"""

    upstream_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "evaluation_request.json": {"signal_structure": "cross_sectional"},
        "evaluation_protocol_snapshot.json": {"version": "cs_equity_daily_v1"},
        "evaluation_input_manifest.json": {},
        "data_quality_summary.json": {"status": "passed"},
        "label_spec.json": {"horizons": [5]},
        "leakage_audit_report.json": {"status": "passed"},
    }
    for name, payload in files.items():
        (upstream_dir / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return upstream_dir


def _write_yurui_smoke_configs(config_dir: Path, output_root: Path) -> Path:
    """复制并改写 Yurui 配置，使 smoke 产物落到临时输出目录。"""

    config_dir.mkdir(parents=True, exist_ok=True)
    source_config = DOWNSTREAM_ROOT / "configs"

    tag_policy = json.loads((source_config / "tag_policy.json").read_text(encoding="utf-8"))
    route_policy = json.loads((source_config / "route_policy.json").read_text(encoding="utf-8"))
    route_policy["tier_paths"] = {
        tier: str(output_root / path_name)
        for tier, path_name in {
            "Tier3A": "tier3a_core",
            "Tier3B": "tier3b_satellite",
            "Tier3C": "tier3c_feature",
            "Tier3D": "tier3d_optimized_reserve",
            "Tier2": "tier2_incubator",
            "Tier2X": "tier2x_optimization_factory",
            "Tier4": "tier4_archive",
        }.items()
    }
    storage_paths = json.loads((source_config / "storage_paths.json").read_text(encoding="utf-8"))
    storage_paths["artifact_output_root"] = str(output_root)
    storage_paths["data_lake_root"] = str(output_root / "data_lake")
    storage_paths["local_state"] = {"state_dir": str(output_root / "_state")}

    (config_dir / "tag_policy.json").write_text(json.dumps(tag_policy, ensure_ascii=False, indent=2), encoding="utf-8")
    (config_dir / "route_policy.json").write_text(json.dumps(route_policy, ensure_ascii=False, indent=2), encoding="utf-8")
    (config_dir / "storage_paths.json").write_text(json.dumps(storage_paths, ensure_ascii=False, indent=2), encoding="utf-8")
    return config_dir


def _write_pure_factor_dir(root: Path, factor_id: str, fe: pd.DataFrame) -> Path:
    """写出上游阶段 4-1 所需的最小 PureFactor 磁盘目录。"""

    pf = root / "pure_factor_base" / "price_volume" / "daily" / factor_id
    pf.mkdir(parents=True, exist_ok=True)
    candidate = _factor_meta()
    candidate.update(
        {
            "schema_version": "disk.v1",
            "Gateway": {"Label": "Pass"},
            "Assetization": {"Label": "Raw"},
            "Purification": {"Label": "Pure"},
        }
    )
    (pf / "candidate.json").write_text(json.dumps(candidate, ensure_ascii=False, indent=2), encoding="utf-8")
    fe.to_parquet(pf / "data.parquet", index=False)
    (pf / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "disk.v1",
                "library": "pure_factor_base",
                "artifact_type": "PureFactor",
                "factor_id": factor_id,
                "files": {"candidate": "candidate.json", "data": "data.parquet"},
                "validation": {"status": "passed", "errors": []},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return pf


def _synthetic_panel(n_days: int, n_assets: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """构造上游 smoke 使用的因子暴露、forward return 和 universe 面板。"""

    rng = np.random.default_rng(20260530)
    rows_fe = []
    rows_fr = []
    rows_u = []
    dates = pd.bdate_range("2025-01-02", periods=n_days)
    for dt in dates:
        common_noise = rng.normal(0, 0.01)
        for i in range(n_assets):
            asset = f"A{i:04d}"
            factor_value = rng.standard_normal()
            forward_return = 0.035 * factor_value + common_noise + rng.normal(0, 0.08)
            rows_fe.append(
                {
                    COL_DATETIME: dt,
                    COL_ASSET: asset,
                    COL_FACTOR_ID: FACTOR_ID,
                    COL_FACTOR_VALUE: factor_value,
                    COL_KNOWLEDGE_TS: dt,
                }
            )
            rows_fr.append(
                {
                    COL_DATETIME: dt,
                    COL_ASSET: asset,
                    COL_HORIZON: 5,
                    COL_FORWARD_RETURN: forward_return,
                }
            )
            rows_u.append(
                {
                    COL_DATETIME: dt,
                    COL_ASSET: asset,
                    COL_IS_ACTIVE: True,
                    COL_IS_TRADABLE: True,
                }
            )
    return pd.DataFrame(rows_fe), pd.DataFrame(rows_fr), pd.DataFrame(rows_u)


def _factor_meta() -> dict:
    """构造下游 Yurui/Dev4 smoke 使用的最小 factor metadata。"""

    return {
        "factor_id": FACTOR_ID,
        "candidate_id": "C_PIPELINE_SMOKE_001",
        "expression": "rank(ts_delta(close, 5)) / rank(ts_sum(volume, 5))",
        "formula_ast": {
            "op": "div",
            "left": {"op": "rank", "args": [{"op": "ts_delta", "args": ["close", 5]}]},
            "right": {"op": "rank", "args": [{"op": "ts_sum", "args": ["volume", 5]}]},
        },
        "generator_name": "LocalSmoke",
        "campaign_id": "pipeline_alignment",
        "iteration_id": "iter_001",
        "batch_id": str(uuid.uuid4()),
        "signal_structure": "cross_sectional",
        "asset_class": "equity",
        "frequency_bucket": "daily",
        "domain_root": "price_volume",
        "lib_coordinates": {},
    }


def _tag_payload(eval_summary: dict, factor_meta: dict) -> dict:
    """把阶段 4-2 摘要和 factor metadata 组装成 Yurui tagging 入参。"""

    return {
        "factor_id": factor_meta["factor_id"],
        "expression": factor_meta["expression"],
        "ast": factor_meta["ast"],
        "metric_summary": eval_summary["summary_scorecard"],
        "style_exposure": {},
        "correlation_info": {},
    }


if __name__ == "__main__":
    main()
