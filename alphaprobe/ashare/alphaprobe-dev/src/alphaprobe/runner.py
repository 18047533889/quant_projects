"""AlphaPROBE 单轮挖掘入口（训练 + candidate_pool 投递）。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv

from alphaprobe.cold_start import (
    ALPHA101_DEFAULT_YAML,
    ALPHA158_DEFAULT_YAML,
    ALPHA191_DEFAULT_YAML,
    DEFAULT_YAML,
    load_cold_start_for_training,
    load_cold_start_mixed_for_training,
    load_cold_start_multi_library_for_training,
    recommended_max_backtrack_days,
)
from alphaprobe.delivery.config import ExperimentConfig
from alphaprobe.delivery.exporter import DiskV1DeliveryExporter
from alphaprobe.fe_bridge.bootstrap import enable_factor_engine_evaluation
from alphaprobe.fe_bridge.stock_data import FactorEngineStockData
from alphaprobe.trainer.pool import AlphaKnowledgePool
from alphaprobe.trainer.trainer import AlphaKnowledgeTrainer
from shared.alphagen.data.expression import Feature, Ref
from shared.alphagen_qlib.stock_data import FeatureType, StockData

# alphaprobe-dev 项目根（src/alphaprobe/runner.py → parents[2]）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT = PROJECT_ROOT / "configs" / "alphaprobe" / "experiment_ashare_pv.yaml"

# Phase 7：continuous 在每轮 campaign 结束后读取本进程内 pool 快照做全池
# upsert（跨轮 lineage 持久化）。本进程单轮挖掘场景下不入 memory，仅占位。
_LAST_POOL: list[Any] = [None]
_LAST_POOL_ATTR = "value"


def _last_pool_value() -> Any:
    return _LAST_POOL[0]


def _set_last_pool(pool: Any) -> None:
    _LAST_POOL[0] = pool

# 项目 .env 优先于 shell 里残留的旧 OPENAI_*（如先前其它网关）
load_dotenv(PROJECT_ROOT / ".env", override=True)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def _apply_checkpoint_settings(experiment: ExperimentConfig, args: Any, project_root: Path) -> None:
    mining_raw = experiment.raw.get("mining") or {}
    cp_raw = mining_raw.get("checkpoint") or {}
    from alphaprobe.trainer.checkpoint import resolve_checkpoint_dir

    args.checkpoint_dir = str(
        resolve_checkpoint_dir(project_root, experiment.delivery.campaign_id, mining_raw)
    )
    args.checkpoint_enabled = bool(cp_raw.get("enabled", True))
    if getattr(args, "resume", None) is not None:
        args.checkpoint_resume = bool(args.resume)
    else:
        args.checkpoint_resume = bool(cp_raw.get("resume", True))


def _init_prompts(experiment: ExperimentConfig) -> None:
    mining_raw = experiment.raw.get("mining") or {}
    prompts_cfg = mining_raw.get("prompts") or {}
    prompt_dir = prompts_cfg.get("dir")
    if not prompt_dir:
        return
    from shared.utils.prompt_loader import set_prompts_dir
    from shared.utils.prompt import reload_prompts

    set_prompts_dir(prompt_dir)
    reload_prompts()
    print(f"[prompts] loaded from {prompt_dir}")


def _resolve_cold_start_library_paths(
    experiment: ExperimentConfig,
    mining_raw: dict,
    project_root: Path,
    default_pv_yaml: Path,
) -> dict[str, Path]:
    mix_raw = mining_raw.get("cold_start_mix") or {}
    libs_raw = dict(mix_raw.get("libraries") or {})
    market = str(experiment.delivery.market or "ashare").lower()

    paths: dict[str, Path] = {
        "pv": default_pv_yaml,
        "alpha101": ALPHA101_DEFAULT_YAML,
        "alpha191": ALPHA191_DEFAULT_YAML,
        "alpha158": ALPHA158_DEFAULT_YAML,
    }
    if market not in ("ashare", "a_share", "cn"):
        print(f"[cold_start] market={market}: 请在 cold_start_mix.libraries 指定对应 yaml")

    for key, val in libs_raw.items():
        if not val:
            continue
        p = Path(str(val)).expanduser()
        if not p.is_absolute():
            p = project_root / p
        paths[key] = p
    return paths


def resolve_cold_start(
    experiment: ExperimentConfig,
    args: Any,
    project_root: Path = PROJECT_ROOT,
) -> list[tuple[str, str, str]]:
    mining_raw = experiment.raw.get("mining") or {}
    mix_raw = mining_raw.get("cold_start_mix") or {}

    yaml_path = args.cold_start_library or mining_raw.get("cold_start_library")
    if yaml_path:
        default_pv = Path(yaml_path).expanduser()
        if not default_pv.is_absolute():
            default_pv = project_root / default_pv
    else:
        default_pv = DEFAULT_YAML

    sample_size = args.cold_start_sample_size
    if sample_size is None:
        sample_size = mining_raw.get("cold_start_sample_size", 100)

    seed = args.cold_start_seed
    if seed is None and mining_raw.get("cold_start_seed") is not None:
        seed = int(mining_raw["cold_start_seed"])

    if mix_raw.get("enabled", True) and sample_size:
        ratios = dict(mix_raw.get("ratios") or {
            "pv": 0.7,
            "alpha101": 0.1,
            "alpha191": 0.1,
            "alpha158": 0.1,
        })
        library_paths = _resolve_cold_start_library_paths(
            experiment, mining_raw, project_root, default_pv
        )
        pv_structural = bool(mix_raw.get("pv_structural_diversity", True))
        entries = load_cold_start_multi_library_for_training(
            total_size=int(sample_size),
            ratios=ratios,
            library_paths={k: str(v) for k, v in library_paths.items()},
            seed=seed,
            pv_structural_diversity=pv_structural,
        )
        print(
            f"[cold_start] multi-library {len(entries)} seeds "
            f"(ratios={ratios}, structural_pv={pv_structural}, seed={seed})"
        )
        return entries

    pv_ratio = mining_raw.get("cold_start_pv_ratio")
    if pv_ratio is None:
        pv_ratio = getattr(args, "cold_start_pv_ratio", None)
    if pv_ratio is not None and sample_size:
        entries = load_cold_start_mixed_for_training(
            total_size=int(sample_size),
            pv_yaml=default_pv,
            alpha191_yaml=ALPHA191_DEFAULT_YAML,
            pv_ratio=float(pv_ratio),
            seed=seed,
        )
        print(f"[cold_start] legacy mixed {len(entries)} seeds (pv_ratio={pv_ratio})")
        return entries

    entries = load_cold_start_for_training(
        yaml_path=default_pv,
        sample_size=sample_size,
        seed=seed,
    )
    print(f"[cold_start] loaded {len(entries)} seeds from {default_pv}")
    return entries


def _apply_env_llm_settings(experiment: ExperimentConfig) -> None:
    model = os.getenv("OPENAI_MODEL_NAME", "deepseek-v4-flash").strip()
    base = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com").strip()
    provider = os.getenv("OPENAI_PROVIDER", "deepseek").strip()
    experiment.delivery.llm_model = model
    experiment.delivery.llm_base_url = base
    experiment.delivery.llm_provider = provider


def _apply_mining_yaml_defaults(experiment: ExperimentConfig, args: Any) -> None:
    m = experiment.mining
    args.label_days = m.label_days
    args.pool_capacity = m.pool_capacity
    args.search_time = m.search_time
    args.ic_threshold = m.ic_export_threshold
    experiment.delivery.ic_export_threshold = m.ic_export_threshold


def _make_stock_data(
    experiment: ExperimentConfig,
    args: Any,
    start_time: str,
    end_time: str,
):
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    backend = args.data_backend
    instruments = args.instruments or experiment.data.instruments

    if backend == "factor_engine" and instruments != "sp500":
        # 新路径优先：FactorEngineAdapter 可用时不再 monkey-patch Expression.evaluate；
        # 不可用（FE import 失败等）才 fallback 旧 enable_factor_engine_evaluation。
        try:
            from alphaprobe.integration.factor_engine_adapter import FactorEngineAdapter

            adapter = FactorEngineAdapter()
            ok, _ = adapter.validate("rank(close)", surface="daily")
            if not ok:
                enable_factor_engine_evaluation()
        except Exception:
            enable_factor_engine_evaluation()
        ds_cfg = experiment.factor_engine_data_source_config(
            start_date=start_time,
            end_date=end_time,
        )
        backtrack = recommended_max_backtrack_days(label_days=int(args.label_days))
        return FactorEngineStockData(
            instrument=instruments,
            start_time=start_time,
            end_time=end_time,
            device=device,
            max_backtrack_days=backtrack,
            max_files=experiment.data.max_files if args.fe_max_files is None else args.fe_max_files,
            data_source_config=ds_cfg,
        )

    if instruments == "sp500":
        qlib_path = os.getenv("QLIB_PATH_SP500", "PATH/TO/data/qlib_data/us_data_qlib")
    else:
        qlib_path = os.getenv("QLIB_PATH_CN", "PATH/TO/.qlib/qlib_data/cn_data")
    return StockData(
        instrument=instruments,
        start_time=start_time,
        end_time=end_time,
        qlib_path=qlib_path,
        device=device,
    )


def run_mining_campaign(
    args: Any,
    *,
    campaign_id: str | None = None,
    experiment: ExperimentConfig | None = None,
) -> dict[str, Any]:
    """执行一轮完整挖掘：冷启动 → 迭代挖掘 → disk.v1 投递 candidate_pool。"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    if experiment is None:
        experiment = ExperimentConfig.from_yaml(args.experiment_config)

    if campaign_id:
        experiment.delivery.campaign_id = campaign_id
        print(f"[mining] campaign_id={campaign_id}")

    _apply_env_llm_settings(experiment)
    _apply_mining_yaml_defaults(experiment, args)
    _init_prompts(experiment)
    _apply_checkpoint_settings(experiment, args, PROJECT_ROOT)
    print(
        f"[checkpoint] dir={args.checkpoint_dir} "
        f"enabled={args.checkpoint_enabled} resume={args.checkpoint_resume}"
    )

    mining = experiment.delivery.mining_config
    train_period = mining.get("train_period", ["2016-01-01", "2021-12-31"])
    valid_period = mining.get("valid_period", ["2022-01-01", "2023-12-31"])
    test_period = mining.get("test_period", ["2024-01-01", "2026-07-31"])
    print(
        f"[split] train={train_period[0]}..{train_period[1]} "
        f"valid={valid_period[0]}..{valid_period[1]} "
        f"test={test_period[0]}..{test_period[1]}"
    )

    data = _make_stock_data(experiment, args, train_period[0], train_period[1])
    # §3.3 Test 封存：data_test 构造保留（test_period 仍写入 config/投递 metadata），
    # 但不传入 trainer 训练路径（传 None）——search loop 不触碰 test 段。
    data_test = _make_stock_data(experiment, args, test_period[0], test_period[1])

    # Fitness 标签：vwap→vwap 远期收益（非 close→close / open→open）
    # RankIC = Spearman(factor, target)；IC = Pearson(factor, target)
    vwap = Feature(FeatureType.VWAP)
    label_days = int(args.label_days)
    target = Ref(vwap, -label_days) / vwap - 1

    initial_exprs = resolve_cold_start(experiment, args)

    pool = AlphaKnowledgePool(
        capacity=args.pool_capacity,
        stock_data=data,
        target=target,
        ic_mut_threshold=args.ic_mut_threshold,
        top_k=args.top_k,
        depth_decay=args.depth_decay,
        embedding_model_name=args.embedding_model_name,
        use_res_correlation=args.use_res_correlation,
        use_semantic_similarity=args.use_semantic_similarity,
        use_edit_distance=args.use_edit_distance,
        separate_leaf_non_leaf=args.separate_leaf_non_leaf,
        device="cuda:0",
    )
    trainer = AlphaKnowledgeTrainer(
        pool=pool,
        initial_expressions=initial_exprs,
        test_data=None,  # §3.3 Test 封存：trainer 训练路径不再接收 test 段
        target=target,
        args=args,
    )
    trainer.train(args)

    export_result: dict[str, Any] = {}
    if experiment.has_delivery_block:
        device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
        exporter = DiskV1DeliveryExporter(experiment)
        export_result = exporter.export_from_pool(pool, target, experiment, device)
        print("[delivery] candidate_pool export:", export_result)
        _set_last_pool(pool)
        _record_exported_to_memory(experiment, pool, export_result)

    return {
        "campaign_id": experiment.delivery.resolved_campaign_id(),
        "pool_size": pool.size,
        "export": export_result,
    }


def _record_exported_to_memory(
    experiment: ExperimentConfig,
    pool: Any,
    export_result: dict[str, Any],
) -> None:
    """把本轮 export 的候选因子 mark_exported 进 GlobalMemoryStore（最小 diff）。

    只在 has_delivery_block 时调用；store 打开/落库异常不阻塞主流程。
    仅处理真正 export 出的因子（manifest 存在的 formula），并预置 _LAST_POOL
    供 continuous 做全池快照 upsert（§76 跨轮 lineage 持久化）。
    """
    manifests = export_result.get("exported_manifests") or []
    if not manifests:
        return
    from alphaprobe.fe_bridge.dsl_convert import expression_to_dsl
    from alphaprobe.memory import GlobalMemoryStore
    from alphaprobe.dedup import canonical_ast_hash, canonicalize_dsl, signal_equivalence_id

    # 收集真正 export 出的 formula（manifest.json 为准）
    exported_formulas: set[str] = set()
    for m in manifests:
        try:
            payload = json.loads(Path(str(m)).read_text(encoding="utf-8"))
        except Exception:
            continue
        f = str(payload.get("formula") or "").strip()
        if f:
            exported_formulas.add(f)

    try:
        store = GlobalMemoryStore()
    except Exception as exc:
        print(f"[memory] open GlobalMemoryStore failed, skip export registry: {exc}")
        return
    try:
        campaign_id = experiment.delivery.resolved_campaign_id()
        for i in range(pool.size):
            expr = pool.exprs[i]
            if expr is None:
                continue
            try:
                formula = expression_to_dsl(expr)
            except Exception:
                continue
            if formula not in exported_formulas:
                continue
            canonical = canonicalize_dsl(formula)
            ok, existing = store.upsert_factor_node(
                factor_id=f"ap_{campaign_id}_{i}",
                canonical_formula=canonical,
                canonical_ast_hash=canonical_ast_hash(formula),
                signal_equivalence_id=signal_equivalence_id(formula),
                parameter_family_id=None,
                source_system="alphaprobe",
                source_snapshot=campaign_id,
                source_type="MINED",
                exportable=True,
            )
            fid = existing if existing is not None else f"ap_{campaign_id}_{i}"
            store.mark_exported(factor_id=fid, campaign_id=campaign_id, manifest_path="")
    except Exception as exc:
        print(f"[memory] record exported factors failed (non-blocking): {exc}")
    finally:
        store.close()


def build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AlphaPROBE 单轮挖掘")
    parser.add_argument(
        "--experiment_config",
        type=str,
        default=str(DEFAULT_EXPERIMENT),
    )
    parser.add_argument("--cold_start_library", type=str, default=None)
    parser.add_argument("--cold_start_sample_size", type=int, default=None)
    parser.add_argument("--cold_start_pv_ratio", type=float, default=None)
    parser.add_argument("--cold_start_seed", type=int, default=None)
    parser.add_argument("--campaign_id", type=str, default=None, help="覆盖 delivery.campaign_id")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", type=int, default=1)
    parser.add_argument("--instruments", type=str, default=None)
    parser.add_argument(
        "--data_backend",
        type=str,
        default=os.getenv("ALPHAPROBE_DATA_BACKEND", "factor_engine"),
        choices=("factor_engine", "qlib"),
    )
    parser.add_argument("--fe_max_files", type=int, default=None)
    parser.add_argument("--temp", type=float, default=0.5)
    parser.add_argument("--pool_capacity", type=int, default=2000)
    parser.add_argument("--log_freq", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=50)
    parser.add_argument("--depth_limit", type=int, default=7)
    parser.add_argument("--generate_num", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=15)
    parser.add_argument("--depth_decay", type=float, default=0.05)
    parser.add_argument(
        "--embedding_model_name",
        type=str,
        default=os.getenv(
            "EMBEDDING_MODEL_NAME",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ),
    )
    parser.add_argument("--use_res_correlation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_semantic_similarity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_edit_distance", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--separate_leaf_non_leaf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--search_time", type=int, default=40)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--times_decay", type=float, default=0.10)
    parser.add_argument("--start_times", type=int, default=2)
    parser.add_argument("--ic_threshold", type=float, default=0.006)
    parser.add_argument("--ic_mut_threshold", type=float, default=0.9)
    parser.add_argument("--ic_new_threshold", type=float, default=0.45)
    parser.add_argument("--icir_decay_threshold", type=float, default=0.7)
    parser.add_argument("--threshold_ric", type=float, default=0.015)
    parser.add_argument("--threshold_ricir", type=float, default=0.15)
    parser.add_argument("--n_factors", type=int, default=15)
    parser.add_argument("--chunk_size", type=int, default=180)
    parser.add_argument("--window", type=str, default="inf")
    parser.add_argument("--label_days", type=int, default=20)
    parser.add_argument("--corr_threshold", type=float, default=0.95)
    parser.add_argument("--ridge_alpha", type=float, default=1e-6)
    parser.add_argument("--use_vif", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--linear_dep_tol", type=float, default=1e-10)
    return parser
