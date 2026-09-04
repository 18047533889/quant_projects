"""AlphaPROBE 单轮挖掘入口（训练 + candidate_pool 投递）。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

# Task 1（A11）：顶层**不** import torch / dotenv / cold_start / trainer / shared
# 厚依赖——无 torch venv 下 `import alphaprobe.runner` 必须成功（legacy 路径未用时）。
# torch / dotenv / cold_start / FactorEngineStockData / qlib StockData 全部 lazy 进
# 各自使用点（_make_stock_data / resolve_cold_start / legacy 分支 / export 段）。
from alphaprobe.delivery.config import ExperimentConfig
from alphaprobe.delivery.exporter import DiskV1DeliveryExporter

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


def _apply_env_defaults() -> None:
    """进程级环境默认（原模块级 load_dotenv 副作用，Task 1 移入函数）。

    dotenv 是可选依赖（alphaprobe-dev 环境有；通用无 torch venv 无）——用
    try-import 绕行；环境变量显式设过时绝不覆盖（load_dotenv override=False
    语义；原模块级 override=True 只针对 .env 文件与 shell 残留冲突，这里保留
    setdefault 幂等）。
    """
    # 项目 .env 优先于 shell 里残留的旧 OPENAI_*（如先前其它网关）
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env", override=True)
    except Exception:  # noqa: BLE001 - dotenv 缺失时跳过（环境变量已由外层注入）
        pass
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


_apply_env_defaults()


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
    from alphaprobe.cold_start import (
        ALPHA101_DEFAULT_YAML,
        ALPHA158_DEFAULT_YAML,
        ALPHA191_DEFAULT_YAML,
    )

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
    from alphaprobe.cold_start import (
        DEFAULT_YAML,
        load_cold_start_for_training,
        load_cold_start_mixed_for_training,
        load_cold_start_multi_library_for_training,
    )

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


def _resolve_llm_mode(experiment: ExperimentConfig, args: Any) -> Any:
    """从 experiment config / args 读 execution mode（Task 1 A10）。

    优先级（Task 1）：
    1. ``--execution-mode`` CLI（build_train_parser 显式项，choices 三值）；
    2. yaml ``mining.llm.mode``（向后兼容旧字段）；
    3. ``args.llm_mode``（旧 CLI，backward compat）；
    4. 全部未配 → OFFLINE_TEST（仅 CI/默认 stub 路径允许）。

    **非法字符串 → raise（LLMConfigurationError）**，绝不静默回 OFFLINE_TEST
    （A10：production campaign 必须显式声明 mode；拼错值属配置错误应 fail）。
    """
    from alphaprobe.llm_client import ExecutionMode, LLMConfigurationError

    raw_mode = None
    # 1) 显式 CLI --execution-mode 最优先
    exec_mode = getattr(args, "execution_mode", None)
    if exec_mode is not None:
        raw_mode = exec_mode
    # 2) yaml mining.llm.mode（向后兼容）
    if raw_mode is None:
        try:
            mining_raw = dict((experiment.raw.get("mining") or {}))
            raw_mode = (mining_raw.get("llm") or {}).get("mode")
        except Exception:  # noqa: BLE001
            raw_mode = None
    # 3) 旧 CLI args.llm_mode
    if raw_mode is None:
        raw_mode = getattr(args, "llm_mode", None)
    if raw_mode is None:
        return ExecutionMode.OFFLINE_TEST
    try:
        return ExecutionMode(str(raw_mode).strip().lower())
    except ValueError as exc:
        raise LLMConfigurationError(
            f"invalid execution mode {raw_mode!r} (A10: 绝不静默回 OFFLINE_TEST). "
            f"合法值: {[m.value for m in ExecutionMode]}"
        ) from exc


def _resolve_llm_client(experiment: ExperimentConfig, args: Any) -> Any:
    """从 experiment config / args 装配真实 LLM client（P0-A）。

    真实接入点当前在 runner 外层注入（``args.llm_client``）；yaml/环境变量
    provider 装配留 P1（真实 OpenAI-compatible client 需 network）。本函数
    OFFLINE_TEST 下返回 None（由 resolve_llm_fn 落 stub）。
    """
    return getattr(args, "llm_client", None)


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
    # Task 1（A11）：torch / FE bootstrap / FE StockData / qlib StockData 全部 lazy。
    # 无 torch venv 下此函数（真实数据路径）才需要 torch；模块顶层绝不 import。
    import torch

    from alphaprobe.fe_bridge.bootstrap import enable_factor_engine_evaluation
    from alphaprobe.fe_bridge.stock_data import FactorEngineStockData
    from alphaprobe.cold_start import recommended_max_backtrack_days

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
    from shared.alphagen_qlib.stock_data import StockData

    return StockData(
        instrument=instruments,
        start_time=start_time,
        end_time=end_time,
        qlib_path=qlib_path,
        device=device,
    )


def _run_pipeline_mining(
    args: Any,
    experiment: ExperimentConfig,
    data: Any,
    *,
    campaign_id: str,
    parents: list[dict[str, Any]] | None = None,
    data_search_valid: Any = None,
    execution_mode: Any = None,
) -> dict[str, Any]:
    """--pipeline new 主链：SearchPipeline.run_round 循环 + disk.v1 导出。

    沿用现有 quota/stop 逻辑（search_time 轮次上限）；全程不触 test 段。

    Task 1（A2）：SearchPipeline 显式接收 train / search_valid 两段 provider
    （``data_train`` / ``data_search_valid``）。L2 只吃 Train、L3 只吃 SearchValid。
    ``execution_mode`` 由 run_mining_campaign 解析后传入（本函数不再自解析，
    保持单一入口）。
    """
    from alphaprobe.llm_client import ExecutionMode, resolve_llm_fn
    from alphaprobe.pipeline import PipelineConfig, SearchPipeline

    llm_mode = execution_mode if execution_mode is not None else _resolve_llm_mode(experiment, args)
    config = PipelineConfig(
        pool_target=max(16, args.pool_capacity),
        pool_max=max(32, args.pool_capacity * 2),
        budget_per_round=max(4, args.generate_num),
        max_rounds=int(args.search_time),
        # P0-A：生产语义默认结构化 generation（runner 主链即生产主链）。
        structured_generation=True,
    )
    llm_client = getattr(args, "llm_client", None)
    if llm_client is None:
        # 从 experiment config 读 llm client（真实接入点在 runner 外层注入；
        # yaml 侧暂不装配真实 client——OFFLINE_TEST 默认 stub）。
        llm_client = _resolve_llm_client(experiment, args)
    pipeline_kwargs: dict[str, Any] = dict(
        experiment=experiment,
        data_train=data,
        config=config,
        memory_store=None,
        mode=llm_mode,
        data_search_valid=data_search_valid,
    )
    if llm_mode in (ExecutionMode.PRODUCTION, ExecutionMode.RESEARCH_DEGRADED):
        # P0-A：真实 LLM client 由外层注入（args.llm_client）。未注入即 fail-closed
        # 抛（绝不静默切 stub）。llm_fn 显式传 None 也走 resolve_llm_fn 守卫。
        pipeline_kwargs["llm_client"] = llm_client
        pipeline_kwargs["llm_fn"] = None
    else:
        # OFFLINE_TEST：默认确定性 stub（保持现状行为）。resolve_llm_fn 在
        # llm_client 为 None 时落 DeterministicStubLLMClient；llm_fn 在
        # __post_init__ 里非 None 优先于 stub 默认，行为与旧版一致。
        llm_fn = resolve_llm_fn(llm_client, ExecutionMode.OFFLINE_TEST, structured=True)
        pipeline_kwargs["llm_fn"] = llm_fn
    pipeline = SearchPipeline(**pipeline_kwargs)
    round_result = None
    for _round in range(int(args.search_time)):
        round_result = pipeline.run_round(
            round_id=f"round_{_round + 1}",
            parents=parents,
        )
        if pipeline.pool_size() >= config.pool_target:
            break
    print(
        "[pipeline] rounds done: "
        f"pool={pipeline.pool_size()} last_round={round_result}"
    )
    return {
        "campaign_id": campaign_id,
        "pool_size": pipeline.pool_size(),
        "round_result": round_result,
        "pipeline": pipeline,
        "execution_mode": llm_mode.value,
        "degraded_reasons": list(getattr(pipeline, "degraded_reasons", []) or []),
    }


def _exportable_pool_from_pipeline(pipeline: Any) -> Any:
    """把 SearchPipeline 的 ActivePool 映射为 export_from_pool 可消费的池对象。

    export_from_pool 需要 .exprs / .size / .topics / .descriptions / .single_ics /
    .icir（AlphaKnowledgePool 契约）。ActivePool 是 Pareto+QD 结构，这里构造一个
    最小适配对象，用 canonical_formula 作 exprs，其余字段置空。
    """
    from alphaprobe.pool import ActivePool

    pool = getattr(pipeline, "pool", None)
    if not isinstance(pool, ActivePool) or len(pool.members) == 0:
        return None
    snapshot = pool.snapshot()

    class _ExportPool:
        def __init__(self, members: list[dict[str, Any]]) -> None:
            self.exprs: list[Any] = [m["formula"] for m in members]
            self.size = len(self.exprs)
            self.topics: list[str] = ["" for _ in self.exprs]
            self.descriptions: list[str] = [f"alpha-probe factor {m['factor_id']}" for m in members]
            self.single_ics: list[float] = [m.get("fitness", 0.0) for m in members]
            self.icir: list[float] = [0.0 for _ in self.exprs]

    return _ExportPool(snapshot)


def _resolve_initial_parents(
    experiment: ExperimentConfig,
    args: Any,
    *,
    exec_mode: Any,
    project_root: Path = PROJECT_ROOT,
) -> list[dict[str, Any]]:
    """构造搜索首轮的 parents（Task 20 SeedProvider 接线）。

    优先级：
    1. 显式 ``--seed-provider-fa-lib``（factor_assets 全局库路径）→ 用
       SeedProvider 从该库 metadata 采样 K 个 seed（**只读 metadata，不拉因子
       值面板**；选中 seed 才转 parents，10 万冷启动不整体物化进 DAG）；
    2. 未显式指定 → OFFLINE_TEST 兼容开关（默认开）：走既有
       ``resolve_cold_start`` 位置序路径（YAML seed 全量转 parents，行为与
       Task 1 完全一致）。

    开关（Ablation，Non-negotiable #30）：
    - ``--seed-provider`` ``on|off``：显式启用/禁用 SeedProvider 路径；
    - ``--seed-provider-fa-lib PATH``：factor_assets 全局库的 metadata 源路径
      （目录内 catalog.json 或 sqlite）。缺省且 seed_provider 开启时回落
      resolve_cold_start（无全局库时绝不停摆）。
    - ``--seed-sample-size K``：SeedProvider 采样数（缺省 = generate_num*4）。
    """
    from alphaprobe.llm_client import ExecutionMode

    mode_on = getattr(args, "seed_provider", None)
    fa_lib = getattr(args, "seed_provider_fa_lib", None)
    # 显式开关决定是否走 SeedProvider；未显式给 on/off 时默认兼容路径
    want_seed_provider = bool(
        fa_lib or (mode_on is not None and str(mode_on).strip().lower() == "on")
    )
    if not want_seed_provider:
        return _parents_from_cold_start(experiment, args)

    try:
        from alphaprobe.seed_provider import SeedProvider, SeedProviderConfig
    except Exception as exc:  # noqa: BLE001 - SeedProvider 不可用 → 兼容路径
        print(f"[seed] SeedProvider unavailable, fallback cold-start: {exc}")
        return _parents_from_cold_start(experiment, args)

    try:
        provider = SeedProvider(
            config=SeedProviderConfig(
                rng_seed=int(getattr(args, "seed", 0) or 0),
            )
        )
        catalog = _open_factor_assets_catalog(fa_lib)
        if catalog is None:
            print("[seed] FactorAssets global library unavailable, fallback cold-start")
            return _parents_from_cold_start(experiment, args)
        sample_size = getattr(args, "seed_sample_size", None)
        if sample_size is None:
            sample_size = max(4, int(getattr(args, "generate_num", 5)) * 4)
        seeds = provider.sample(catalog, k=int(sample_size))
        diag = provider.diagnostics.to_dict()
        print(
            f"[seed] SeedProvider sampled {len(seeds)} seeds from FactorAssets "
            f"library (requested={sample_size}, degraded={diag.get('degraded_buckets')})"
        )
        if not seeds:
            return _parents_from_cold_start(experiment, args)
        return [s.to_parent_dict() for s in seeds]
    except Exception as exc:  # noqa: BLE001 - 采样失败不阻塞主链
        print(f"[seed] SeedProvider sampling failed, fallback cold-start: {exc}")
        return _parents_from_cold_start(experiment, args)


def _parents_from_cold_start(
    experiment: ExperimentConfig, args: Any,
) -> list[dict[str, Any]]:
    """既有 OFFLINE_TEST 兼容路径：resolve_cold_start 位置序全量转 parents。

    不截断到 5（Task 1 行为：上限放宽到 generate_num*4，至少覆盖全部 seed）。
    """
    initial_exprs = resolve_cold_start(experiment, args)
    parent_cap = max(4, int(getattr(args, "generate_num", 5)) * 4)
    parents: list[dict[str, Any]] = []
    for i, expr in enumerate(initial_exprs[: min(parent_cap, len(initial_exprs))]):
        parents.append(
            {
                "formula": str(expr) if not hasattr(expr, "dsl") else getattr(expr, "dsl"),
                "factor_id": f"seed_{i}",
                "fitness": 0.0,
            }
        )
    return parents


def _open_factor_assets_catalog(
    fa_lib: Any,
) -> list[dict[str, Any]] | None:
    """打开 factor_assets 全局库的 metadata catalog（SeedProvider 候选源）。

    fa_lib 可以是：
    - 目录路径：读取其下 ``seed_catalog.json``（list[dict]，含 factor_id /
      canonical_repr / canonical_hash / fe_identity_ref / parameter_family_id /
      bucket）；
    - sqlite 路径：读取 assets 表（factor_assets registry 的 SQLite 库），只取
      metadata 字段（不拉因子值面板）。

    打开失败返回 None（调用方回落 resolve_cold_start）。零网络零模型。
    """
    import json
    import os

    if fa_lib is None:
        return None
    p = os.fspath(fa_lib) if not hasattr(fa_lib, "__fspath__") else str(fa_lib)
    from pathlib import Path as _P

    path = _P(p)
    if path.is_dir():
        cand = path / "seed_catalog.json"
        if cand.exists():
            try:
                data = json.loads(cand.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return [dict(m) for m in data if isinstance(m, dict)]
                if isinstance(data, dict) and isinstance(data.get("seeds"), list):
                    return [dict(m) for m in data["seeds"] if isinstance(m, dict)]
            except (OSError, json.JSONDecodeError) as exc:
                print(f"[seed] catalog unreadable: {cand}: {exc}")
                return None
        return None
    if path.is_file():
        try:
            import sqlite3

            conn = sqlite3.connect(str(path))
            try:
                rows = conn.execute(
                    "SELECT payload FROM assets ORDER BY factor_id LIMIT 500000"
                ).fetchall()
            except sqlite3.Error as exc:
                print(f"[seed] sqlite assets table unreadable: {exc}")
                return None
            finally:
                conn.close()
            if not rows:
                return None
            out: list[dict[str, Any]] = []
            for (payload,) in rows:
                try:
                    obj = json.loads(payload)
                except (json.JSONDecodeError, TypeError):
                    continue
                md = obj.get("metadata") or obj
                meta = {
                    "factor_id": str(md.get("factor_id") or ""),
                    "canonical_repr": str(md.get("canonical_repr") or md.get("canonical_formula") or ""),
                    "canonical_hash": str(md.get("canonical_hash") or ""),
                    "fe_identity_ref": str(md.get("fe_identity_ref") or md.get("signal_equivalence_id") or ""),
                    "parameter_family_id": md.get("parameter_family_id"),
                }
                # bucket：从 payload schema_json / 顶层 tags 推断，缺省 None
                # （SeedProvider 把缺省归 RANDOM 桶，随机探索兜底）
                if meta["factor_id"] and meta["canonical_repr"]:
                    out.append(meta)
            return out
        except Exception as exc:  # noqa: BLE001 - sqlite 打开失败
            print(f"[seed] FactorAssets sqlite catalog open failed: {exc}")
            return None
    print(f"[seed] FactorAssets library path not found: {path}")
    return None


def run_mining_campaign(
    args: Any,
    *,
    campaign_id: str | None = None,
    experiment: ExperimentConfig | None = None,
) -> dict[str, Any]:
    """执行一轮完整挖掘：冷启动 → 迭代挖掘 → disk.v1 投递 candidate_pool。

    Task 1（A2/A10）：按 ExperimentConfig split 构造 Train / SearchValid provider。
    - OFFLINE_TEST：默认 stub 路径，search_valid provider 不构造（保持离线行为）；
    - PRODUCTION / RESEARCH_DEGRADED：真构造 valid 段 provider（QE 权威 valid 指标
      需要独立 valid 段 stock_data）——构造失败 fail-closed raise。
    """
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

    from alphaprobe.llm_client import ExecutionMode

    exec_mode = _resolve_llm_mode(experiment, args)
    print(f"[mining] execution_mode={exec_mode.value}")

    data = _make_stock_data(experiment, args, train_period[0], train_period[1])
    # §3.3 Test 封存：搜索期绝不构造 test 段数据（L0-L4 Test Data Zero Touch）。
    # test_period 仅写入 config / 投递 metadata（_make_stock_data 不再为 test 段调用）。

    # Task 1（A2）：SearchValid provider。仅生产/研究模式真构造（QE 权威 valid 段
    # 是 L3_search_valid 指标唯一来源）；OFFLINE_TEST 保持 None（离线 stub 行为）。
    data_search_valid = None
    if exec_mode in (ExecutionMode.PRODUCTION, ExecutionMode.RESEARCH_DEGRADED):
        try:
            data_search_valid = _make_stock_data(
                experiment, args, valid_period[0], valid_period[1]
            )
        except Exception as exc:  # noqa: BLE001 - fail-closed：生产模式 valid 段必须可构造
            from alphaprobe.pipeline import PipelineAuthorityError

            raise PipelineAuthorityError(
                f"{exec_mode.value} requires search_valid provider "
                f"({valid_period[0]}..{valid_period[1]}); construction failed: {exc}"
            ) from exc

    # Fitness 标签：vwap→vwap 远期收益（非 close→close / open→open）
    # RankIC = Spearman(factor, target)；IC = Pearson(factor, target)
    from shared.alphagen.data.expression import Feature, Ref
    from shared.alphagen_qlib.stock_data import FeatureType

    vwap = Feature(FeatureType.VWAP)
    label_days = int(args.label_days)
    target = Ref(vwap, -label_days) / vwap - 1

    if getattr(args, "pipeline", "new") == "new":
        # 默认主链：SearchPipeline（新架构真身），不走 AlphaKnowledgeTrainer。
        # Task 20（SeedProvider）：parents 的来源从「YAML 位置序全量」升级为
        # 「SeedProvider 从 FactorAssets 全局库按配置比例采样」。未启用
        # FactorAssets 全局库（OFFLINE_TEST / 无 factor_assets 库 / 用户显式
        # 关）时保持既有 OFFLINE_TEST 兼容路径（resolve_cold_start 位置序）。
        parents = _resolve_initial_parents(
            experiment,
            args,
            exec_mode=exec_mode,
            project_root=PROJECT_ROOT,
        )
        campaign_id = experiment.delivery.resolved_campaign_id()
        mining_result = _run_pipeline_mining(
            args,
            experiment,
            data,
            campaign_id=campaign_id,
            parents=parents,
            data_search_valid=data_search_valid,
            execution_mode=exec_mode,
        )
        export_result: dict[str, Any] = {}
        if experiment.has_delivery_block:
            # 新链导出走 DiskV1DeliveryExporter（pool 对象 → export_from_pool）。
            # 兼容池契约：构造 ActivePool 兼容的最小映射对象（exprs/topics/size）。
            # 注：SearchPipeline.pool 是 alphaprobe.pool.ActivePool（Pareto+QD），
            # export_from_pool 需要 .exprs —— 用 pool.snapshot() 构造导出源。
            pipeline = mining_result.get("pipeline")
            pool = _exportable_pool_from_pipeline(pipeline) if pipeline is not None else None
            if pool is not None:
                import torch

                device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
                exporter = DiskV1DeliveryExporter(experiment)
                export_result = exporter.export_from_pool(pool, target, experiment, device)
                print("[delivery] candidate_pool export:", export_result)
                _set_last_pool(pool)
                _record_exported_to_memory(experiment, pool, export_result)
        return {
            "campaign_id": campaign_id,
            "pool_size": mining_result["pool_size"],
            "execution_mode": exec_mode.value,
            "export": export_result,
        }

    initial_exprs = resolve_cold_start(experiment, args)

    from alphaprobe.trainer.pool import AlphaKnowledgePool
    from alphaprobe.trainer.trainer import AlphaKnowledgeTrainer

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
        import torch

        device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
        exporter = DiskV1DeliveryExporter(experiment)
        export_result = exporter.export_from_pool(pool, target, experiment, device)
        print("[delivery] candidate_pool export:", export_result)
        _set_last_pool(pool)
        _record_exported_to_memory(experiment, pool, export_result)

    return {
        "campaign_id": experiment.delivery.resolved_campaign_id(),
        "pool_size": pool.size,
        "execution_mode": exec_mode.value,
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
    from alphaprobe.memory import GlobalMemoryStore
    from alphaprobe.authority import FactorIdentityAuthorityError, build_identity_view

    # 收集真正 export 出的 formula（manifest.json 为准）
    exported_formulas: set[str] = set()
    for m in manifests:
        try:
            payload = json.loads(Path(str(m)).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            # fail-fast：清单不可读/坏 JSON 记日志，绝不吞 NameError 类编程错误
            print(f"[memory] manifest unreadable, skip: {m!r}: {exc}")
            continue
        f = str(payload.get("formula") or "").strip()
        if f:
            exported_formulas.add(f)

    # formula 抽取：池条目是 FE DSL 字符串（new 主链）或 alphagen Expression
    # （legacy 主链）。Task 1 后 new 主链池条目直接存 DSL 文本，不再需要
    # expression_to_dsl（该转换 import 会拉 shared/alphagen → torch，破坏无
    # torch 环境）；legacy 主链的 Expression 对象用其 __str__ 源文本即可
    # （fe_bridge.dsl_expression 的 __str__ 返回源 DSL）。
    def _formula_of(expr: Any) -> str | None:
        if expr is None:
            return None
        if isinstance(expr, str):
            return expr
        dsl = getattr(expr, "dsl", None)
        if dsl:
            return str(dsl)
        s = str(expr)
        return s or None

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
            formula = _formula_of(expr)
            if not formula:
                continue
            if formula not in exported_formulas:
                continue
            # FE 权威身份（§28）；FE 不可用 fail-closed 跳过，绝不回落文本 regex。
            try:
                iv = build_identity_view(formula)
            except FactorIdentityAuthorityError as exc:
                print(f"[memory] authority identity unavailable, skip: {formula!r}: {exc}")
                continue
            canonical = str(iv.get("canonical_formula") or formula)
            ok, existing = store.upsert_factor_node(
                factor_id=f"ap_{campaign_id}_{i}",
                canonical_formula=canonical,
                canonical_ast_hash=str(iv.get("canonical_ast_hash", "") or ""),
                signal_equivalence_id=str(iv.get("signal_equivalence_id", "") or ""),
                parameter_family_id=str(iv.get("parameter_family_id", "") or None),
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
    parser.add_argument(
        "--seed-provider",
        type=str,
        default=None,
        choices=("on", "off"),
        help="Task 20：是否用 SeedProvider（从 FactorAssets 全局库采样 seed）"
        "替代 YAML 位置序 parents。缺省 None → 兼容路径（resolve_cold_start）。",
    )
    parser.add_argument(
        "--seed-provider-fa-lib",
        type=str,
        default=None,
        help="Task 20：factor_assets 全局库路径（目录 seed_catalog.json 或 "
        "SQLite assets 表）；SeedProvider 只读 metadata 采样，不物化全库。",
    )
    parser.add_argument(
        "--seed-sample-size",
        type=int,
        default=None,
        help="Task 20：SeedProvider 采样 seed 数；缺省 = generate_num*4。",
    )
    parser.add_argument("--campaign_id", type=str, default=None, help="覆盖 delivery.campaign_id")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", type=int, default=1)
    parser.add_argument(
        "--pipeline",
        type=str,
        default="new",
        choices=("legacy", "new"),
        help="搜索主链：new=SearchPipeline（默认，Test 零构造）；legacy=AlphaKnowledgeTrainer",
    )
    parser.add_argument(
        "--execution-mode",
        type=str,
        default=None,
        choices=("production", "research_degraded", "offline_test"),
        help="Task 1（A10）：运行模式显式声明。缺省读 yaml mining.llm.mode / 旧 "
        "--llm_mode；全部未配 → offline_test（仅 CI/默认 stub 路径）。非法值报错，"
        "绝不静默回 offline_test。",
    )
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
