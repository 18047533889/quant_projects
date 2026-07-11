"""
AutoFactorEvaluation 全流程管道

架构:
    candidate_pool/candidate/
        ↓ [Gateway Worker Pool]
    tier0/gateway_temp/
        ↓ [Router]
    tier1/gateway_pass_base/  (及其他目标目录)
        ↓ [Assetization Worker Pool]
    tier0/assetization_temp/
        ↓ [Assetization Router]
    tier1/assetization_raw_factor_base/{factor_id}/
        ↓ [Purification Worker Pool]
    tier0/purification_temp/
        ↓ [Purification Router]
    tier1/purification_pure_factor_base/{factor_id}/
        ↓ [Evaluation Worker Pool]
    tier0/evaluation_temp/
        ↓ [Evaluation Router]
    tier2/... tier3/... tier4/...  (根据 route_recommendation)

使用方式:
    python -m pipeline                         # 全流程
    python -m pipeline --gateway-only          # 仅 Gateway 审查 + 路由
    python -m pipeline --assetization-only     # 仅 Assetization 计算 + 路由
    python -m pipeline --purification-only     # 仅 Purification 纯化 + 路由
    python -m pipeline --evaluation-only       # 仅 Evaluation 评估 + 路由
    python -m pipeline --all                   # 全流程（Gateway → Assetization → Purification → Evaluation）
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from multiprocessing import Process, Queue, cpu_count
from pathlib import Path
from typing import Any, Optional

# ---- 日志初始化 ----
_PIPELINE_LOGGER_SETUP: bool = False


def _require_dir(path: Path, name: str = ""):
    """验证目录存在，不存在则自动创建。"""
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise NotADirectoryError(f"路径不是目录: {path} ({name})")


def _setup_logging(log_dir: str | Path | None = None):
    """初始化流水线日志系统（File + Console），仅执行一次。"""
    global _PIPELINE_LOGGER_SETUP
    if _PIPELINE_LOGGER_SETUP:
        return
    _PIPELINE_LOGGER_SETUP = True

    _log_dir: Path
    if log_dir:
        _log_dir = Path(log_dir)
    else:
        try:
            from config_manager import ConfigManager
            _cfg = ConfigManager()
            _log_dir = _cfg.logging_directory
        except Exception:
            raise RuntimeError(
                "无法初始化日志：ConfigManager 加载失败且未指定 log_dir"
            )

    _pipeline_log_dir = _log_dir / "pipeline"
    _require_dir(_pipeline_log_dir, "pipeline 日志目录")

    _fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    _log_path = _pipeline_log_dir / f"pipeline_{datetime.now():%Y%m%d}.log"

    _fh = logging.FileHandler(_log_path, encoding="utf-8")
    _fh.setFormatter(_fmt)

    _root = logging.getLogger()
    _root.setLevel(logging.INFO)
    _root.addHandler(_fh)

    # 确保 StreamHandler 也存在
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in _root.handlers):
        _sh = logging.StreamHandler(sys.stdout)
        _sh.setFormatter(_fmt)
        _root.addHandler(_sh)


def _setup_worker_logging(stage: str, worker_id: int, log_dir: str | Path):
    """为单个 Worker 进程初始化文件日志。"""
    _log_dir = Path(log_dir) / stage
    _require_dir(_log_dir, f"Worker 日志目录 ({stage})")
    _log_path = _log_dir / f"worker_{worker_id:03d}.log"

    _fmt = logging.Formatter(f"%(asctime)s | {stage.upper()}{worker_id:03d} | %(message)s")
    _fh = logging.FileHandler(_log_path, encoding="utf-8")
    _fh.setFormatter(_fmt)

    _root = logging.getLogger()
    _root.setLevel(logging.INFO)
    _root.addHandler(_fh)

    # Worker 也保持 stdout 输出
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in _root.handlers):
        _sh = logging.StreamHandler(sys.stdout)
        _sh.setFormatter(_fmt)
        _root.addHandler(_sh)


logger = logging.getLogger("pipeline")

_PROJECT_ROOT = Path(__file__).resolve().parent

# ---- 配置总线：所有路径来自外部 config.yaml ----
_CONFIG: Optional[Any] = None


def _get_config() -> Any:
    """延迟加载 ConfigManager 单例。"""
    global _CONFIG
    if _CONFIG is None:
        from config_manager import ConfigManager
        _CONFIG = ConfigManager()
    return _CONFIG


# ============================================================
# Gateway 工作进程
# ============================================================

def _gateway_worker(
    task_queue: Queue,
    result_queue: Queue,
    worker_id: int,
    gateway_config_path: str | None = None,
    market_data_path: str | None = None,
):
    """Gateway 审查工作进程。"""
    import os, sys as _sys
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if _script_dir not in _sys.path:
        _sys.path.insert(0, _script_dir)

    from gateway.config import GatewayConfig
    from gateway.gateway_core import GatewayCore

    cfg = GatewayConfig(config_dir=Path(gateway_config_path) if gateway_config_path else None)
    gw = GatewayCore(cfg, market_data_path=market_data_path)
    _setup_worker_logging("gateway", worker_id, _get_config().logging_directory)
    _lg = logging.getLogger("gateway.worker")
    _lg.info("Worker %d 启动", worker_id)

    while True:
        task = task_queue.get()
        if task is None:
            _lg.info("Worker %d 停止", worker_id)
            break
        factor_dir, campaign_config = task
        try:
            result, temp_dir = gw.process_factor_dir(factor_dir, campaign_config=campaign_config)
            result_queue.put({
                "status": "done",
                "candidate_id": Path(factor_dir).name,
                "label": result.label.value,
                "temp_dir": str(temp_dir),
                "worker_id": worker_id,
            })
        except Exception as e:
            result_queue.put({
                "status": "error",
                "candidate_id": Path(factor_dir).name,
                "error": str(e),
                "worker_id": worker_id,
            })


# ============================================================
# 候选池扫描
# ============================================================

def _discover_tasks(pool_dir: Path) -> list[tuple[Path, dict]]:
    """扫描 candidate_pool/candidate，发现因子目录。"""
    tasks = []
    if not pool_dir.exists():
        return tasks
    for campaign_dir in sorted(pool_dir.iterdir()):
        if not campaign_dir.is_dir():
            continue
        cfg_path = campaign_dir / "config.json"
        campaign_config = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        for fd in sorted(campaign_dir.iterdir()):
            if fd.is_dir() and (fd / "manifest.json").exists():
                tasks.append((fd, campaign_config))
    return tasks


# ============================================================
# Gateway 多进程审查
# ============================================================

def run_gateway_pipeline(
    n_workers: int = 0,
    candidate_pool: str | Path | None = None,
    market_data_path: str | Path | None = None,
    gateway_config_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Gateway 多进程审查管道。

    扫描 candidate_pool/candidate → 多进程审查 → afv.json → tier0/gateway_temp

    Args:
        n_workers: 进程数，0=CPU核数-1。
        candidate_pool: 候选因子池路径（candidate 子目录）。
        market_data_path: 行情路径（数据质量检测用）。
        gateway_config_dir: Gateway 配置目录。

    Returns:
        处理统计。
    """
    _cfg = _get_config()
    pool_dir = Path(candidate_pool) if candidate_pool else _cfg.path("candidate")
    temp_base = _cfg.path("gateway_temp")
    cache_dir = _cfg.path("gateway_report_cache")
    n_workers = n_workers or max(1, cpu_count() - 1)
    temp_base.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Gateway 多进程审查启动: workers=%d pool=%s", n_workers, pool_dir)

    tasks = _discover_tasks(pool_dir)
    if not tasks:
        logger.info("candidate_pool 中无待处理因子")
        return {"status": "idle", "total": 0}

    logger.info("发现 %d 个待处理因子", len(tasks))
    task_queue: Queue = Queue()
    result_queue: Queue = Queue()
    workers = []

    for i in range(n_workers):
        p = Process(
            target=_gateway_worker,
            args=(
                task_queue, result_queue, i + 1,
                str(gateway_config_dir) if gateway_config_dir else None,
                str(market_data_path) if market_data_path else None,
            ),
        )
        p.start()
        workers.append(p)

    for t in tasks:
        task_queue.put(t)

    results = []
    for _ in range(len(tasks)):
        r = result_queue.get()
        results.append(r)
        logger.info("%s %s (%s)", "✅" if r["status"] == "done" else "❌",
                     r["candidate_id"], r.get("label", "?"))

    for _ in workers:
        task_queue.put(None)
    for p in workers:
        p.join(timeout=5)

    done = sum(1 for r in results if r["status"] == "done")
    # 空池归档：所有因子被抽离后保留 config.json 并转移至 archive 路径
    _archive_empty_campaigns(pool_dir, _get_config().path("archive"))
    logger.info("Gateway 完成: %d/%d 成功", done, len(tasks))
    return {"total": len(tasks), "done": done, "results": results}


# ============================================================
# 路由
# ============================================================

def run_router(
    temp_base: str | Path | None = None,
    db_root: str | Path | None = None,
) -> list[dict]:
    """路由：将 tier0/gateway_temp 中的因子分发到目标目录。"""
    from gateway.router import FactorRouter

    _cfg = _get_config()
    temp_base = Path(temp_base) if temp_base else _cfg.path("gateway_temp")
    db_root = Path(db_root) if db_root else _cfg.database_root
    router = FactorRouter(temp_base, db_root)
    records = router.scan_and_route()
    for r in records:
        logger.info("路由: %s → %s (%s)", r["candidate_id"], r["label"],
                     Path(r["target"]).parent.name)
    return records


# ============================================================
# Assetization Worker
# ============================================================

def _assetization_worker(task_queue: Queue, result_queue: Queue, worker_id: int,
                          market_data_path: str | None = None):
    """Assetization 工作进程。"""
    import os, sys as _sys
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if _script_dir not in _sys.path:
        _sys.path.insert(0, _script_dir)

    from assetization.scripts.worker import process_factor_dir as pfd

    _cfg = _get_config()
    _assn_temp = _cfg.path("assetization_temp")
    _assn_output = _cfg.path("assetization_raw_factor_base")
    _mkt_path = market_data_path or str(_cfg.market_data_path)
    _setup_worker_logging("assetization", worker_id, _cfg.logging_directory)
    _lg = logging.getLogger("assetization.worker")
    _lg.info("Assetization Worker %d 启动", worker_id)

    while True:
        task = task_queue.get()
        if task is None:
            _lg.info("Worker %d 停止", worker_id)
            break

        factor_dir, _ = task
        try:
            factor_id, temp_dir = pfd(
                factor_dir,
                market_data_path=_mkt_path,
                output_base=_assn_output,
                temp_base=_assn_temp,
            )
            result_queue.put({
                "status": "done",
                "factor_id": factor_id,
                "candidate_id": Path(factor_dir).name,
                "temp_dir": temp_dir,
                "worker_id": worker_id,
            })
        except Exception as e:
            result_queue.put({
                "status": "error",
                "candidate_id": Path(factor_dir).name,
                "error": str(e),
                "worker_id": worker_id,
            })


def _discover_gateway_pass(pool_dir: Path) -> list[Path]:
    """扫描 gateway_pass_base，发现待资产化的因子。"""
    factors = []
    if not pool_dir.exists():
        return factors
    for fd in sorted(pool_dir.iterdir()):
        if fd.is_dir() and (fd / "afv.json").exists():
            afv = json.loads((fd / "afv.json").read_text())
            gw = afv.get("Gateway", {})
            if gw.get("Label") == "Pass" and "factor_id" not in afv:
                factors.append(fd)
    return factors


# ============================================================
# Assetization 多进程管道
# ============================================================

def run_assetization_pipeline(
    n_workers: int = 0,
    gateway_pass_base: str | Path | None = None,
    market_data_path: str | Path | None = None,
) -> dict[str, Any]:
    """Assetization 多进程计算管道。

    扫描 tier1/gateway_pass_base → 多进程因子计算 → data/（按日分片 parquet）→ tier0/assetization_temp
    """
    _cfg = _get_config()
    pass_dir = Path(gateway_pass_base) if gateway_pass_base else _cfg.path("gateway_pass_base")
    assn_temp = _cfg.path("assetization_temp")
    _require_dir(assn_temp, "assetization_temp")
    n_workers = n_workers or max(1, cpu_count() - 1)

    logger.info("Assetization 多进程启动: workers=%d input=%s", n_workers, pass_dir)

    tasks = _discover_gateway_pass(pass_dir)
    if not tasks:
        logger.info("gateway_pass_base 中无待资产化因子")
        return {"status": "idle", "total": 0}

    logger.info("发现 %d 个待资产化因子", len(tasks))
    task_queue: Queue = Queue()
    result_queue: Queue = Queue()

    # 从配置读取 market_data_path
    mkt_path = str(market_data_path) if market_data_path else str(_cfg.market_data_path)

    workers = []
    for i in range(n_workers):
        p = Process(
            target=_assetization_worker,
            args=(task_queue, result_queue, i + 1, mkt_path),
        )
        p.start()
        workers.append(p)

    for t in tasks:
        task_queue.put((t, mkt_path))

    results = []
    for _ in range(len(tasks)):
        r = result_queue.get()
        results.append(r)
        logger.info("%s %s → factor_id=%s",
                     "✅" if r["status"] == "done" else "❌",
                     r["candidate_id"], r.get("factor_id", "?"))

    for _ in workers:
        task_queue.put(None)
    for p in workers:
        p.join(timeout=10)

    done = sum(1 for r in results if r["status"] == "done")
    logger.info("Assetization 完成: %d/%d 成功", done, len(tasks))
    return {"total": len(tasks), "done": done, "results": results}


# ============================================================
# Asssetization 路由
# ============================================================

def run_assetization_router() -> list[dict]:
    """路由：将 tier0/assetization_temp 中的因子分发到目标目录。"""
    from assetization.scripts.router import AssetizationRouter

    _cfg = _get_config()
    router = AssetizationRouter(
        _cfg.path("assetization_temp"),
        _cfg.path("assetization_raw_factor_base"),
    )
    records = router.scan_and_route()
    for r in records:
        logger.info("资产化路由: %s → %s (%s)", r["candidate_id"], r["factor_id"], r["label"])
    return records


# ============================================================
# Purification Worker
# ============================================================


def _purification_worker(
    task_queue: Queue,
    result_queue: Queue,
    worker_id: int,
):
    """Purification 纯化工作进程。"""
    import os, sys as _sys
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if _script_dir not in _sys.path:
        _sys.path.insert(0, _script_dir)

    from purification.scripts.worker import process_factor_dir as pfd
    from purification.config import PurificationConfig

    cfg = PurificationConfig(config_dir=_get_config().config_dir)
    _setup_worker_logging("purification", worker_id, _get_config().logging_directory)
    _lg = logging.getLogger("purification.worker")
    _lg.info("Purification Worker %d 启动", worker_id)

    while True:
        task = task_queue.get()
        if task is None:
            _lg.info("Worker %d 停止", worker_id)
            break

        factor_dir = task
        try:
            factor_id, temp_dir = pfd(
                factor_dir,
                config=cfg,
                temp_base=str(_get_config().path("purification_temp")),
            )
            result_queue.put({
                "status": "done",
                "factor_id": factor_id,
                "candidate_id": Path(factor_dir).name,
                "temp_dir": temp_dir,
                "worker_id": worker_id,
            })
        except Exception as e:
            result_queue.put({
                "status": "error",
                "candidate_id": Path(factor_dir).name,
                "error": str(e),
                "worker_id": worker_id,
            })


def _discover_assetization_output(pool_dir: Path) -> list[Path]:
    """扫描 assetization_raw_factor_base，发现待纯化的因子。

    判断标准：目录中有 afv.json 且尚未包含 Purification 段。
    """
    factors = []
    if not pool_dir.exists():
        return factors
    for fd in sorted(pool_dir.iterdir()):
        if not fd.is_dir():
            continue
        afv_path = fd / "afv.json"
        if not afv_path.exists():
            continue
        try:
            afv = json.loads(afv_path.read_text())
            if "Purification" not in afv:
                factors.append(fd)
        except Exception:
            continue
    return factors


# ============================================================
# Purification 多进程管道
# ============================================================

def run_purification_pipeline(
    n_workers: int = 0,
    raw_factor_base: str | Path | None = None,
) -> dict[str, Any]:
    """Purification 多进程纯化管道。

    扫描 tier1/assetization_raw_factor_base → 多进程纯化 → data/（按日分片）→ tier0/purification_temp
    """
    _cfg = _get_config()
    raw_base = Path(raw_factor_base) if raw_factor_base else _cfg.path("assetization_raw_factor_base")
    pu_temp = _cfg.path("purification_temp")
    _require_dir(pu_temp, "purification_temp")
    n_workers = n_workers or max(1, cpu_count() - 1)

    logger.info("Purification 多进程启动: workers=%d input=%s", n_workers, raw_base)

    tasks = _discover_assetization_output(raw_base)
    if not tasks:
        logger.info("assetization_raw_factor_base 中无待纯化因子")
        return {"status": "idle", "total": 0}

    logger.info("发现 %d 个待纯化因子", len(tasks))
    task_queue: Queue = Queue()
    result_queue: Queue = Queue()

    workers = []
    for i in range(n_workers):
        p = Process(
            target=_purification_worker,
            args=(task_queue, result_queue, i + 1),
        )
        p.start()
        workers.append(p)

    for t in tasks:
        task_queue.put(t)

    results = []
    for _ in range(len(tasks)):
        r = result_queue.get()
        results.append(r)
        logger.info("%s %s → factor_id=%s",
                     "✅" if r["status"] == "done" else "❌",
                     r["candidate_id"], r.get("factor_id", "?"))

    for _ in workers:
        task_queue.put(None)
    for p in workers:
        p.join(timeout=10)

    done = sum(1 for r in results if r["status"] == "done")
    logger.info("Purification 完成: %d/%d 成功", done, len(tasks))
    return {"total": len(tasks), "done": done, "results": results}


# ============================================================
# Purification 路由
# ============================================================

def run_purification_router() -> list[dict]:
    """路由：将 tier0/purification_temp 中的因子分发到目标目录。"""
    from purification.scripts.router import PurificationRouter

    _cfg = _get_config()
    router = PurificationRouter(
        _cfg.path("purification_temp"),
        _cfg.path("purification_pure_factor_base"),
    )
    records = router.scan_and_route()
    for r in records:
        logger.info("纯化路由: %s → %s (%s)", r["candidate_id"], r["factor_id"], r["label"])
    return records


# ============================================================
# Evaluation Worker
# ============================================================


def _evaluation_worker(
    task_queue: Queue,
    result_queue: Queue,
    worker_id: int,
):
    """Evaluation 评估工作进程。"""
    import os, sys as _sys
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if _script_dir not in _sys.path:
        _sys.path.insert(0, _script_dir)

    from evaluation.scripts.worker import process_factor_dir as pfd
    from evaluation.config import EvaluationConfig

    cfg = EvaluationConfig(config_dir=_get_config().config_dir)
    _setup_worker_logging("evaluation", worker_id, _get_config().logging_directory)
    _lg = logging.getLogger("evaluation.worker")
    _lg.info("Evaluation Worker %d 启动", worker_id)

    while True:
        task = task_queue.get()
        if task is None:
            _lg.info("Worker %d 停止", worker_id)
            break

        factor_dir = task
        try:
            factor_id, temp_dir, route_info = pfd(
                factor_dir,
                config=cfg,
                temp_base=str(_get_config().path("evaluation_temp")),
            )
            result_queue.put({
                "status": "done",
                "factor_id": factor_id,
                "candidate_id": Path(factor_dir).name,
                "temp_dir": temp_dir,
                "route_recommendation": route_info.get("route_recommendation", ""),
                "target_tier": route_info.get("target_tier", ""),
                "worker_id": worker_id,
            })
        except Exception as e:
            result_queue.put({
                "status": "error",
                "candidate_id": Path(factor_dir).name,
                "error": str(e),
                "worker_id": worker_id,
            })


def _discover_purification_output(pool_dir: Path) -> list[Path]:
    """扫描 purification_pure_factor_base，发现待评估的因子。

    判断标准：目录中有 afv.json 且尚未包含 Evaluation 段。
    """
    factors = []
    if not pool_dir.exists():
        return factors
    for fd in sorted(pool_dir.iterdir()):
        if not fd.is_dir():
            continue
        afv_path = fd / "afv.json"
        if not afv_path.exists():
            continue
        try:
            afv = json.loads(afv_path.read_text())
            if "Evaluation" not in afv:
                factors.append(fd)
        except Exception:
            continue
    return factors


# ============================================================
# Evaluation 多进程管道
# ============================================================

def run_evaluation_pipeline(
    n_workers: int = 0,
    pure_factor_base: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluation 多进程评估管道。

    扫描 tier1/purification_pure_factor_base → 多进程评估 → tier0/evaluation_temp
    """
    _cfg = _get_config()
    pure_base = Path(pure_factor_base) if pure_factor_base else _cfg.path("purification_pure_factor_base")
    ev_temp = _cfg.path("evaluation_temp")
    _require_dir(ev_temp, "evaluation_temp")
    n_workers = n_workers or max(1, cpu_count() - 1)

    logger.info("Evaluation 多进程启动: workers=%d input=%s", n_workers, pure_base)

    tasks = _discover_purification_output(pure_base)
    if not tasks:
        logger.info("purification_pure_factor_base 中无待评估因子")
        return {"status": "idle", "total": 0}

    logger.info("发现 %d 个待评估因子", len(tasks))
    task_queue: Queue = Queue()
    result_queue: Queue = Queue()

    workers = []
    for i in range(n_workers):
        p = Process(
            target=_evaluation_worker,
            args=(task_queue, result_queue, i + 1),
        )
        p.start()
        workers.append(p)

    for t in tasks:
        task_queue.put(t)

    results = []
    for _ in range(len(tasks)):
        r = result_queue.get()
        results.append(r)
        logger.info("%s %s → factor_id=%s route=%s",
                     "✅" if r["status"] == "done" else "❌",
                     r["candidate_id"], r.get("factor_id", "?"),
                     r.get("route_recommendation", "?"))

    for _ in workers:
        task_queue.put(None)
    for p in workers:
        p.join(timeout=60)

    done = sum(1 for r in results if r["status"] == "done")
    logger.info("Evaluation 完成: %d/%d 成功", done, len(tasks))
    return {"total": len(tasks), "done": done, "results": results}


# ============================================================
# Evaluation 路由
# ============================================================

def run_evaluation_router() -> list[dict]:
    """路由：将 tier0/evaluation_temp 中的因子分发到目标 tier 目录。"""
    from evaluation.scripts.router import EvaluationRouter
    from evaluation.config import EvaluationConfig

    _cfg = _get_config()
    cfg = EvaluationConfig(config_dir=_cfg.config_dir)
    router = EvaluationRouter(_cfg.path("evaluation_temp"), config=cfg)
    records = router.scan_and_route()
    for r in records:
        logger.info("评估路由: %s → %s (%s)", r["candidate_id"], r["target"], r["route_recommendation"])
    return records


# ============================================================
# 空候选池归档
# ============================================================

def _archive_empty_campaigns(pool_dir: Path, record_dir: Path) -> list[str]:
    """扫描候选池，将已空的 campaign 目录移至 archive 路径下归档。

    空判断标准：目录内仅有 config.json 而无任何因子子目录（含 manifest.json）。
    归档后该 campaign 的历史信息（config.json）得以保留。

    Args:
        pool_dir: candidate_pool/candidate 路径。
        record_dir: candidate_pool/archive 归档路径。

    Returns:
        已归档的 campaign_id 列表。
    """
    import shutil
    archived = []
    if not pool_dir.exists():
        return archived
    record_dir.mkdir(parents=True, exist_ok=True)

    for item in sorted(pool_dir.iterdir()):
        if not item.is_dir():
            continue
        cfg = item / "config.json"
        if not cfg.exists():
            continue  # 无 config.json，不视为有效 campaign

        # 检查是否还有因子子目录
        has_factors = False
        for sub in item.iterdir():
            if sub.is_dir() and (sub / "manifest.json").exists():
                has_factors = True
                break

        if not has_factors:
            target = record_dir / item.name
            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(item), str(target))
            archived.append(item.name)
            logger.info("空池归档: %s → %s", item.name, target)

    return archived


def run_gtja185_batch_pipeline(
    *,
    output_dir: str | Path,
    start_date: str | None = None,
    end_date: str | None = None,
    horizons: tuple[int, ...] = (1, 5, 21),
    batch_size: int = 8,
    min_assets: int = 20,
    limit: int | None = None,
    synthetic: bool = False,
    materialize_staging: bool = False,
    publish: bool = False,
    strict: bool = True,
):
    """Run the canonical 185-factor pack through the new batch evaluator."""
    from evaluation.gtja185_batch import (
        BatchEvaluationConfig,
        build_synthetic_market_frame,
        run_gtja185_evaluation,
    )

    config = BatchEvaluationConfig(
        market="ashare",
        start_date=start_date,
        end_date=end_date,
        horizons=horizons,
        batch_size=batch_size,
        min_assets=min_assets,
        limit=limit,
        materialize_staging=materialize_staging,
        publish=publish,
        strict=strict,
    )
    frame = build_synthetic_market_frame(periods=420, symbols=max(8, min_assets)) if synthetic else None
    snapshot_id = "synthetic:gtja185-cli" if synthetic else None
    return run_gtja185_evaluation(
        config,
        output_dir=output_dir,
        market_frame=frame,
        snapshot_id=snapshot_id,
    )


# ============================================================
# CLI
# ============================================================

def _validate_all_paths(cfg) -> None:
    """启动前验证所有本地必需路径是否存在。缺失则自动创建。

    路径分两类:
      - COS 端（candidate_pool / factor_pool）: 由 ConfigManager 初始化时自动 ensure。
      - 本地端（local_tmp）: 包括 tier0 临时目录、缓存、日志目录，自动创建。
    """
    # 本地端路径 — 自动创建
    local_keys = [
        "gateway_temp", "assetization_temp", "purification_temp", "evaluation_temp",
        "market_data_cache", "timeseries_forward_return_cache", "evaluation_label_cache",
    ]
    for key in local_keys:
        p = cfg.path(key)
        p.mkdir(parents=True, exist_ok=True)

    # 日志子目录
    _log_dir = cfg.logging_directory
    _log_dir.mkdir(parents=True, exist_ok=True)
    for stage in ["pipeline", "gateway", "assetization", "purification", "evaluation"]:
        (cfg.logging_directory / stage).mkdir(parents=True, exist_ok=True)

    logger.info("路径验证通过: 本地临时目录已就绪 (%d 个)", len(local_keys))


def main():
    import argparse
    import time as _time

    # 初始化日志（加载配置前用默认位置，后续 main 中切换）
    _setup_logging()

    p = argparse.ArgumentParser(description="AutoFactorEvaluation 全流程管道")
    p.add_argument("--gtja185", action="store_true", help="评估完整 GTJA185 内置因子包")
    p.add_argument("--gtja-output", default="AutoFactorEvaluation-RECONSTRUCT/output/gtja185")
    p.add_argument("--gtja-start-date", default=None)
    p.add_argument("--gtja-end-date", default=None)
    p.add_argument("--gtja-horizons", default="1,5,21")
    p.add_argument("--gtja-batch-size", type=int, default=8)
    p.add_argument("--gtja-min-assets", type=int, default=20)
    p.add_argument("--gtja-limit", type=int, default=None)
    p.add_argument("--gtja-synthetic", action="store_true")
    p.add_argument("--gtja-materialize-staging", action="store_true")
    p.add_argument("--gtja-publish", action="store_true")
    p.add_argument("--gtja-allow-partial", action="store_true")
    p.add_argument("--gateway-only", action="store_true", help="仅 Gateway 审查+路由")
    p.add_argument("--assetization-only", action="store_true", help="仅 Assetization 计算+路由")
    p.add_argument("--all", action="store_true", help="全流程（Gateway → Assetization → Purification → Evaluation）")
    p.add_argument("--purification-only", action="store_true", help="仅 Purification 纯化+路由")
    p.add_argument("--evaluation-only", action="store_true", help="仅 Evaluation 评估+路由")
    p.add_argument("--workers", type=int, default=0, help="全局 Worker 数（0=CPU-1），被各阶段独立参数覆盖")
    p.add_argument("--gw-workers", type=int, default=None, help="Gateway Worker 数")
    p.add_argument("--as-workers", type=int, default=None, help="Assetization Worker 数")
    p.add_argument("--pu-workers", type=int, default=None, help="Purification Worker 数")
    p.add_argument("--ev-workers", type=int, default=4, help="Evaluation Worker 数（推荐多于前序阶段）")
    p.add_argument("--market-data", default=None, help="行情数据路径（默认从 Gateway 配置读取）")
    p.add_argument("--gateway-config", default=None, help="Gateway 配置目录")
    p.add_argument("--preload", action="store_true", default=True,
                    help="预加载行情数据到进程内存（fork 子进程继承，零 I/O 读取，默认开启）")
    p.add_argument("--no-preload", action="store_false", dest="preload",
                    help="禁用进程内存预加载")
    args = p.parse_args()

    if args.gtja185:
        from dataclasses import asdict

        horizons = tuple(int(part.strip()) for part in args.gtja_horizons.split(",") if part.strip())
        summary = run_gtja185_batch_pipeline(
            output_dir=args.gtja_output,
            start_date=args.gtja_start_date,
            end_date=args.gtja_end_date,
            horizons=horizons,
            batch_size=args.gtja_batch_size,
            min_assets=args.gtja_min_assets,
            limit=args.gtja_limit,
            synthetic=args.gtja_synthetic,
            materialize_staging=args.gtja_materialize_staging,
            publish=args.gtja_publish,
            strict=not args.gtja_allow_partial,
        )
        print(json.dumps(asdict(summary), ensure_ascii=False, indent=2, default=str))
        return

    # 全流程启动前验证所有路径已就绪
    if not any([args.gateway_only, args.assetization_only, args.purification_only, args.evaluation_only]):
        _validate_all_paths(_get_config())

    if args.gateway_only:
        stats = run_gateway_pipeline(n_workers=args.workers, market_data_path=args.market_data, gateway_config_dir=args.gateway_config)
        if stats.get("done", 0) > 0:
            run_router()
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return

    if args.assetization_only:
        stats = run_assetization_pipeline(n_workers=args.workers, market_data_path=args.market_data)
        if stats.get("done", 0) > 0:
            run_assetization_router()
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return

    if args.purification_only:
        stats = run_purification_pipeline(n_workers=args.workers)
        if stats.get("done", 0) > 0:
            run_purification_router()
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return

    if args.evaluation_only:
        stats = run_evaluation_pipeline(n_workers=args.workers)
        if stats.get("done", 0) > 0:
            run_evaluation_router()
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return

    # === 预加载行情数据到进程内存（fork 子进程继承，零 I/O 读取）===
    if args.preload:
        logger.info("=" * 60)
        logger.info("预加载行情数据到进程内存 (fork 继承)")
        logger.info("=" * 60)
        from cache_utils import preload_market_data
        _cfg = _get_config()
        _bar_dir = _cfg.market_data_path
        _t0 = _time.perf_counter()
        preload_market_data(_bar_dir)
        _t1 = _time.perf_counter()
        logger.info("预加载完成: %.1f 秒 (后续 Worker fork 继承，无需重复 I/O)", _t1 - _t0)

    # 全流程（带计时）
    logger.info("=" * 60)
    logger.info("全流程管道启动 - 性能基准测试")
    logger.info("=" * 60)
    _bench = {}

    # 阶段1: Gateway
    _t0 = _time.perf_counter()
    gw_stats = run_gateway_pipeline(n_workers=args.gw_workers or args.workers, market_data_path=args.market_data, gateway_config_dir=args.gateway_config)
    _t1 = _time.perf_counter()
    _bench["Gateway"] = {"sec": round(_t1 - _t0, 1), "done": gw_stats.get("done", 0)}
    if gw_stats.get("done", 0) > 0:
        run_router()
    else:
        logger.info("Gateway 无通过因子，跳过后续")

    # 阶段2: Assetization
    _t0 = _time.perf_counter()
    as_stats = run_assetization_pipeline(n_workers=args.as_workers or args.workers, market_data_path=args.market_data)
    _t1 = _time.perf_counter()
    _bench["Assetization"] = {"sec": round(_t1 - _t0, 1), "done": as_stats.get("done", 0)}
    if as_stats.get("done", 0) > 0:
        run_assetization_router()

    # 阶段3: Purification
    _t0 = _time.perf_counter()
    pu_stats = run_purification_pipeline(n_workers=args.pu_workers or args.workers)
    _t1 = _time.perf_counter()
    _bench["Purification"] = {"sec": round(_t1 - _t0, 1), "done": pu_stats.get("done", 0)}
    if pu_stats.get("done", 0) > 0:
        run_purification_router()

    # 阶段4: Evaluation
    _t0 = _time.perf_counter()
    ev_stats = run_evaluation_pipeline(n_workers=args.ev_workers or args.workers)
    _t1 = _time.perf_counter()
    _bench["Evaluation"] = {"sec": round(_t1 - _t0, 1), "done": ev_stats.get("done", 0)}
    if ev_stats.get("done", 0) > 0:
        run_evaluation_router()

    # 报告
    logger.info("=" * 60)
    logger.info("🏁 性能基准报告")
    logger.info("=" * 60)
    _total_sec = sum(v["sec"] for v in _bench.values())
    for _stage, _info in _bench.items():
        _pct = _info["sec"] / _total_sec * 100 if _total_sec > 0 else 0
        logger.info("  %-15s %6.1f秒 (%5.1f%%)  处理 %d 个因子",
                     _stage, _info["sec"], _pct, _info["done"])
    logger.info("  %-15s %6.1f秒 (总计)", "总耗时", _total_sec)
    logger.info("全流程完成")


if __name__ == "__main__":
    main()
