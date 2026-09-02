"""7×24 连续挖掘：每轮新 campaign、新冷启动、投递 candidate_pool；崩溃可衔接续跑。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alphaprobe.delivery.config import DOMAIN_ROOT_TO_SLUG, ExperimentConfig
from alphaprobe.memory import GlobalMemoryStore
from alphaprobe.memory.seed_ingestion import ingest_cold_start_yaml
from alphaprobe.runner import PROJECT_ROOT, _LAST_POOL, build_train_parser, run_mining_campaign
from shared.utils.llm import LLMQuotaExhaustedError

STATE_FILENAME = "active_round.json"
STOP_QUOTA_FILENAME = "STOP_QUOTA"


def run_continuous_v2(args: Any) -> None:
    """§84 可选 v2 entry：用 RoundManager 跑 7×24（不动原 run_continuous）。

    args 与 run_continuous 同构；campaign 回调默认委托 run_mining_campaign。
    max_rounds/max_hours 显式给才停（§56），否则无限。
    """
    from alphaprobe.continuous.round_manager import RoundManager

    continuous_raw: dict = {}
    if getattr(args, "experiment_config", None):
        try:
            experiment = ExperimentConfig.from_yaml(args.experiment_config)
            continuous_raw = dict(experiment.raw.get("continuous") or {})
        except Exception:  # noqa: BLE001 - yaml 缺失时用纯 CLI 参数
            continuous_raw = {}

    max_rounds = args.max_rounds
    if max_rounds is None and continuous_raw.get("max_rounds") is not None:
        max_rounds = int(continuous_raw["max_rounds"])
    max_hours = args.max_hours
    if max_hours is None and continuous_raw.get("max_hours") is not None:
        max_hours = float(continuous_raw["max_hours"])
    sleep_seconds = int(
        args.sleep_seconds
        if args.sleep_seconds is not None
        else continuous_raw.get("sleep_seconds", 0)
    )

    def campaign_cb(round_no: int, calibrator: Any = None, **kwargs: Any) -> dict[str, Any]:
        experiment = ExperimentConfig.from_yaml(args.experiment_config)
        campaign_id = str(kwargs.get("campaign_id") or f"campaign_{round_no}")
        args.resume = False
        return run_mining_campaign(args, campaign_id=campaign_id, experiment=experiment)

    rm = RoundManager(
        store=GlobalMemoryStore() if _memory_db_ok() else None,
        campaign_callback=campaign_cb,
        max_hours=max_hours,
        max_rounds=max_rounds,
        sleep_seconds=sleep_seconds,
    )
    rm.run()


def _memory_db_ok() -> bool:
    try:
        GlobalMemoryStore()
        return True
    except Exception:  # noqa: BLE001
        return False


def _setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "continuous.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def _generate_campaign_id(experiment: ExperimentConfig) -> str:
    settings = experiment.delivery
    slug = DOMAIN_ROOT_TO_SLUG.get(settings.domain_root, "pv")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
    base = f"{settings.generator_name}_{settings.market}_{slug}_{stamp}"
    out_root = Path(settings.output_dir).expanduser()
    candidate_dir = out_root / base
    if candidate_dir.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        base = f"{settings.generator_name}_{settings.market}_{slug}_{stamp}"
    return base


def _state_path(log_dir: Path) -> Path:
    return log_dir / STATE_FILENAME


def _load_state(log_dir: Path) -> dict[str, Any] | None:
    path = _state_path(log_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_state(log_dir: Path, state: dict[str, Any]) -> None:
    path = _state_path(log_dir)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _clear_state(log_dir: Path) -> None:
    path = _state_path(log_dir)
    if path.exists():
        path.unlink()


def _write_stop_quota(log_dir: Path, reason: str) -> None:
    path = log_dir / STOP_QUOTA_FILENAME
    path.write_text(
        json.dumps(
            {
                "reason": reason,
                "at": datetime.now(timezone.utc).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def run_continuous(args: Any) -> None:
    continuous_raw: dict = {}
    sleep_seconds = 60
    max_rounds = None
    error_sleep_seconds = 300
    max_hours: float | None = 24.0
    quota_max_consecutive = 1

    if args.experiment_config:
        experiment = ExperimentConfig.from_yaml(args.experiment_config)
        continuous_raw = dict(experiment.raw.get("continuous") or {})
        sleep_seconds = int(
            args.sleep_seconds
            if args.sleep_seconds is not None
            else continuous_raw.get("sleep_seconds", 60)
        )
        max_rounds = args.max_rounds
        if max_rounds is None and continuous_raw.get("max_rounds") is not None:
            max_rounds = int(continuous_raw["max_rounds"])
        error_sleep_seconds = int(continuous_raw.get("error_sleep_seconds", 300))
        if args.max_hours is not None:
            max_hours = float(args.max_hours)
        elif continuous_raw.get("max_hours") is not None:
            max_hours = float(continuous_raw["max_hours"])
        quota_max_consecutive = int(continuous_raw.get("llm_quota_max_consecutive", 1))

    log_dir = PROJECT_ROOT / "data" / "logs" / "continuous"
    _setup_logging(log_dir)

    # Phase 7：全局记忆单例（跨轮持久 lineage）。db 路径来自环境变量
    # ALPHAPROBE_MEMORY_DB，默认 ~/quant_projects/data/alphaprobe/global_memory.sqlite3。
    # 初始化失败只打印告警，不阻塞挖掘主循环。
    memory_store: GlobalMemoryStore | None = None
    try:
        memory_store = GlobalMemoryStore()
        logging.info("global memory ready db=%s", memory_store.db_path)
    except Exception as exc:
        logging.warning("global memory init failed (non-blocking): %s", exc)

    stop_flag = log_dir / STOP_QUOTA_FILENAME
    if stop_flag.exists() and not getattr(args, "ignore_stop_quota", False):
        logging.error(
            "STOP_QUOTA present at %s — refuse to start (quota previously exhausted). "
            "Delete the file or pass --ignore_stop_quota after topping up.",
            stop_flag,
        )
        return

    started_mono = time.monotonic()
    logging.info(
        "continuous mining start experiment=%s sleep=%ss max_rounds=%s max_hours=%s",
        args.experiment_config,
        sleep_seconds,
        max_rounds,
        max_hours,
    )

    round_no = 0
    quota_fails = 0
    memory_seed_ingested = False

    while True:
        if max_hours is not None and (time.monotonic() - started_mono) >= max_hours * 3600:
            logging.info("reached max_hours=%.2f, stopping cleanly", max_hours)
            _clear_state(log_dir)
            break

        round_no += 1
        if max_rounds is not None and round_no > max_rounds:
            logging.info("reached max_rounds=%s, stopping", max_rounds)
            _clear_state(log_dir)
            break

        experiment = ExperimentConfig.from_yaml(args.experiment_config)

        # 崩溃衔接：若上一轮未完成，续同一 campaign + checkpoint
        prev = _load_state(log_dir)
        resume_round = bool(
            prev
            and prev.get("status") == "running"
            and prev.get("campaign_id")
        )
        if resume_round:
            campaign_id = str(prev["campaign_id"])
            round_no = int(prev.get("round_no") or round_no)
            args.resume = True
            logging.info(
                "=== resume round %s campaign_id=%s (checkpoint continue) ===",
                round_no,
                campaign_id,
            )
        else:
            campaign_id = _generate_campaign_id(experiment)
            args.resume = False
            logging.info("=== round %s campaign_id=%s ===", round_no, campaign_id)

        args.cold_start_seed = None

        # Phase 7：冷启动 seed 摄入 global memory（yaml 缺失/为空 → 0 条，不阻塞）。
        # 每轮 seed 不同（mix 抽样），摄入天然去重（signal id 相同 → rediscovery 合并）。
        if memory_store is not None:
            try:
                cs_cfg = (experiment.raw.get("mining") or {}).get("cold_start_library")
                if cs_cfg:
                    yaml_path = str(cs_cfg)
                    if not Path(yaml_path).expanduser().is_absolute():
                        yaml_path = str(PROJECT_ROOT / yaml_path)
                    ingest_cold_start_yaml(yaml_path, memory_store)
                elif not memory_seed_ingested:
                    # 兜底：库默认目录存在则尝试，缺失/空 → 0（已知状态）
                    from alphaprobe.cold_start import package_root

                    pkg_data = Path(package_root()) / "data"
                    for sub in ("ashare", "alpha101", "alpha191", "alpha158"):
                        candidate = pkg_data / sub
                        if candidate.is_dir():
                            for f in sorted(candidate.glob("*.yaml")):
                                ingest_cold_start_yaml(f, memory_store)
                    memory_seed_ingested = True
            except Exception as exc:
                logging.warning("seed ingestion skipped (non-blocking): %s", exc)

        _save_state(
            log_dir,
            {
                "status": "running",
                "round_no": round_no,
                "campaign_id": campaign_id,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "resume": bool(args.resume),
            },
        )

        try:
            result = run_mining_campaign(args, campaign_id=campaign_id, experiment=experiment)
            quota_fails = 0
            _clear_state(log_dir)
            # Phase 7：每轮 campaign 结束后把该轮 pool 快照 upsert 进 factor_nodes
            # （source_system='alphaprobe'）——跨轮 lineage 持久化的核心写入点。
            if memory_store is not None:
                try:
                    _upsert_pool_snapshot(memory_store, experiment, result)
                except Exception as exc:
                    logging.warning("pool snapshot upsert failed (non-blocking): %s", exc)
            logging.info(
                "round %s finished pool_size=%s submitted=%s dir=%s",
                round_no,
                result.get("pool_size"),
                (result.get("export") or {}).get("candidates_submitted"),
                (result.get("export") or {}).get("campaign_dir"),
            )
        except LLMQuotaExhaustedError as exc:
            # llm.py 内已限次重试；再空转开新轮无意义，直接停
            quota_fails += 1
            logging.error("round %s LLM quota exhausted: %s", round_no, exc)
            _save_state(
                log_dir,
                {
                    "status": "quota_stopped",
                    "round_no": round_no,
                    "campaign_id": campaign_id,
                    "error": str(exc),
                    "at": datetime.now(timezone.utc).isoformat(),
                },
            )
            if quota_fails >= quota_max_consecutive:
                _write_stop_quota(log_dir, str(exc))
                logging.error(
                    "quota failures=%s >= %s — writing STOP_QUOTA and exiting (no empty spinning)",
                    quota_fails,
                    quota_max_consecutive,
                )
                return
            # 未达阈值：保留 campaign，下一圈 resume 同一轮，不新开冷启动
            _save_state(
                log_dir,
                {
                    "status": "running",
                    "round_no": round_no,
                    "campaign_id": campaign_id,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "resume": True,
                    "last_quota_error": str(exc),
                },
            )
            logging.info("sleep %ss before resume-retry after quota error", error_sleep_seconds)
            time.sleep(error_sleep_seconds)
            continue
        except Exception as exc:
            logging.error("round %s failed: %s", round_no, exc)
            logging.error(traceback.format_exc())
            # 保留 active_round，systemd/手动重启时可 resume 衔接
            logging.info("sleep %ss before retry (state kept for resume)", error_sleep_seconds)
            time.sleep(error_sleep_seconds)
            continue

        if sleep_seconds > 0:
            logging.info("sleep %ss before next round", sleep_seconds)
            time.sleep(sleep_seconds)


def _upsert_pool_snapshot(
    store: GlobalMemoryStore,
    experiment: ExperimentConfig,
    result: dict[str, Any],
) -> None:
    """每轮 campaign 结束后把该轮 pool 快照 upsert 进 factor_nodes。

    source_system='alphaprobe'，signal id 去重（同 signal 跨轮 rediscovery
    合并，不重复建节点）。snapshot 路径来自 export.campaign_dir，无 export
    则跳过——pool 本体在 run_mining_campaign 内已释放。
    """
    export = result.get("export") or {}
    snapshot = str(export.get("campaign_dir") or "").strip() or None
    if not snapshot:
        return
    pool = _LAST_POOL[0]
    if pool is None:
        logging.info("no pool snapshot available (in-process) — skipped")
        return
    from alphaprobe.authority import FactorIdentityAuthorityError, build_identity_view
    from alphaprobe.fe_bridge.dsl_convert import expression_to_dsl
    from alphaprobe.delivery.exporter import _candidate_hash

    campaign_id = str(experiment.delivery.resolved_campaign_id())
    settings = experiment.delivery
    created = rediscovered = 0
    for i in range(pool.size):
        expr = pool.exprs[i]
        if expr is None:
            continue
        try:
            formula = expression_to_dsl(expr)
        except Exception:
            continue
        # FE 权威身份（§28）；FE 不可用 fail-closed skip，不回落文本 regex。
        try:
            iv = build_identity_view(formula)
        except FactorIdentityAuthorityError as exc:
            logging.warning("authority identity unavailable, skip pool snapshot: %r: %s", formula, exc)
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
            schema_json={
                "fitness": None,
                "candidate_hash": _candidate_hash(
                    formula, settings.universe_id, settings.frequency_bucket
                ),
            },
        )
        if ok:
            created += 1
        elif existing is not None:
            rediscovered += 1
    logging.info(
        "pool snapshot upsert campaign=%s created=%s rediscovered=%s snapshot=%s",
        campaign_id,
        created,
        rediscovered,
        snapshot,
    )


def build_continuous_parser() -> argparse.ArgumentParser:
    parser = build_train_parser()
    parser.description = "AlphaPROBE 7×24 连续挖掘（每轮新冷启动 + candidate_pool 投递）"
    parser.add_argument(
        "--sleep_seconds",
        type=int,
        default=None,
        help="每轮结束后的休眠秒数（默认读 yaml continuous.sleep_seconds）",
    )
    parser.add_argument(
        "--max_rounds",
        type=int,
        default=None,
        help="最大轮数；不设则按 max_hours / 无限",
    )
    parser.add_argument(
        "--max_hours",
        type=float,
        default=None,
        help="墙钟最大运行小时数（默认读 yaml continuous.max_hours，常用 24）",
    )
    parser.add_argument(
        "--ignore_stop_quota",
        action="store_true",
        help="忽略 STOP_QUOTA 停止旗标，强制继续跑",
    )
    return parser


def main() -> None:
    parser = build_continuous_parser()
    args = parser.parse_args()
    run_continuous(args)


if __name__ == "__main__":
    main()
