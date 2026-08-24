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
from alphaprobe.runner import PROJECT_ROOT, build_train_parser, run_mining_campaign
from shared.utils.llm import LLMQuotaExhaustedError

STATE_FILENAME = "active_round.json"
STOP_QUOTA_FILENAME = "STOP_QUOTA"


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
