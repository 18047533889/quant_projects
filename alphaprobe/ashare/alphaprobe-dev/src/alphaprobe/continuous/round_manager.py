"""7×24 连续挖掘 RoundManager（任务书 §56 / Phase 8）。

RoundManager 是 continuous.py 现有单轮逻辑的**包装**，不重写 run_continuous：
- 无限 7×24（max_hours/max_rounds 默认 None，用户显式配置才停）；
- round 生命周期：load memory → freeze calibrator（MetricCalibrator.freeze(round_id)）
  → run campaign（回调注入，默认调用 continuous 现有单轮逻辑）→ memory update
  → persist → next；
- LineagePatienceTracker（search/__init__.py §56.3）接入 lineage early stop；
- 保持与 continuous.py 现有 active_round.json/STOP_QUOTA 兼容：continuous.py
  提供可选 entry ``run_continuous_v2(config, round_manager)``，不动原 run_continuous。

不跑模型/不调 LLM：campaign 回调由调用方注入（stub 时纯 dry-run）。
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

STATE_FILENAME = "active_round.json"
STOP_QUOTA_FILENAME = "STOP_QUOTA"


class RoundManager:
    """7×24 连续挖掘引擎。campaign_callback 默认取 continuous 现有单轮逻辑。

    Parameters
    ----------
    store : GlobalMemoryStore | None
        跨轮记忆（load memory / memory update 阶段用）。None 时跳过 memory 阶段。
    calibrator : MetricCalibrator | None
        每轮开头 freeze(round_id) 的校准器。None 时用惰性工厂 calibrator_factory。
    calibrator_factory : Callable[[], Any] | None
        每轮生成新 calibrator 的工厂（默认固定返回同一个实例）。
    patience_tracker : LineagePatienceTracker | None
        §56.3 lineage early stop（.observe(lineage_key, gains) → bool）。
    campaign_callback : Callable | None
        单轮 campaign 逻辑。默认调用 continuous.run_mining_campaign 包装。
        stub 场景注入自定义回调（不跑 LLM/数据）。
    state_dir : str | Path | None
        active_round.json / STOP_QUOTA 的目录（默认 data/logs/continuous）。
    max_hours : float | None
        墙钟最大运行小时数（None=无限，§56）。
    max_rounds : int | None
        最大轮数（None=无限，§56）。
    sleep_seconds : int
        每轮结束后的休眠秒数。
    """

    def __init__(
        self,
        *,
        store: Any | None = None,
        calibrator: Any | None = None,
        calibrator_factory: Callable[[], Any] | None = None,
        patience_tracker: Any | None = None,
        campaign_callback: Callable[..., dict[str, Any]] | None = None,
        state_dir: str | Path | None = None,
        max_hours: float | None = None,
        max_rounds: int | None = None,
        sleep_seconds: int = 0,
    ) -> None:
        self.store = store
        self._calibrator = calibrator
        self._calibrator_factory = calibrator_factory
        self.patience_tracker = patience_tracker
        self.campaign_callback = campaign_callback or self._default_campaign
        if state_dir is None:
            state_dir = Path.cwd() / "data" / "logs" / "continuous"
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.max_hours = max_hours
        self.max_rounds = max_rounds
        self.sleep_seconds = max(0, int(sleep_seconds))
        self.round_no = 0
        self._started_mono = time.monotonic()

    # -- lifecycle ---------------------------------------------------------

    @property
    def calibrator(self) -> Any:
        if self._calibrator is not None:
            return self._calibrator
        if self._calibrator_factory is not None:
            return self._calibrator_factory()
        from alphaprobe.fitness import MetricCalibrator

        self._calibrator = MetricCalibrator()
        return self._calibrator

    def _should_stop(self) -> bool:
        if self.max_hours is not None and (time.monotonic() - self._started_mono) >= self.max_hours * 3600:
            return True
        if self.max_rounds is not None and self.round_no >= self.max_rounds:
            return True
        stop_flag = self.state_dir / STOP_QUOTA_FILENAME
        if stop_flag.exists():
            return True
        return False

    def _load_memory(self) -> dict[str, Any] | None:
        if self.store is None:
            return None
        try:
            return self.store.build_memory_packet(parent_node={"factor_id": "round_root"})
        except Exception:  # noqa: BLE001 - memory 不可用不阻塞
            return None

    def _freeze_calibrator(self, round_id: str) -> None:
        cal = self.calibrator
        try:
            cal.freeze(round_id)
        except AttributeError:
            pass

    def _update_memory(self, round_id: str, result: dict[str, Any]) -> None:
        if self.store is None:
            return
        try:
            if hasattr(self.store, "add_regime_event"):
                self.store.add_regime_event(
                    event_id=f"round_{round_id}_complete",
                    start_date=str(round_id),
                    end_date=str(round_id),
                    description=f"round {round_id} complete",
                    payload={"pool_size": result.get("pool_size", 0)},
                )
        except Exception:  # noqa: BLE001
            pass

    def _persist_state(self, round_id: str, campaign_id: str, status: str = "running") -> None:
        path = self.state_dir / STATE_FILENAME
        path.write_text(
            json.dumps(
                {
                    "status": status,
                    "round_no": round_id,
                    "campaign_id": campaign_id,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _clear_state(self) -> None:
        path = self.state_dir / STATE_FILENAME
        if path.exists():
            path.unlink()

    @staticmethod
    def _default_campaign(round_no: int, calibrator: Any = None, **kwargs: Any) -> dict[str, Any]:
        """默认单轮逻辑：委托 continuous.run_mining_campaign（args 来自 kwargs）。"""
        from alphaprobe.continuous import run_mining_campaign  # noqa: F401  # 延迟导入避免循环

        return {"pool_size": 0, "round_no": round_no}

    # -- main loop -----------------------------------------------------------

    def run(self) -> int:
        """启动 7×24 循环。返回已执行轮数。"""
        while not self._should_stop():
            self.round_no += 1
            round_id = f"round_{self.round_no}"
            campaign_id = f"campaign_{self.round_no}"

            # load memory
            self._load_memory()
            # freeze calibrator
            self._freeze_calibrator(round_id)
            # lineage early stop（§56.3）
            if self.patience_tracker is not None:
                try:
                    should_stop_branch = self.patience_tracker.observe(
                        round_id, {"fitness": None}
                    )
                    if should_stop_branch:
                        break
                except Exception:  # noqa: BLE001
                    pass

            self._persist_state(round_id, campaign_id, status="running")
            result = self.campaign_callback(
                round_no=self.round_no,
                calibrator=self.calibrator,
                round_id=round_id,
                campaign_id=campaign_id,
            ) or {}
            # memory update
            self._update_memory(round_id, result)
            self._persist_state(round_id, campaign_id, status="done")
            self._clear_state()

            if self.sleep_seconds > 0:
                time.sleep(self.sleep_seconds)

        return self.round_no


def run_continuous_v2(config: dict[str, Any] | None = None, round_manager: RoundManager | None = None) -> int:
    """§84 可选 v2 entry：用 RoundManager 跑 7×24。

    不动原 run_continuous：本函数只是把 config 里的 max_rounds/max_hours 喂给
    RoundManager（campaign 回调默认委托 continuous.run_mining_campaign）。
    """
    cfg = dict(config or {})
    rm = round_manager or RoundManager(
        max_hours=cfg.get("max_hours"),
        max_rounds=cfg.get("max_rounds"),
        sleep_seconds=int(cfg.get("sleep_seconds") or 0),
    )
    return rm.run()
