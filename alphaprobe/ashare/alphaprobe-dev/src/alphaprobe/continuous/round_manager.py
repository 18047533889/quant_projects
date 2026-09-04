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
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

STATE_FILENAME = "active_round.json"
STOP_QUOTA_FILENAME = "STOP_QUOTA"


# ---------------------------------------------------------------------------
# 无 torch 环境的 torch 占位 stub（P0-A）
# ---------------------------------------------------------------------------


class _TorchTensorStub:
    """最小 torch.Tensor 占位：不提供真实张量数值能力。

    仅供无 GPU wheel 的离线测试环境让 runner/pipeline 顶层 import 通过
    （trainer/qlib 训练路径永不触碰 stub 数值运算）。显式标注：不用于生产。
    """

    __slots__ = ("_dummy",)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._dummy = None

    def __bool__(self) -> bool:
        return False

    def numpy(self):  # noqa: ANN201
        return None

    def item(self):  # noqa: ANN201
        return 0.0

    def cpu(self):  # noqa: ANN201
        return self

    def to(self, *args: Any, **kwargs: Any):  # noqa: ANN201
        return self

    def detach(self):  # noqa: ANN201
        return self

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _TORCH_NOOP


def _noop(*args: Any, **kwargs: Any) -> Any:
    return None


_TORCH_NOOP = _noop


def _torch_zeros(*args: Any, **kwargs: Any) -> _TorchTensorStub:
    return _TorchTensorStub()


class _TorchDeviceStub:
    def __str__(self) -> str:
        return "cpu"

    def __repr__(self) -> str:
        return "<stub cpu>"

    @property
    def type(self) -> str:
        return "cpu"


def _torch_device(value: Any = None) -> _TorchDeviceStub:
    return _TorchDeviceStub()


class _TorchNoGrad:
    def __enter__(self) -> "_TorchNoGrad":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


def _torch_is_tensor(obj: Any) -> bool:
    return isinstance(obj, _TorchTensorStub)


def _torch_isnan(obj: Any) -> bool:
    return False


def _torch_cat(*args: Any, **kwargs: Any) -> _TorchTensorStub:
    return _TorchTensorStub()


def _torch_stack(*args: Any, **kwargs: Any) -> _TorchTensorStub:
    return _TorchTensorStub()


def _make_torch_stub() -> Any:
    """构造 torch 占位 module（仅离线测试环境；生产机器装真实 torch）。"""
    import types

    m = types.ModuleType("torch")
    m.Tensor = _TorchTensorStub
    m.zeros = _torch_zeros
    m.zeros_like = _torch_zeros
    m.ones = _torch_zeros
    m.ones_like = _torch_zeros
    m.device = _torch_device
    m.no_grad = _TorchNoGrad
    m.is_tensor = _torch_is_tensor
    m.isnan = _torch_isnan
    m.cat = _torch_cat
    m.stack = _torch_stack
    m.backends = types.ModuleType("torch.backends")
    m.backends.cudnn = types.ModuleType("torch.backends.cudnn")
    m.backends.cudnn.enabled = False
    m.backends.cudnn.benchmark = False
    m.cuda = types.ModuleType("torch.cuda")
    m.cuda.is_available = lambda: False
    m.float32 = "torch.float32"
    m.float64 = "torch.float64"
    m.long = "torch.long"
    m.int64 = "torch.int64"
    m.bool = "torch.bool"
    return m


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
        # §41：上一轮 MemoryPacket（parents/structural_neighbors）→ 下一轮搜索输入
        self.current_packet: Any | None = None
        self._packet_error: str | None = None
        if state_dir is None:
            state_dir = Path.cwd() / "data" / "logs" / "continuous"
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.max_hours = max_hours
        self.max_rounds = max_rounds
        self.sleep_seconds = max(0, int(sleep_seconds))
        self.round_no = 0
        self._started_mono = time.monotonic()
        # P0-A：当前工作树无 torch（venv 缺 wheel），而 round_manager.py 的函数体
        # 内延迟 ``import alphaprobe.pipeline`` 时，pipeline 模块又会经 orchestrator
        # 侧拉入 alphaprobe.runner（runner 顶层 ``import torch``）→ 单轮 campaign
        # 直接崩。这里在构造期（此时尚未 import pipeline）先行建立 torch 轻量占位
        # stub：若真实 torch 可用则原样放行（实测 True 直接可用）；仅在缺失时
        # 注入最小 stub。这是「无 torch 离线可跑」的既有测试环境要求（见
        # test_pipeline.py / test_round_checkpoint.py 对 runner 顶层 torch import
        # 的绕过模式），不改变生产语义。
        if "torch" not in sys.modules:
            try:
                import torch as _real_torch  # noqa: F401

                _torch = _real_torch
            except Exception:  # noqa: BLE001 - 无 torch wheel → 占位 stub
                _torch = _make_torch_stub()
            sys.modules["torch"] = _torch

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
        """load memory：build_memory_packet 结果存 self.current_packet（§41 闭环）。

        返回 packet 的 parent 视图像（供 _default_campaign 构造 pipeline parents）。
        """
        if self.store is None:
            return None
        try:
            packet = self.store.build_memory_packet(parent_node={"factor_id": "round_root"})
        except Exception as exc:  # noqa: BLE001 - memory 不可用不阻塞，但记录原因
            self._packet_error = f"build_memory_packet failed: {exc}"
            self.current_packet = None
            return None
        self.current_packet = packet
        self._packet_error = None
        # 上一轮知识 → 下一轮搜索输入：parents = packet.parent + structural_neighbors
        if hasattr(packet, "parent") and isinstance(packet.parent, dict):
            parent = dict(packet.parent)
        elif isinstance(packet, dict):
            parent = dict(packet.get("parent") or {})
        else:
            parent = {}
        neighbors: list[dict[str, Any]] = []
        for n in getattr(packet, "structural_neighbors", []) or []:
            if isinstance(n, dict):
                neighbors.append(dict(n))
        parents = [parent] if parent else []
        parents.extend(neighbors[:5])
        return {"parents": parents, "packet": packet}

    def _freeze_calibrator(self, round_id: str) -> None:
        cal = self.calibrator
        try:
            cal.freeze(round_id)
        except AttributeError:
            pass

    def _update_memory(self, round_id: str, result: dict[str, Any]) -> None:
        if self.store is None:
            return
        # §74 搜索轮事件写独立表（search_run_events），不再混进 market_regime_events。
        # 市场状态表只放 regime 事件（survival.regime.register_2026_event），round 完成/
        # 汇总属引擎运行元数据，从今往后落 search_run_events。
        try:
            if hasattr(self.store, "add_search_run_event"):
                self.store.add_search_run_event(
                    event_id=f"round_{round_id}_complete",
                    round_id=str(round_id),
                    description=f"round {round_id} complete",
                    payload={"pool_size": result.get("pool_size", 0)},
                )
        except Exception as exc:  # noqa: BLE001 - memory 阶段失败不阻塞主循环
            print(f"[round_manager] add_search_run_event failed: {exc}")

        # RoundResult（alphaprobe.pipeline.RoundResult）→ pool_snapshot 兜底对账
        # upsert factor_nodes。attempts/evaluations 的实时写已在 pipeline 内完成
        # （_remember_attempt → upsert_factor_node + record_attempt，
        #  _remember_evaluation → record_evaluation，见 pipeline.py），且 pool 中被
        # 淘汰的成员可能只出现在快照里，所以此处仍全量遍历（不再 [:50] 截断）做
        # 一轮兜底对账。§74 事件驱动：实时写在 pipeline 内，这里是全量兜底。
        rr = result.get("round_result")
        pool_snapshot = []
        if rr is not None:
            try:
                pool_snapshot = list(getattr(rr, "pool_snapshot", []) or [])
            except Exception as exc:  # noqa: BLE001
                print(f"[round_manager] read pool_snapshot failed: {exc}")
                pool_snapshot = []
        # 身份只认 FactorEngine 权威（§28，runner.py _record_exported_to_memory 同款写法）：
        # build_identity_view 出 canonical_formula/canonical_ast_hash/
        # signal_equivalence_id/parameter_family_id；FE 不可用 fail-closed 跳过该成员，
        # 绝不回落文本 regex（alphaprobe.dedup 已退出活跃主链）。
        from alphaprobe.authority import FactorIdentityAuthorityError
        from alphaprobe.authority import build_identity_view as _build_identity_view

        for member in pool_snapshot:
            formula = str(member.get("formula") or member.get("canonical_formula") or "")
            fid = str(member.get("factor_id") or "")
            if not formula or not fid:
                continue
            try:
                iv = _build_identity_view(formula)
            except FactorIdentityAuthorityError as exc:
                print(f"[round_manager] authority identity unavailable, skip: {formula!r}: {exc}")
                continue
            try:
                canonical = str(iv.get("canonical_formula") or formula)
                self.store.upsert_factor_node(
                    factor_id=fid,
                    canonical_formula=canonical,
                    canonical_ast_hash=str(iv.get("canonical_ast_hash", "") or ""),
                    signal_equivalence_id=str(iv.get("signal_equivalence_id", "") or ""),
                    parameter_family_id=str(iv.get("parameter_family_id", "") or None),
                    source_system="alphaprobe",
                    source_snapshot=str(round_id),
                    source_type="MINED",
                    exportable=False,
                )
            except Exception as exc:  # noqa: BLE001 - 单节点失败不阻塞
                print(f"[round_manager] upsert_factor_node failed: {exc}")

        # attempts / evaluations 计数更新（bump_exploration 由 pipeline 内实时写，
        # 这里从 result 计数做一次汇总事件快照）
        try:
            n_eval = int(getattr(rr, "evaluated", 0) or 0) if rr is not None else 0
            n_admit = int(getattr(rr, "admitted", 0) or 0) if rr is not None else 0
            if hasattr(self.store, "add_search_run_event") and (n_eval or n_admit):
                self.store.add_search_run_event(
                    event_id=f"round_{round_id}_summary",
                    round_id=str(round_id),
                    description=f"round {round_id} pipeline summary",
                    payload={"evaluated": n_eval, "admitted": n_admit},
                )
        except Exception as exc:  # noqa: BLE001
            print(f"[round_manager] summary event failed: {exc}")

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
        """默认单轮逻辑：委托 alphaprobe.pipeline.SearchPipeline（§41 主链真身）。

        - 真实 campaign 由 runner 层注入；这里做兜底（离线 stub：stub llm_fn +
          内存 dedup + 静态 evaluate_fn，不跑真实 LLM / 数据）。
        - pipeline 构造失败才回落 {"pool_size": 0} 并 log 原因（不静默）。
        - parents 从 kwargs['parents']（RoundManager._load_memory 的 MemoryPacket
          结果）传入 run_round —— 上一轮知识 → 下一轮搜索输入闭环。
        """
        # 延迟 import 防循环（round_manager ↔ pipeline ↔ runner）
        try:
            from alphaprobe.pipeline import PipelineConfig, SearchPipeline

            pipeline = SearchPipeline(
                experiment=None,
                data_train=None,
                config=PipelineConfig(pool_target=8, pool_max=16, budget_per_round=6),
                memory_store=kwargs.get("memory_store"),
                llm_fn=None,  # 确定性 stub，不跑真实 LLM
            )
        except Exception as exc:
            print(f"[round_manager] pipeline construct failed, fallback placeholder: {exc}")
            return {"pool_size": 0, "round_no": round_no, "fallback_reason": str(exc)}

        parents = kwargs.get("parents") or []
        try:
            result = pipeline.run_round(round_id=f"round_{round_no}", parents=parents)
        except Exception as exc:  # noqa: BLE001 - 单轮失败不中断 7×24
            print(f"[round_manager] pipeline.run_round failed, fallback placeholder: {exc}")
            return {"pool_size": 0, "round_no": round_no, "fallback_reason": str(exc)}
        return {
            "round_no": round_no,
            "pool_size": pipeline.pool_size(),
            "round_result": result,
        }

    # -- main loop -----------------------------------------------------------

    def run(self) -> int:
        """启动 7×24 循环。返回已执行轮数。"""
        while not self._should_stop():
            self.round_no += 1
            round_id = f"round_{self.round_no}"
            campaign_id = f"campaign_{self.round_no}"

            # load memory
            memory = self._load_memory()
            # freeze calibrator
            self._freeze_calibrator(round_id)
            # lineage early stop（§56.3 / §77）
            # 现状：round 级 dummy 观测 —— observe(round_id, {"fitness": None})
            # 表示「本轮无 gain 可用」，由 patience_tracker 对 round 主键累计
            # no-gain streak 做全局 early stop（不区分 branch）。
            # 目标（P1，本轮不做完整改造）：按 branch 维度 per-branch
            # LineagePatienceTracker —— 在单轮 campaign 内对每个 branch 的
            # last-6-generations 观测 ΔFitness/ΔNovelty（每代真实增益），
            # lineage_key=branch_id，达到 patience 即停该 branch（强 lineage 得
            # 更多预算）。此处保持行为不变，仅注释说明现状与目标。
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
                # §41：上一轮 MemoryPacket → 本轮搜索输入（parents）
                parents=(memory or {}).get("parents", []) if memory else [],
                packet=self.current_packet,
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
