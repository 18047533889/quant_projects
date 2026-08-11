"""data_access.r30.experiment_snapshot —— R30-P0-009 ExperimentDataSnapshot。

实验（experiment）读到的数据身份由整条依赖链决定：每个数据集的 source snapshot、
日历快照、股票池快照、semantic contract、registry、代码 build SHA、安全 scope。
任一环改变，实验的 snapshot 身份就必须改变——否则缓存/结果复用会把「换了数据 /
代码」的实验当成同一个。

本模块的 ``ExperimentDataSnapshot`` 把整条链折叠成一个稳定 digest；datasets 的
source snapshot id 优先从 ``store.manifest_version(dataset)`` 的 epoch 字段构造，
否则 fallback 到 ``stable_digest(manifest_version(...))``。不修改任何既有文件。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from data_access.r30._shared import security_scope, stable_digest, stable_digest_full, version_gate
from data_access.r30.calendar_snapshot import CalendarSnapshot, canonical
from data_access.r30.universe_snapshot import UniverseSnapshot


def _build_sha() -> str | None:
    """当前代码 build SHA（env 优先，回退 git HEAD；无 git → None，调用方容忍）。"""
    try:
        from data_access._build_meta import build_sha

        return build_sha()
    except Exception:
        return None


def _safe_digest(store: Any, name: str) -> str | None:
    """调用 ``store.<name>()`` 取指纹；不存在 / 抛错 → None。"""
    fn = getattr(store, name, None)
    if not callable(fn):
        return None
    try:
        value = fn()
        return str(value) if value is not None else None
    except Exception:
        return None


def dataset_source_id(store: Any, dataset: str) -> str | None:
    """从 store 取某数据集的 source snapshot id。

    优先 ``store.manifest_version(dataset)``（返回 dict，取其中的快照/epoch 字段
    构造 id）；dict 无可用 epoch 字段或非 dict 时 fallback 到
    ``stable_digest(manifest_version(...))``；store 无法提供 → None（调用方回退
    caller 提供的 id）。
    """
    mv = None
    fn = getattr(store, "manifest_version", None)
    if callable(fn):
        try:
            mv = fn(dataset)
        except Exception:
            mv = None
    if isinstance(mv, dict) and mv:
        if mv.get("has_manifest"):
            gen = mv.get("manifest_generation_id")
            if gen:
                return f"{dataset}:gen:{gen}"
            token = (
                mv.get("source_epoch")
                or mv.get("manifest_epoch")
                or mv.get("manifest_built_epoch")
                or mv.get("dataset_version")
                or mv.get("partition_version")
            )
            if token is not None:
                return f"{dataset}:{token}"
        return f"{dataset}:{stable_digest(mv)}"
    if mv is not None:
        return f"{dataset}:{stable_digest(mv)}"
    return None


def datasets_source_ids(store: Any, datasets: Mapping[str, str]) -> dict[str, str]:
    """把 ``{dataset_name: caller_source_id}`` 解析成 ``{dataset_name: 稳定 source id}``。

    每个名字先问 store（manifest 的 epoch 字段）；store 提供不了 → 用调用方传入的
    id；再没有 → 用数据集名的稳定 digest（确定性占位）。
    """
    out: dict[str, str] = {}
    for name in sorted(datasets):
        sid = dataset_source_id(store, name)
        if sid is None:
            caller_val = datasets[name]
            sid = str(caller_val) if caller_val is not None else stable_digest(name)
        out[name] = sid
    return out


@dataclass(frozen=True)
class ExperimentDataSnapshot:
    """一次实验读取的**整条数据依赖链**快照身份（R30-P0-009）。"""

    experiment_id: str
    market: str | None
    datasets: Mapping[str, str]  # {dataset_name: source_snapshot_id}
    calendar_snapshot: CalendarSnapshot | None
    universe_snapshot: UniverseSnapshot | None
    semantic_contract_digest: str | None
    registry_digest: str | None
    code_build_sha: str | None
    security_scope_digest: str | None
    snapshot_id: str
    version_gate: Mapping[str, str]

    # ---- 构建 ----

    @classmethod
    def build(
        cls,
        store: Any,
        experiment_id: str,
        market: str | None,
        datasets: Mapping[str, str],
        calendar: "CalendarSnapshot | Any | None" = None,
        universe: "UniverseSnapshot | str | None" = None,
    ) -> "ExperimentDataSnapshot":
        """从 store 构建实验数据快照。

        - ``calendar``：已是 ``CalendarSnapshot`` 直接用；原始 ``MarketCalendar``
          则现场 ``CalendarSnapshot.build``；None → ``CalendarSnapshot.from_store``；
        - ``universe``：已是 ``UniverseSnapshot`` 直接用；str（universe_id）→
          ``UniverseSnapshot.from_store``；None → 不绑定 universe；
        - ``datasets``：{name: caller_id}，实际 source id 由 store 解析。
        """
        experiment_id = str(experiment_id)
        market_s = str(market) if market is not None else None

        # calendar
        cal_snap: CalendarSnapshot | None = (
            calendar if isinstance(calendar, CalendarSnapshot) else None
        )
        if cal_snap is None and calendar is not None:
            cal_snap = CalendarSnapshot.build(market_s, calendar)
        if cal_snap is None and market_s:
            cal_snap = CalendarSnapshot.from_store(store, market_s)

        # universe
        uni_snap: UniverseSnapshot | None = (
            universe if isinstance(universe, UniverseSnapshot) else None
        )
        if uni_snap is None and isinstance(universe, str) and universe.strip():
            uni_snap = UniverseSnapshot.from_store(store, universe.strip())

        ds_ids = datasets_source_ids(store, datasets)

        semantic = _safe_digest(store, "contract_ir_fingerprint") or _safe_digest(
            store, "registry_fingerprint"
        )
        registry = _safe_digest(store, "registry_fingerprint")
        build_sha = _build_sha()
        sec_scope = security_scope(store)
        vg = version_gate()

        snapshot_id = stable_digest_full(
            canonical(experiment_id),
            canonical(market_s),
            canonical(tuple(sorted(f"{k}={v}" for k, v in ds_ids.items()))),
            canonical(cal_snap.snapshot_id if cal_snap is not None else None),
            canonical(uni_snap.snapshot_id if uni_snap is not None else None),
            canonical(semantic),
            canonical(registry),
            canonical(build_sha),
            canonical(sec_scope),
            canonical(vg),
        )
        return cls(
            experiment_id=experiment_id,
            market=market_s,
            datasets=dict(ds_ids),
            calendar_snapshot=cal_snap,
            universe_snapshot=uni_snap,
            semantic_contract_digest=semantic,
            registry_digest=registry,
            code_build_sha=build_sha,
            security_scope_digest=sec_scope,
            snapshot_id=snapshot_id,
            version_gate=dict(vg),
        )

    # ---- 访问 ----

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "experiment_id": self.experiment_id,
            "market": self.market,
            "datasets": dict(self.datasets),
            "calendar_snapshot": (
                self.calendar_snapshot.to_dict()
                if self.calendar_snapshot is not None
                else None
            ),
            "universe_snapshot": (
                self.universe_snapshot.to_dict()
                if self.universe_snapshot is not None
                else None
            ),
            "semantic_contract_digest": self.semantic_contract_digest,
            "registry_digest": self.registry_digest,
            "code_build_sha": self.code_build_sha,
            "security_scope_digest": self.security_scope_digest,
            "version_gate": dict(self.version_gate),
        }


def current_experiment_snapshot(
    store: Any,
    experiment_id: str,
    market: str | None,
    datasets: Mapping[str, str],
) -> ExperimentDataSnapshot:
    """便捷入口：从 store 构建当前实验的完整数据快照。"""
    return ExperimentDataSnapshot.build(store, experiment_id, market, datasets)


__all__ = [
    "ExperimentDataSnapshot",
    "current_experiment_snapshot",
    "dataset_source_id",
    "datasets_source_ids",
]
