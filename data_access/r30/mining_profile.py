"""
data_access.r30.mining_profile —— R30-P1-022/023 MiningFieldProfile + FieldCapabilityCatalog

R30 全维度成熟度层是 **additive** 的。本模块为 LLM / AlphaProbe 提供**合法的 field
space**：

    - ``MiningFieldProfile``：单个逻辑字段的挖掘画像——``mining_allowed``、
      ``pit_fidelity``、``coverage_class``、``cost_class``、频率/粒度、推荐/禁止
      算子族。``from_semantic`` 从 ``read/semantic_catalog.SemanticField`` 编译，
      默认规则：普通 daily panel 字段可挖（True），但 snapshot_only / event /
      financial_event 类默认 False；sparse event / snapshot-only 字段**禁止**普通
      daily panel 滚 252；
    - ``FieldCapabilityCatalog``：行式索引，``search`` 按 concept / market /
      pit_min / coverage_min / cost_max 过滤，供生成表达式时先落在合法子空间。

**不修改** ``read/semantic_catalog.py`` / registry。``from_store`` 是防御性遍历：
registry 或 semantic catalog 缺失时返回空目录，绝不抛。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


# ---- 等级标尺（search 过滤用） ----
_PIT_RANK = {
    "unsupported": 0,
    "effective_only": 1,
    "knowledge_date_pit": 3,
    "vintage_pit": 4,
}
_COVERAGE_RANK = {"low": 1, "limited": 2, "medium": 3, "good": 4, "high": 5}
_COST_RANK = {"cheap": 1, "medium": 2, "high": 3, "expensive": 4}

# 事件/快照类 temporal_model 与 grain → 默认禁挖 + 禁 daily panel 滚 252
_EVENT_TEMPORAL_MODELS = frozenset({"event", "sparse_event", "financial_event", "financial", "e2"})
_SNAPSHOT_GRAINS = frozenset({"snapshot", "sparse_event", "event"})
_FORBIDDEN_DAILY_PANEL = ("daily_panel", "rolling_window_252", "ts_rolling")


# ---------------------------------------------------------------------------
# MiningFieldProfile
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MiningFieldProfile:
    concept_id: str
    mining_allowed: bool = True
    pit_fidelity: str = "unsupported"
    coverage_class: str = "low"
    cost_class: str = "medium"
    frequency: str | None = None
    grain: str | None = None
    recommended_operator_families: tuple[str, ...] = ()
    forbidden_operator_families: tuple[str, ...] = ()

    @classmethod
    def from_semantic(
        cls, store: Any, dataset: str | None, field: Any
    ) -> "MiningFieldProfile":
        """从 SemanticField（对象或 dict）编译挖掘画像。

        规则：
            - ``mining_allowed``：显式声明优先；否则 financial/sparse_event/snapshot
              类默认 False，其余默认 True；
            - ``pit_fidelity``：显式声明优先；financial_event → knowledge_date_pit；
              event/snapshot → effective_only；其余 unsupported；
            - sparse event / snapshot-only → 禁止普通 daily panel 滚 252。
        """
        f = _as_field(field)
        pit = _pit_fidelity(f)
        mining = _mining_allowed(f)
        cost = _cost_class(f)
        cov = _coverage_class(pit)
        rec, forb = _operator_families(f, mining)
        concept = (
            f.get("logical_name")
            or f.get("physical_name")
            or dataset
            or "unknown"
        )
        return cls(
            concept_id=str(concept),
            mining_allowed=mining,
            pit_fidelity=pit,
            coverage_class=cov,
            cost_class=cost,
            frequency=f.get("frequency"),
            grain=f.get("grain"),
            recommended_operator_families=tuple(rec),
            forbidden_operator_families=tuple(forb),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "mining_allowed": self.mining_allowed,
            "pit_fidelity": self.pit_fidelity,
            "coverage_class": self.coverage_class,
            "cost_class": self.cost_class,
            "frequency": self.frequency,
            "grain": self.grain,
            "recommended_operator_families": list(self.recommended_operator_families),
            "forbidden_operator_families": list(self.forbidden_operator_families),
        }


def _as_field(field: Any) -> dict[str, Any]:
    if isinstance(field, Mapping):
        return dict(field)
    return {
        "logical_name": getattr(field, "logical_name", None),
        "physical_name": getattr(field, "physical_name", None),
        "dataset": getattr(field, "dataset", None),
        "market": getattr(field, "market", None),
        "dtype": getattr(field, "dtype", None),
        "frequency": getattr(field, "frequency", None),
        "grain": getattr(field, "grain", None),
        "temporal_model": getattr(field, "temporal_model", None),
        "pit_fidelity": getattr(field, "pit_fidelity", None),
        "mining_allowed": getattr(field, "mining_allowed", None),
        "source_unit": getattr(field, "source_unit", None),
        "canonical_unit": getattr(field, "canonical_unit", None),
        "currency": getattr(field, "currency", None),
        "dimension": getattr(field, "dimension", None),
    }


def _pit_fidelity(f: Mapping[str, Any]) -> str:
    pf = f.get("pit_fidelity")
    if pf:
        return str(pf)
    temporal = str(f.get("temporal_model") or "").lower()
    grain = str(f.get("grain") or "").lower()
    if temporal in {"financial", "financial_event", "e2"}:
        return "knowledge_date_pit"
    if temporal in {"event", "sparse_event"} or grain in _SNAPSHOT_GRAINS:
        return "effective_only"
    return "unsupported"


def _mining_allowed(f: Mapping[str, Any]) -> bool:
    explicit = f.get("mining_allowed")
    if explicit is not None:
        return bool(explicit)
    temporal = str(f.get("temporal_model") or "").lower()
    grain = str(f.get("grain") or "").lower()
    if temporal in _EVENT_TEMPORAL_MODELS or grain in _SNAPSHOT_GRAINS:
        return False
    return True


def _cost_class(f: Mapping[str, Any]) -> str:
    freq = str(f.get("frequency") or "").lower()
    grain = str(f.get("grain") or "").lower()
    if freq in {"tick", "micro", "1tick"} or grain in {"tick", "micro"}:
        return "expensive"
    if freq in {"minute", "1min", "5min", "15min", "intraday"}:
        return "high"
    if freq in {"daily", "day", "1d"}:
        return "medium"
    if freq in {"weekly", "monthly", "quarterly", "yearly", "annual", "semiannual"}:
        return "cheap"
    return "medium"


def _coverage_class(pit: str) -> str:
    if pit in {"knowledge_date_pit", "vintage_pit"}:
        return "good"
    if pit == "effective_only":
        return "limited"
    return "low"


def _operator_families(
    f: Mapping[str, Any], mining_allowed: bool
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    temporal = str(f.get("temporal_model") or "").lower()
    grain = str(f.get("grain") or "").lower()
    freq = str(f.get("frequency") or "").lower()
    if not mining_allowed:
        return (), tuple(_FORBIDDEN_DAILY_PANEL) + ("cross_sectional",)
    if temporal in _EVENT_TEMPORAL_MODELS or grain in _SNAPSHOT_GRAINS:
        # sparse event / snapshot-only：禁普通 daily panel 滚 252
        return (
            ("latest_period", "pit_asof", "cross_sectional", "point_in_time"),
            tuple(_FORBIDDEN_DAILY_PANEL),
        )
    if freq in {"daily", "day", "1d"}:
        return ("rolling", "cross_sectional", "ts", "ratio"), ()
    if freq in {"quarterly", "yearly", "annual", "semiannual", "monthly"}:
        return ("latest_period", "pit_asof", "cross_sectional", "financial_ratio"), ()
    return ("cross_sectional", "latest_period"), ()


# ---------------------------------------------------------------------------
# FieldCapabilityCatalog
# ---------------------------------------------------------------------------
def _get_semantic_catalog():
    from data_access.read.semantic_catalog import get_semantic_catalog

    return get_semantic_catalog()


def _infer_market(dataset: str | None) -> str:
    name = str(dataset or "").strip().lower()
    if name.startswith("ashare") or name.startswith("a_"):
        return "ashare"
    if name.startswith("us") or name.startswith("am"):
        return "us"
    return "any"


def _source_availability(profile: MiningFieldProfile) -> str:
    return "usable" if profile.mining_allowed else "restricted"


def _row_from_profile(
    profile: MiningFieldProfile,
    field: Mapping[str, Any] | None = None,
    *,
    dataset: str | None = None,
) -> dict[str, Any]:
    market = None
    unit = None
    if field is not None:
        market = field.get("market") or _infer_market(field.get("dataset"))
        unit = (
            field.get("canonical_unit")
            or field.get("source_unit")
            or field.get("currency")
        )
    return {
        "concept": profile.concept_id,
        "market": market or _infer_market(dataset or profile.concept_id),
        "dataset": dataset or (field.get("dataset") if field else None),
        "frequency": profile.frequency,
        "grain": profile.grain,
        "unit": unit,
        "pit": profile.pit_fidelity,
        "coverage": profile.coverage_class,
        "cost_class": profile.cost_class,
        "mining_allowed": profile.mining_allowed,
        "source_availability": _source_availability(profile),
    }


def _registry_names(reg: Any) -> list[str]:
    for attr in ("names", "get_all"):
        fn = getattr(reg, attr, None)
        if callable(fn):
            try:
                names = fn()
                return list(names)
            except Exception:
                continue
    return []


def _registry_get(reg: Any, name: str) -> Any:
    try:
        return reg.get(name)
    except Exception:
        return None


class FieldCapabilityCatalog:
    """字段能力行式索引：LLM/AlphaProbe 在合法 field space 内生成表达式。"""

    def __init__(self, rows: Sequence[Mapping[str, Any]] | None = None) -> None:
        self._rows = [dict(r) for r in (rows or [])]

    def rows(self) -> list[dict[str, Any]]:
        return list(self._rows)

    def search(
        self,
        concept: str | None = None,
        market: str | None = None,
        pit_min: int | None = None,
        coverage_min: int | None = None,
        cost_max: int | None = None,
    ) -> list[dict[str, Any]]:
        """按概念/市场/PIT 下限/覆盖下限/成本上限过滤。

        ``pit_min`` 取值 0-4（unsupported/effective_only/.../vintage_pit）；
        ``coverage_min`` 1-5（low/limited/medium/good/high）；
        ``cost_max`` 1-4（cheap/medium/high/expensive）。
        """
        out: list[dict[str, Any]] = []
        for r in self._rows:
            if concept is not None and concept not in str(r.get("concept", "")):
                continue
            if market is not None and str(r.get("market", "any")) not in {
                str(market),
                "any",
            }:
                continue
            if pit_min is not None and _PIT_RANK.get(r.get("pit"), 0) < pit_min:
                continue
            if (
                coverage_min is not None
                and _COVERAGE_RANK.get(r.get("coverage"), 0) < coverage_min
            ):
                continue
            if (
                cost_max is not None
                and _COST_RANK.get(r.get("cost_class"), 5) > cost_max
            ):
                continue
            out.append(r)
        return out

    @classmethod
    def from_store(cls, store: Any) -> "FieldCapabilityCatalog":
        """从 store 遍历 registry + semantic catalog 构建。

        防御性：registry / semantic catalog 缺失或遍历失败时返回空目录，绝不抛。
        """
        rows: list[dict[str, Any]] = []
        reg = getattr(store, "_registry", None) or getattr(store, "registry", None)
        catalog = None
        try:
            catalog = _get_semantic_catalog()
        except Exception:
            catalog = None
        if catalog is not None:
            try:
                names = catalog.names() if callable(getattr(catalog, "names", None)) else []
            except Exception:
                names = []
            for name in names:
                try:
                    f = catalog.resolve_one(name)
                except Exception:
                    f = None
                if f is None:
                    continue
                try:
                    profile = MiningFieldProfile.from_semantic(
                        store, getattr(f, "dataset", None) or name, f
                    )
                except Exception:
                    continue
                rows.append(_row_from_profile(profile, _as_field(f)))
        elif reg is not None:
            # 无 semantic catalog：回退 registry 数据集 schema 物理列
            for ds_name in _registry_names(reg):
                ds = _registry_get(reg, ds_name)
                if ds is None:
                    continue
                schema = getattr(ds, "schema", None) or {}
                if not isinstance(schema, Mapping):
                    continue
                semantic = getattr(ds, "semantic", None)
                for col in schema:
                    try:
                        profile = MiningFieldProfile.from_semantic(
                            store,
                            ds_name,
                            {
                                "logical_name": col,
                                "physical_name": col,
                                "dataset": ds_name,
                                "market": _infer_market(ds_name),
                                "frequency": None,
                                "grain": None,
                                "temporal_model": semantic,
                                "mining_allowed": None,
                            },
                        )
                    except Exception:
                        continue
                    rows.append(
                        _row_from_profile(profile, None, dataset=ds_name)
                    )
        return cls(rows)


__all__ = [
    "MiningFieldProfile",
    "FieldCapabilityCatalog",
]
