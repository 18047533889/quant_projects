"""
data_access.read.semantic_catalog —— 字段语义单一事实源（SemanticFieldCatalog）

职责
    1. 把「逻辑字段名 → 物理数据集 + 物理列」的映射集中管理，作为 FactorEngine
       FIELD_REGISTRY 与 DataAccess schema/COS 契约之上的统一语义层。
    2. 每个字段声明 dtype / frequency / grain / source_unit / canonical_unit /
       scale（单位归一化乘子）/ 时间语义角色 / PIT·join 策略 / aliases /
       mining_allowed。
    3. 提供 ``scale_to_canonical`` 的输出层归一化（``normalize_table_units``），
       使 Return(bp)、ROE(%)、TurnoverRatio(%) 等字段在 DataAccess 输出层统一
       变成 canonical 单位，FactorEngine 不再二次修正。

设计要点
    1. 配置放 ``config/semantic_fields.yaml``；进程内惰性加载并缓存。
    2. 解析时若逻辑名不在 catalog，store.resolve_fields 会回退到 registry 的
       dataset schema（把物理列名当逻辑名），保证「物理列 == 逻辑名」的老代码
       不受影响。
    3. 单位归一化是显式行为：``normalize_units=True`` 时对浮点列乘 scale；
       默认 False，保持与旧 read 路径逐字节一致。
    4. scale 的语义：从 source_unit 换算到 canonical_unit 的乘子
       （percent→ratio 为 0.01；bp→ratio 为 0.0001；无换算为 None 或 1.0）。

非职责
    不做文件 IO；不解析 datasets.yaml（registry）；不替代 COS PIT 契约（那套
    是单数据集的时间可见性语义，这里是跨数据集字段语义的权威登记处）。

维护人：quant 基础平台组    最后更新：2026-08-07
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from data_access.core.exceptions import ValidationError

try:
    import yaml  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    yaml = None


@dataclass(frozen=True)
class SemanticField:
    """一个逻辑字段的完整语义声明（catalog 的原子条目）。"""

    logical_name: str
    dataset: str                       # 物理数据集（registry 名）
    physical_name: str                 # 数据集内的物理列名
    dtype: str | None = None           # float64 / int64 / string / date ...
    frequency: str | None = None       # daily / minute / tick / quarterly ...
    grain: str | None = None           # instrument / cross_section / snapshot ...
    source_unit: str | None = None     # percent / bp / yuan / yuan_mm ...
    canonical_unit: str | None = None  # ratio / yuan ...
    scale: float | None = None         # source → canonical 的乘子（None 表示不归一化）
    time_role: str | None = None       # event_time / knowledge_time / effective_time ...
    temporal_model: str | None = None  # panel / financial_event / sparse_event ...
    knowledge_time: str | None = None  # 数据可见时间列（如 PubDate）
    effective_time: str | None = None  # 数据生效时间列（如 ReportPeriodEndDate）
    period_time: str | None = None     # 会计期间列（别名 ReportPeriodEndDate）
    join_policy: str | None = None     # exact / pit_asof_backward / latest_period ...
    required_filters: tuple[str, ...] = ()   # 读取时必须附带的条件（如行业过滤）
    aliases: tuple[str, ...] = ()            # 其它叫法（含 FactorEngine 里的别名）
    mining_allowed: bool = True

    @property
    def is_scale_applicable(self) -> bool:
        """该字段需要输出层单位归一化（scale 存在且 != 1.0）。"""
        return self.scale is not None and self.scale != 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_name": self.logical_name,
            "dataset": self.dataset,
            "physical_name": self.physical_name,
            "dtype": self.dtype,
            "frequency": self.frequency,
            "grain": self.grain,
            "source_unit": self.source_unit,
            "canonical_unit": self.canonical_unit,
            "scale": self.scale,
            "time_role": self.time_role,
            "temporal_model": self.temporal_model,
            "knowledge_time": self.knowledge_time,
            "effective_time": self.effective_time,
            "period_time": self.period_time,
            "join_policy": self.join_policy,
            "required_filters": list(self.required_filters),
            "aliases": list(self.aliases),
            "mining_allowed": self.mining_allowed,
        }


def _tuple_of(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value if v is not None)
    return ()


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool_or_default(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    return bool(value)


def parse_semantic_field(name: str, raw: dict[str, Any]) -> SemanticField:
    """把 YAML 单条声明解析成 SemanticField。``dataset`` / ``physical_name`` 必填。"""
    context = f"semantic field '{name}'"
    dataset = raw.get("dataset")
    physical = raw.get("physical_name") or raw.get("physical") or name
    if not dataset:
        raise ValidationError(f"{context}: 缺少 dataset（逻辑字段必须落到某个数据集）")
    return SemanticField(
        logical_name=name,
        dataset=str(dataset),
        physical_name=str(physical),
        dtype=_str_or_none(raw.get("dtype")),
        frequency=_str_or_none(raw.get("frequency")),
        grain=_str_or_none(raw.get("grain")),
        source_unit=_str_or_none(raw.get("source_unit")),
        canonical_unit=_str_or_none(raw.get("canonical_unit")),
        scale=_float_or_none(raw.get("scale")),
        time_role=_str_or_none(raw.get("time_role")),
        temporal_model=_str_or_none(raw.get("temporal_model")),
        knowledge_time=_str_or_none(raw.get("knowledge_time")),
        effective_time=_str_or_none(raw.get("effective_time")),
        period_time=_str_or_none(raw.get("period_time")),
        join_policy=_str_or_none(raw.get("join_policy")),
        required_filters=_tuple_of(raw.get("required_filters")),
        aliases=_tuple_of(raw.get("aliases")),
        mining_allowed=_bool_or_default(raw.get("mining_allowed"), True),
    )


class SemanticFieldCatalog:
    """逻辑字段 → 物理字段的权威映射表。"""

    def __init__(
        self,
        fields: Mapping[str, SemanticField] | None = None,
        *,
        source_path: str | Path | None = None,
    ) -> None:
        self._fields: dict[str, SemanticField] = dict(fields or {})
        self._by_name: dict[str, SemanticField] = {}
        for name, f in self._fields.items():
            self._by_name.setdefault(name, f)
            for alias in f.aliases:
                self._by_name.setdefault(alias, f)
        self._source_path = str(source_path) if source_path is not None else None

    @property
    def source_path(self) -> str | None:
        return self._source_path

    def names(self) -> list[str]:
        return sorted(self._fields)

    def resolve_one(self, name: str) -> SemanticField | None:
        """按逻辑名或别名解析；找不到返回 None（调用方可再回退 registry）。"""
        return self._by_name.get(name)

    def resolve_by_physical(self, dataset: str, physical_name: str) -> SemanticField | None:
        """按 (dataset, 物理列名) 反查逻辑字段（调用方传物理列时的单位归一化用）。

        同一物理列可能被多个逻辑字段引用，返回第一个。
        """
        for f in self._fields.values():
            if f.dataset == dataset and f.physical_name == physical_name:
                return f
        return None

    def get(self, name: str) -> SemanticField:
        """严格解析；找不到抛 ValidationError。"""
        f = self.resolve_one(name)
        if f is None:
            raise ValidationError(
                f"逻辑字段 '{name}' 不在 SemanticFieldCatalog 中。"
                f"可用: {self.names()}"
            )
        return f

    def resolve(self, *names: str) -> list[SemanticField]:
        out: list[SemanticField] = []
        for name in names:
            out.append(self.get(name))
        return out

    def to_dict(self) -> dict[str, Any]:
        return {name: f.to_dict() for name, f in self._fields.items()}

    @classmethod
    def from_yaml(
        cls, path: str | Path | None = None
    ) -> "SemanticFieldCatalog":
        """从 YAML 加载。path 缺省用 ``data_access/config/semantic_fields.yaml``。

        YAML 顶层是 mapping（逻辑字段名 → 声明），``_`` 前缀的键当作锚点模板跳过。
        """
        if path is None:
            path = (
                Path(__file__).resolve().parent.parent
                / "config"
                / "semantic_fields.yaml"
            )
        path = Path(path)
        if not path.exists():
            raise ValidationError(f"semantic_fields.yaml 不存在：{path}")
        if yaml is None:
            raise ValidationError("缺少 PyYAML 依赖，无法加载 SemanticFieldCatalog")
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict):
            raise ValidationError(
                f"{path}: 顶层必须是 mapping（字段名 → 声明）"
            )
        fields: dict[str, SemanticField] = {}
        for name, body in raw.items():
            if name.startswith("_"):
                continue  # YAML 锚点模板键
            if body is None:
                body = {}
            if not isinstance(body, dict):
                raise ValidationError(
                    f"{path}: 字段 '{name}' 的声明必须是 mapping"
                )
            fields[name] = parse_semantic_field(name, body)
        return cls(fields, source_path=path)


# ---- 进程内缓存（惰性加载） ----

_catalog: SemanticFieldCatalog | None = None
_catalog_lock = threading.Lock()


def get_semantic_catalog(
    path: str | Path | None = None,
) -> SemanticFieldCatalog:
    """进程级 SemanticFieldCatalog 单例。

    默认读 ``config/semantic_fields.yaml``；可用环境变量 ``DATA_ACCESS_SEMANTIC_FIELDS``
    覆盖路径（部署/测试用）。显式传 ``path`` 时不缓存。
    """
    global _catalog
    if path is None:
        env_path = os.environ.get("DATA_ACCESS_SEMANTIC_FIELDS")
        if env_path and not _catalog:
            path = env_path
    if _catalog is not None and path is None:
        return _catalog
    with _catalog_lock:
        if _catalog is not None and path is None:
            return _catalog
        built = SemanticFieldCatalog.from_yaml(path)
        if path is None:
            _catalog = built
        return built


def reset_semantic_catalog() -> None:
    """清空缓存（测试用）。"""
    global _catalog
    _catalog = None


def normalize_table_units(
    table: Any,
    fields: Sequence[SemanticField],
    *,
    column_of: Sequence[SemanticField] | None = None,
) -> Any:
    """输出层单位归一化：把需要 scale 的字段乘上乘子，返回新 Arrow Table。

    只对浮点（含整型）数值列生效；找不到列的字段静默跳过。``column_of``
    预留：当输出列名与 physical_name 不同时，可传「该列对应哪个 SemanticField」，
    按 logical_name 匹配列名（read_joined 的多表输出就是 physical_name）。
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    if not fields:
        return table
    needs = [f for f in fields if f.is_scale_applicable]
    if not needs:
        return table
    target_names = {f.physical_name for f in needs}
    out_arrays: dict[str, Any] = {}
    for i, col in enumerate(table.column_names):
        arr = table.column(i)
        applied = False
        if col in target_names:
            for f in needs:
                if f.physical_name != col:
                    continue
                if pa.types.is_floating(arr.type):
                    arr = pc.multiply(arr, pa.scalar(float(f.scale)))
                elif pa.types.is_integer(arr.type):
                    arr = pc.multiply(arr.cast(pa.float64()), pa.scalar(float(f.scale)))
                applied = True
                break
        out_arrays[col] = arr
    return pa.table(out_arrays)
