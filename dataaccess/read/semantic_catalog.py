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

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from data_access.core.exceptions import ValidationError

logger = logging.getLogger("data_access.semantic_catalog")

try:
    import yaml  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    yaml = None


@dataclass(frozen=True)
class SemanticField:
    """一个逻辑字段的完整语义声明（catalog 的原子条目）。"""

    logical_name: str
    dataset: str | None = None         # 物理数据集（registry 名）；#P0-24 derived 字段可无
    physical_name: str | None = None   # 数据集内的物理列名（derived 字段为表达式描述）
    market: str = "any"                # ashare / us / any（any=跨市场通用）
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
    revision_order: tuple[str, ...] = ()     # join 前按 (instrument, knowledge_time) 去重的版本排序列
    availability: str = "same_day"           # same_day / next_trading_day（PIT 可见性）
    primary_key: tuple[str, ...] = ()        # exact join 唯一性契约（如 (TradeDate, Symbol)）
    duplicate_policy: str = "latest_revision"  # keep_first / keep_last / latest_revision / error
    aliases: tuple[str, ...] = ()            # 其它叫法（含 FactorEngine 里的别名）
    mining_allowed: bool = True
    period_selection: str = "all"            # latest_period/exact_period/annual/quarterly/ttm/all
    period_values: tuple[Any, ...] = ()      # exact_period 的目标 period 值
    # #P0-24 derived 字段：不是某张物理表的列，而是多表/多列的派生表达式。
    # derived_from 列出依赖的 (dataset.column)；coverage audit 对 derived 字段跳过
    # COLUMN_MISSING 检查（它们不落盘）。
    derived_expression: str | None = None
    derived_from: tuple[str, ...] = ()

    @property
    def is_scale_applicable(self) -> bool:
        """该字段需要输出层单位归一化（scale 存在且 != 1.0）。"""
        return self.scale is not None and self.scale != 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_name": self.logical_name,
            "dataset": self.dataset,
            "physical_name": self.physical_name,
            "market": self.market,
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
            "revision_order": list(self.revision_order),
            "availability": self.availability,
            "primary_key": list(self.primary_key),
            "duplicate_policy": self.duplicate_policy,
            "aliases": list(self.aliases),
            "mining_allowed": self.mining_allowed,
            "period_selection": self.period_selection,
            "period_values": list(self.period_values),
            "derived_expression": self.derived_expression,
            "derived_from": list(self.derived_from),
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


# #31 合法枚举集合
_VALID_AVAILABILITY = {"same_day", "next_trading_day", "session"}
_VALID_DUPLICATE_POLICIES = {"keep_first", "keep_last", "latest_revision", "error"}
_VALID_PERIOD_SELECTIONS = {
    "latest_period",
    "exact_period",
    "annual",
    "quarterly",
    "ttm",
    "all",
}
_VALID_MARKETS = {"any", "ashare", "us"}


def _strict_bool(value: Any, *, context: str, default: bool) -> bool:
    """#31 严格 bool 解析：``"false"`` 字符串 → False；非法值报错。"""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "on"}:
            return True
        if text in {"false", "0", "no", "off", ""}:
            return False
    raise ValidationError(f"{context}: 非法布尔值 {value!r}（应为 true/false）")


def _strict_scale(value: Any, *, context: str) -> float | None:
    """#31 非法 scale 直接报错，禁止静默变 None（单位换算语义不容丢失）。"""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except (TypeError, ValueError):
            pass
    raise ValidationError(f"{context}: 非法 scale {value!r}（应为数值）")


def _strict_enum(
    value: Any,
    *,
    allowed: frozenset[str],
    context: str,
    default: str,
) -> str:
    """#31 枚举字段严格校验；非法值报错。"""
    if value is None:
        return default
    text = str(value).strip().lower()
    if text not in allowed:
        raise ValidationError(
            f"{context}: {text!r} 不在合法集合 {sorted(allowed)} 内"
        )
    return text


def parse_semantic_field(name: str, raw: dict[str, Any]) -> SemanticField:
    """把 YAML 单条声明解析成 SemanticField。``dataset`` / ``physical_name`` 必填。

    ``logical_name`` 可显式覆盖（用于同一逻辑字段的跨市场条目：key 用
    ``us_total_assets`` 保证 YAML key 唯一，logical_name 声明为 ``total_assets``
    让 catalog 的 _by_name 同时挂 A股/美股两个候选）。
    """
    context = f"semantic field '{name}'"
    dataset = raw.get("dataset")
    physical = raw.get("physical_name") or raw.get("physical") or name
    logical = str(raw.get("logical_name") or name)
    derived_expr = _str_or_none(raw.get("derived_expression"))
    # #P0-24 derived 字段不落在单一物理表：允许无 dataset，但必须有 derived_expression。
    if not dataset and not derived_expr:
        raise ValidationError(f"{context}: 缺少 dataset（逻辑字段必须落到某个数据集）")
    temporal_model = _str_or_none(raw.get("temporal_model"))
    # #9：财务事件字段默认 latest_period——「最新可见会计期 + 该期内最新修订」。
    # 未显式写 period_selection 时，financial_event/event 字段默认 latest_period，
    # 除非用户明确要 exact_period/annual/quarterly/all。
    period_selection = _str_or_none(raw.get("period_selection"))
    if period_selection is None:
        period_selection = (
            "latest_period"
            if temporal_model in {"financial_event", "event", "financial", "e2"}
            else "all"
        )
    # #31 strict schema：未知 key 拒绝、bool/enum/scale 严格校验。
    _KNOWN_KEYS = {
        "dataset",
        "physical_name",
        "physical",
        "logical_name",
        "market",
        "dtype",
        "frequency",
        "grain",
        "source_unit",
        "canonical_unit",
        "scale",
        "time_role",
        "temporal_model",
        "knowledge_time",
        "effective_time",
        "period_time",
        "join_policy",
        "required_filters",
        "revision_order",
        "availability",
        "primary_key",
        "duplicate_policy",
        "aliases",
        "mining_allowed",
        "period_selection",
        "period_values",
        "derived_expression",
        "derived_from",
    }
    unknown = sorted(set(raw) - _KNOWN_KEYS)
    if unknown:
        raise ValidationError(
            f"{context}: 未知配置 key {unknown}（应为 {sorted(_KNOWN_KEYS)} 之一）"
        )
    market = _str_or_none(raw.get("market")) or "any"
    if market not in _VALID_MARKETS:
        raise ValidationError(f"{context}: market={market!r} 非法（应为 any/ashare/us）")
    availability = _strict_enum(
        raw.get("availability"),
        allowed=_VALID_AVAILABILITY,
        context=context,
        default="same_day",
    )
    duplicate_policy = _strict_enum(
        raw.get("duplicate_policy"),
        allowed=_VALID_DUPLICATE_POLICIES,
        context=context,
        default="latest_revision",
    )
    period_selection = _strict_enum(
        period_selection,
        allowed=_VALID_PERIOD_SELECTIONS,
        context=context,
        default="all",
    )
    return SemanticField(
        logical_name=logical,
        dataset=str(dataset) if dataset else None,
        physical_name=str(physical),
        market=market,
        dtype=_str_or_none(raw.get("dtype")),
        frequency=_str_or_none(raw.get("frequency")),
        grain=_str_or_none(raw.get("grain")),
        source_unit=_str_or_none(raw.get("source_unit")),
        canonical_unit=_str_or_none(raw.get("canonical_unit")),
        scale=_strict_scale(raw.get("scale"), context=context),
        time_role=_str_or_none(raw.get("time_role")),
        temporal_model=temporal_model,
        knowledge_time=_str_or_none(raw.get("knowledge_time")),
        effective_time=_str_or_none(raw.get("effective_time")),
        period_time=_str_or_none(raw.get("period_time")),
        join_policy=_str_or_none(raw.get("join_policy")),
        required_filters=_tuple_of(raw.get("required_filters")),
        revision_order=_tuple_of(raw.get("revision_order")),
        availability=availability,
        primary_key=_tuple_of(raw.get("primary_key")),
        duplicate_policy=duplicate_policy,
        aliases=_tuple_of(raw.get("aliases")),
        mining_allowed=_strict_bool(
            raw.get("mining_allowed"), context=context, default=True
        ),
        period_selection=period_selection,
        period_values=_tuple_of(raw.get("period_values")) if raw.get("period_values") else (),
        derived_expression=_str_or_none(raw.get("derived_expression")),
        derived_from=_tuple_of(raw.get("derived_from")),
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
        # _by_name：logical_name / YAML key / alias → 候选字段列表。同一逻辑名可被
        # 多个市场占用（如 A股 total_assets 与美股 us_total_assets 的 logical_name
        # 都是 total_assets），resolve 时按 market 消歧。any 表示跨市场通用。
        self._by_name: dict[str, list[SemanticField]] = {}
        for name, f in self._fields.items():
            for key in {name, f.logical_name}:
                self._by_name.setdefault(key, []).append(f)
            for alias in f.aliases:
                self._by_name.setdefault(alias, []).append(f)
        self._source_path = str(source_path) if source_path is not None else None

    @property
    def source_path(self) -> str | None:
        return self._source_path

    def names(self) -> list[str]:
        return sorted(self._fields)

    @staticmethod
    def _market_of(dataset: str | None) -> str | None:
        """由数据集名推断市场（us_*/ashare_*）；无法判断返回 None。"""
        if not dataset:
            return None
        if dataset.startswith("us_") or dataset.startswith("us_stock") or dataset == "us":
            return "us"
        if dataset.startswith("ashare_") or dataset.startswith("a_share") or dataset == "ashare":
            return "ashare"
        return None

    def resolve_one(
        self, name: str, market: str | None = None, *, dataset: str | None = None
    ) -> SemanticField | None:
        """按逻辑名或别名解析；找不到返回 None（调用方可再回退 registry）。

        market 传入时，优先返回该市场专属字段；没有专属字段时回退 any（跨市场
        通用）。dataset 传入时用数据集名前缀推断 market（更高优先级）。

        **#54 fail-closed**：无 market/dataset 上下文且存在多个非 any 候选时，
        production 抛 ``AmbiguousSemanticFieldError``（禁止 YAML 顺序决定市场）；
        research 告警后取第一个。
        """
        candidates = self._by_name.get(name)
        if not candidates:
            return None
        effective = market or (self._market_of(dataset) if dataset else None)
        if effective and effective != "any":
            for f in candidates:
                if f.market == effective:
                    return f
        # 回退：any 通用字段
        for f in candidates:
            if f.market == "any":
                return f
        # 无上下文且多市场候选 → fail-closed
        ambiguous = [f for f in candidates if f.market != "any"]
        if len(ambiguous) > 1:
            markets = sorted({f.market for f in ambiguous})
            from data_access.core.exceptions import AmbiguousSemanticFieldError
            from data_access.read.query_budget import is_strict_semantics

            msg = (
                f"逻辑字段 '{name}' 跨市场歧义（候选市场: {markets}）。"
                "请显式传 market='ashare'/'us' 或 dataset= 让 catalog 消歧。"
            )
            # #16 strict_read 完全共享 production 的 fail-closed：不能只查
            # _production_mode()（DATA_ACCESS_STRICT_READ=1 也要拦）。
            if is_strict_semantics():
                raise AmbiguousSemanticFieldError(msg)
            logger.warning("%s（research 放行，取 YAML 顺序第一个）", msg)
        return candidates[0]

    def resolve_by_physical(self, dataset: str, physical_name: str) -> SemanticField | None:
        """按 (dataset, 物理列名) 反查逻辑字段（调用方传物理列时的单位归一化用）。

        同一物理列可能被多个逻辑字段引用，返回第一个；优先返回与该数据集同市场
        的字段（防止 A股/美股同名物理列串味）。
        """
        candidates = [
            f
            for f in self._fields.values()
            if f.dataset == dataset and f.physical_name == physical_name
        ]
        if not candidates:
            return None
        effective = self._market_of(dataset)
        if effective and effective != "any":
            for f in candidates:
                if f.market == effective:
                    return f
        for f in candidates:
            if f.market == "any":
                return f
        return candidates[0]

    def get(self, name: str, market: str | None = None, *, dataset: str | None = None) -> SemanticField:
        """严格解析；找不到抛 ValidationError。"""
        f = self.resolve_one(name, market=market, dataset=dataset)
        if f is None:
            raise ValidationError(
                f"逻辑字段 '{name}' 不在 SemanticFieldCatalog 中。"
                f"可用: {self.names()}"
            )
        return f

    def resolve(
        self, *names: str, market: str | None = None, dataset: str | None = None
    ) -> list[SemanticField]:
        out: list[SemanticField] = []
        for name in names:
            out.append(self.get(name, market=market, dataset=dataset))
        return out

    def to_dict(self) -> dict[str, Any]:
        return {name: f.to_dict() for name, f in self._fields.items()}

    def fingerprint(self) -> str:
        """语义 catalog 稳定指纹（#3：缓存 key 必须覆盖 catalog 语义版本）。"""
        import hashlib
        import json

        payload = json.dumps(
            {k: f.to_dict() for k, f in self._fields.items()},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def semantic_coverage_audit(self, registry: Any) -> dict[str, Any]:
        """#30 语义覆盖审计：registry 物理字段 vs catalog。

        每个 registry 物理列 → FULL_SEMANTIC / PHYSICAL_ONLY / AMBIGUOUS；
        catalog 引用了 registry 里不存在的 dataset/column → BLOCKED。

        物理列可以读，但进入 PIT/join/因子挖掘前必须有完整 semantic contract——
        本审计是 coverage 的看板，供 CI 对齐。
        """
        from data_access.core.exceptions import ValidationError

        by_phys: dict[tuple[str, str], list[str]] = {}
        for f in self._fields.values():
            by_phys.setdefault((f.dataset, f.physical_name), []).append(f.logical_name)

        fields_report: dict[str, dict[str, Any]] = {}
        for name in registry.names():
            ds = registry.get(name)
            schema = getattr(ds, "schema", None) or {}
            for col in sorted(schema):
                refs = by_phys.get((name, col), [])
                status = "FULL_SEMANTIC"
                if len(refs) > 1:
                    status = "AMBIGUOUS"
                elif not refs:
                    status = "PHYSICAL_ONLY"
                fields_report[f"{name}.{col}"] = {
                    "status": status,
                    "logical": sorted(refs),
                }

        blocked: dict[str, str] = {}
        for f in self._fields.values():
            if f.derived_expression:
                # #P0-24 derived 字段不落盘，不做物理列存在性检查。
                continue
            try:
                ds = registry.get(f.dataset)
            except ValidationError:
                blocked[f"{f.logical_name} -> {f.dataset}.{f.physical_name}"] = (
                    "DATASET_MISSING"
                )
                continue
            schema = getattr(ds, "schema", None) or {}
            if f.physical_name not in schema:
                blocked[f"{f.logical_name} -> {f.dataset}.{f.physical_name}"] = (
                    "COLUMN_MISSING"
                )

        counts = {"FULL_SEMANTIC": 0, "PHYSICAL_ONLY": 0, "AMBIGUOUS": 0}
        for r in fields_report.values():
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        return {
            "fields": fields_report,
            "blocked": blocked,
            "counts": counts,
            "total_physical_columns": len(fields_report),
            "blocked_count": len(blocked),
        }

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
            from data_access.registry.yaml_loader import strict_yaml_load

            raw = strict_yaml_load(fh.read(), context=str(path)) or {}
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
