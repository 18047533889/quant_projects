"""
data_access.registry —— 数据集登记表

职责：
    1. 读 datasets.yaml，把每个数据集解析成 StaticDataset 或 ParametricDataset
    2. 校验 schema：必填字段、类型、access_mode 取值、参数 schema
    3. 把 env var 和 namespace 占位符展开到真实路径
    4. 给 paths.PathAuthorizer 提供白名单根集合

非职责：
    不连接 DuckDB、不读 parquet 文件（那是 engine/store 的事）。
    不校验路径真实存在（运行时读取的时候再说，注册时只做静态检查）。

维护人：quant 基础平台组    最后更新：2026-04-19
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from data_access.core.exceptions import DataError, ValidationError
from .yaml_loader import strict_yaml_load
from .layout_policy import LayoutPolicy, parse_layout_policy
from .params_validation import ParamSpec, parse_params_schema, validate_params
from data_access.read.query_budget import DatasetQueryPolicy, parse_dataset_query_policy
from data_access.read.formats import FormatSpec, default_glob_for_format, normalize_format_name
from data_access.core.storage import StorageSpec
from data_access.read.partition_planner import parse_partitioning
from .paths import canonicalize, expand_env, resolve_namespace_path


# access_mode 三态：
#   published   —— 全员只读的正式数据（massive_parquet、factor_lake 快照等）
#   namespaced  —— 个人实验产物，路径自动带 ${RUN_NAMESPACE}，互不干扰
#   staging     —— 发布前的暂存区，写入仍按 namespace 隔离，校验通过后由 publish()
#                  原子切换到 published_root（PR1 不实现 publish，只登记字段占位）
_VALID_ACCESS_MODES = {"published", "namespaced", "staging"}
_VALID_LAYOUTS = {"plain", "hive"}

# #P0-final closure 2：顶层 Dataset 完整 allowed-key schema（strict）。
# static 与 parametric 各自允许自己的路径字段；其余 key 两侧通用。
_COMMON_DATASET_KEYS = frozenset({
    "kind", "access_mode", "layout", "roles", "time_column", "instrument_column",
    "hive_partitioning", "union_by_name", "format", "storage", "partitioning",
    "semantic", "engine", "schema", "query_policy", "partition_columns",
    "storage_format", "layout_policy", "schema_version", "authorized_root",
    "specialized_only", "specialized_only_reason",
    # R14 #1：factor_matrix 走 generation 指针（manifest.json 的 ``generation``
    # 指向当前不可变代；reader 只读该代，绝不 glob generation/* 混入 previous）。
    "generation_pointer",
    # R29-P0 #200：generation_required（manifest 缺失 → 禁 legacy fallback）。
    "generation_required",
    # R29-P0 #194：跨 epoch schema migration 审批（缺字段/dtype 变化的合法演进）。
    "schema_migrations",
    # R29-P0 #206：数据集的 mutation ownership（manifest 新鲜度语义）。
    "mutation_owner",
})
_STATIC_DATASET_KEYS = _COMMON_DATASET_KEYS | frozenset({"root", "glob"})
_PARAMETRIC_DATASET_KEYS = _COMMON_DATASET_KEYS | frozenset({
    "root_template", "glob_template", "params_schema",
})


@dataclass(frozen=True)
class DatasetBase:
    """所有数据集的共同字段。

    PR8 注：schema 字段放在 StaticDataset / ParametricDataset 上而不是这里，
    因为 dataclass + frozen + 继承对字段默认值有严格顺序要求（非默认字段不能
    跟在默认字段之后）。放 base 上会污染子类的非默认字段，所以两个子类各自带。
    """
    name: str
    access_mode: str                  # published / namespaced / staging
    layout: str                       # plain / hive
    time_column: str | None           # 业务约定的时间列名（semantic roles 的便捷别名，可空）
    instrument_column: str | None     # 业务约定的标的列名（同上，可空）
    hive_partitioning: bool           # 传给 DuckDB read_parquet(hive_partitioning=)
    union_by_name: bool               # 传给 DuckDB read_parquet(union_by_name=)

    @property
    def format(self) -> str:
        """物理文件格式名（格式符快捷入口，等价于 format_spec.type）。"""
        return self.format_spec.type


@dataclass(frozen=True)
class StaticDataset(DatasetBase):
    """「根一次定死」的数据集：启动时算出 glob，read 时直接用。"""
    root: Path                        # 已 canonicalize 的绝对路径
    glob: str                         # 相对 root 的 glob，如 "**/*.parquet"
    # PR8：可选 schema，格式 {列名: 类型字符串}，首访自检用。空 = 不校验。
    schema: Mapping[str, str] = field(default_factory=dict)
    query_policy: DatasetQueryPolicy = field(default_factory=DatasetQueryPolicy)
    partition_columns: tuple[str, ...] = field(default_factory=tuple)
    storage_format: str = "long"
    layout_policy: LayoutPolicy | None = None
    # #P1-final closure：specialized_only 数据集（如 factor_lake_wide——宽表 pivot，
    # 标的在列轴上，不是 time×instrument 普通 long 表）**不进入 generic 读契约**。
    # generic read/scan 遇到直接拒绝，杜绝在「asset」这种列轴上假过滤。
    specialized_only: bool = False
    specialized_only_reason: str | None = None
    # R29-P0 #194：跨 epoch schema migration 审批（typed IR，SchemaEpochGate 消费）。
    schema_migrations: tuple[Mapping[str, Any], ...] = ()
    # R29-P0 #206：mutation ownership——manifest 新鲜度语义的根。
    #   dataaccess          → source_epoch bump（DataAccess 自己写，epoch 权威）
    #   external_versioned  → publisher generation/source manifest
    #   external_mutable    → LIST/stat/checkpoint/watch（epoch 不权威）
    #   immutable           → content identity（只读，identity 即新鲜度）
    mutation_owner: str = "dataaccess"
    # ---- 通用 Data IO Layer 新增字段（全部带默认值，向后兼容） ----
    format_spec: FormatSpec = field(default_factory=FormatSpec)   # 物理文件格式（parquet/csv/...）
    storage: StorageSpec | None = None                            # #P0-final closure 4 typed StorageSpec
    roles: Mapping[str, str] = field(default_factory=dict)        # 语义角色（event_time/instrument/...）
    partitioning: "PartitionSpec | None" = None                   # #P0-final closure 3 typed PartitionSpec
    semantic: str | None = None                                   # panel/event/factor/...
    engine: dict[str, Any] | None = None                          # 优先/兜底执行引擎
    schema_version: str | None = None                             # #36 schema 版本号
    authorized_root: Path | None = None                           # #P0-49 显式授权根

    @property
    def kind(self) -> str:
        return "static"

    def resolve_paths(self, **params: Any) -> list[str]:
        """静态数据集不接受参数；有参数传入说明调用方搞错了。"""
        if params:
            raise ValidationError(
                f"静态数据集 '{self.name}' 不接受参数，收到: {list(params)}"
            )
        # 返回带通配符的 glob 表达式，交给 DuckDB 展开；相比 Python 侧 glob
        # 再传 list，DuckDB 直接吃 glob 能让它做自己的元数据优化。
        # #P1-final closure 12：read/write 时才解析当前 context 的 namespace。
        return [resolve_namespace_path(str(self.root / self.glob))]


@dataclass(frozen=True)
class ParametricDataset(DatasetBase):
    """参数化数据集：根或 glob 含占位符，调用时用 params 填充。

    典型：factor_lake 的 root 是 {lake_root}/factors/{factor_id}/，每次读一个 factor_id。
    """
    root_template: str                # 含 ${ENV} 和 {param} 占位符的 root 模板
    glob_template: str                # 同上，相对 root 的 glob 模板
    params_schema: Mapping[str, str]  # {参数名: "str" | "int"}（YAML 摘要）
    param_specs: Mapping[str, ParamSpec] = field(default_factory=dict)  # 硬校验规格
    # 预先算好的 canonical root（不含参数部分），用于白名单校验。
    # 例如 factor_lake 的 static_root 是 ${lake_root}/factors
    static_root: Path = field(default=Path("/"))
    # PR8：可选 schema，格式 {列名: 类型字符串}，首访自检用。空 = 不校验。
    schema: Mapping[str, str] = field(default_factory=dict)
    query_policy: DatasetQueryPolicy = field(default_factory=DatasetQueryPolicy)
    partition_columns: tuple[str, ...] = field(default_factory=tuple)
    storage_format: str = "long"
    layout_policy: LayoutPolicy | None = None
    # #P1-final closure：specialized_only（见 StaticDataset 注释）。
    specialized_only: bool = False
    specialized_only_reason: str | None = None
    # ---- 通用 Data IO Layer 新增字段（全部带默认值，向后兼容） ----
    format_spec: FormatSpec = field(default_factory=FormatSpec)   # 物理文件格式（parquet/csv/...）
    storage: StorageSpec | None = None                            # #P0-final closure 4 typed StorageSpec
    roles: Mapping[str, str] = field(default_factory=dict)        # 语义角色（event_time/instrument/...）
    partitioning: "PartitionSpec | None" = None                   # #P0-final closure 3 typed PartitionSpec
    semantic: str | None = None                                   # panel/event/factor/...
    engine: dict[str, Any] | None = None                          # 优先/兜底执行引擎
    schema_version: str | None = None                             # #36 schema 版本号
    authorized_root: Path | None = None                           # #P0-49 显式授权根
    # R29-P0 #194：跨 epoch schema migration 审批（typed IR，SchemaEpochGate 消费）。
    schema_migrations: tuple[Mapping[str, Any], ...] = ()
    # R29-P0 #206：mutation ownership（dataaccess/external_versioned/
    # external_mutable/immutable）——见 StaticDataset 注释。
    mutation_owner: str = "dataaccess"
    # R14 #1：factor_matrix 走 generation 指针。读路径只解析
    # ``manifest.json`` 的 ``generation`` 指向那一代；缺该指针才 legacy glob。
    generation_pointer: bool = False
    # R29-P0 #200：正式迁移到 generation 模型的数据集声明 ``generation_required``。
    # manifest.json 缺失/无 generation 时**禁止 legacy fallback**（fail-closed）——
    # manifest 被误删可能把 orphan/legacy 数据重新读出来；legacy fallback 仅限
    # migration/research（generation_pointer=true 但未声明 required）。
    generation_required: bool = False

    @property
    def kind(self) -> str:
        return "parametric"

    def resolve_root(self, **params: Any) -> str:
        """填模板返回数据集根目录（不含 glob / generation 段）。"""
        specs = self.param_specs or {
            k: ParamSpec(name=k, type=t) for k, t in self.params_schema.items()
        }
        validated = validate_params(self.name, specs, params)
        root_tpl = resolve_namespace_path(self.root_template)
        try:
            return str(Path(root_tpl.format(**validated)))
        except KeyError as exc:
            raise ValidationError(f"模板变量缺失：{exc}") from exc

    def current_generation(self, **params: Any) -> str | None:
        """读 ``root/manifest.json`` 的 ``generation`` 指针；无 manifest/无指针 → None。

        R14 #1：只解析 manifest 指向的当前代。**绝不用 ``generation/*`` 通配**——
        那会把 current + previous generations 一起读进来，比不解析更危险。
        """
        if not self.generation_pointer:
            return None
        root = Path(self.resolve_root(**params))
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            return None
        try:
            import json

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise DataError(
                f"{self.name} manifest.json 不可读（{manifest_path}）: {exc}"
            ) from exc
        gid = manifest.get("generation")
        return str(gid) if gid else None

    def resolve_paths(self, **params: Any) -> list[str]:
        """用 params 填模板，返回给 DuckDB 的 glob 表达式。

        R14 #1：``generation_pointer`` 数据集（factor_matrix）先读 manifest 的
        ``generation`` 指针，只解析 ``generation/<gid>/<glob>`` 那一代——
        * manifest 有 ``generation`` 但目录缺失 = 当前发布代损坏 → ``DataError``
          （**fail-closed**，绝不 legacy fallback 把 previous/orphan 数据捞出来）；
        * manifest 无 ``generation`` / 无 manifest = legacy 布局 → 原样 glob。
        """
        specs = self.param_specs or {
            k: ParamSpec(name=k, type=t) for k, t in self.params_schema.items()
        }
        validated = validate_params(self.name, specs, params)
        # #P1-final closure 12：read/write 时才把 ``${RUN_NAMESPACE}`` 解析成当前
        # context 的 namespace（在 ``.format(**validated)`` **之前**——否则
        # ``{RUN_NAMESPACE}`` 会被 format 当成参数占位符而 KeyError）。
        root_tpl = resolve_namespace_path(self.root_template)
        glob_tpl = resolve_namespace_path(self.glob_template)
        try:
            root = root_tpl.format(**validated)
            glob_part = glob_tpl.format(**validated)
        except KeyError as exc:
            raise ValidationError(f"模板变量缺失：{exc}") from exc
        if self.generation_pointer:
            manifest = Path(root) / "manifest.json"
            if manifest.exists():
                import json

                try:
                    data = json.loads(manifest.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    raise DataError(
                        f"{self.name} manifest.json 不可读（{manifest}）: {exc}"
                    ) from exc
                gid = data.get("generation")
                if gid:
                    gen_dir = Path(root) / "generation" / str(gid)
                    if not gen_dir.is_dir():
                        raise DataError(
                            f"{self.name} manifest.generation={gid!r} 指向的目录缺失"
                            f"（{gen_dir}）——当前发布代损坏，拒绝读取（fail-closed）"
                        )
                    return [str(gen_dir / glob_part)]
            # R29-P0 #200：generation_required 数据集 manifest 缺失 → fail-closed，
            # 禁止 legacy fallback（manifest 误删会重新读 orphan/legacy 数据）。
            if self.generation_required:
                raise DataError(
                    f"{self.name} 已声明 generation_required=true，但 manifest.json "
                    f"缺失/无 generation 指针（{manifest}）——拒绝 legacy fallback"
                    "（R29-P0：正式迁移到 generation 模型的数据集不允许 orphan/"
                    "legacy 数据复活）。请用 migration/research 数据集处理旧布局。"
                )
        return [str(Path(root) / glob_part)]


Dataset = StaticDataset | ParametricDataset


def _require_str(data: dict, key: str, *, context: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{context}: 字段 '{key}' 必填且为非空字符串")
    return value


def _strict_bool(value: Any, *, key: str, context: str) -> bool:
    """#32 严格 bool 解析：``"false"`` 字符串 → False；非 bool 类型报错。

    避免 ``bool("false") == True`` 这类 YAML 字符串串味 bug（hive_partitioning /
    union_by_name 等生产配置）。
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "on"}:
            return True
        if text in {"false", "0", "no", "off", ""}:
            return False
        raise ValidationError(f"{context}: 字段 '{key}' 必须是布尔值，收到 {value!r}")
    raise ValidationError(f"{context}: 字段 '{key}' 必须是布尔值，收到 {value!r}")


def _validate_known_keys(
    mapping: dict, *, allowed: set[str], context: str
) -> None:
    """#P0-38 strict typed config：未知 key 拒绝（storage/partitioning/engine）。"""
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValidationError(
            f"{context} 含未知配置 key {unknown}；应为 {sorted(allowed)} 之一"
        )


def _parse_partition_columns(raw: Any, *, context: str) -> tuple[str, ...]:
    """#P0-53 默认/空 → ``()``，不再用 ``("year",)``。

    之前默认 ``("year",)`` 会让 schema checker 把 ``year`` 当分区列豁免 missing——
    普通非 hive 数据集真声明了 ``year`` 列、但物理文件丢了 ``year`` 会被错误豁免。
    """
    if raw is None:
        return ()
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        raise ValidationError(
            f"{context}: partition_columns 必须是字符串列表，收到 {type(raw).__name__}"
        )
    cols = tuple(str(c) for c in raw if str(c))
    return cols


def _parse_schema_version(raw: dict, *, context: str) -> str | None:
    """#P0-45 ``schema_version`` key 出现但非法（非字符串/空串）→ 启动失败。

    之前 ``schema_version: 3`` 会静默变 None——配置写错不能被吞掉。
    """
    if "schema_version" not in raw:
        return None
    sv = raw.get("schema_version")
    if not isinstance(sv, str) or not sv.strip():
        raise ValidationError(
            f"{context}: schema_version 必须是字符串，收到 {sv!r}"
        )
    return sv.strip()


def _parse_schema_migrations(
    raw: dict, *, context: str
) -> tuple[Mapping[str, Any], ...]:
    """R29-P0 #194：``schema_migrations`` 严格解析（typed migration IR 的 registry 源）。

    每项必须含 ``from_fingerprint`` / ``to_fingerprint``（schema 指纹，16 位 hex）
    与 ``kind``（add_column / dtype_change / unit_change）；``approved`` 缺省 False
    ——**未 approved 的 migration 不构成放行依据**（fail-closed：必须显式批准）。
    unknown-key / 非数组 → 启动失败（安全配置 extra=forbid）。
    """
    if "schema_migrations" not in raw:
        return ()
    value = raw.get("schema_migrations")
    if not isinstance(value, (list, tuple)):
        raise ValidationError(
            f"{context}: schema_migrations 必须是数组，收到 {type(value).__name__}"
        )
    _KNOWN = {"from_fingerprint", "to_fingerprint", "kind", "field", "approved", "reviewer"}
    out: list[Mapping[str, Any]] = []
    for i, entry in enumerate(value):
        if not isinstance(entry, dict):
            raise ValidationError(
                f"{context}: schema_migrations[{i}] 必须是 mapping"
            )
        unknown = sorted(set(entry) - _KNOWN)
        if unknown:
            raise ValidationError(
                f"{context}: schema_migrations[{i}] 含未知 key {unknown}；"
                f"应为 {sorted(_KNOWN)} 之一（R29-P0 fail-closed）"
            )
        f_from = entry.get("from_fingerprint")
        f_to = entry.get("to_fingerprint")
        kind = str(entry.get("kind") or "").strip().lower()
        if not isinstance(f_from, str) or not f_from or not isinstance(f_to, str) or not f_to:
            raise ValidationError(
                f"{context}: schema_migrations[{i}] 必须含非空 from_fingerprint "
                "/ to_fingerprint"
            )
        if kind not in {"add_column", "dtype_change", "unit_change"}:
            raise ValidationError(
                f"{context}: schema_migrations[{i}].kind 必须是 add_column / "
                f"dtype_change / unit_change，收到 {kind!r}"
            )
        approved = _strict_bool(
            entry.get("approved", False), key="approved", context=context
        )
        out.append(
            {
                "from_fingerprint": f_from,
                "to_fingerprint": f_to,
                "kind": kind,
                "field": (str(entry["field"]) if entry.get("field") else None),
                "approved": approved,
                "reviewer": (str(entry["reviewer"]) if entry.get("reviewer") else None),
            }
        )
    return tuple(out)


def _parse_authorized_root(
    raw: dict,
    *,
    static_prefix: str | None,
    context: str,
) -> Path | None:
    """#P0-49 显式 ``authorized_root``（授权安全边界）；缺省回退静态前缀。"""
    ar = raw.get("authorized_root")
    if ar is None:
        if static_prefix is None:
            return None
        # 向后兼容：静态前缀（模板里第一个 { 之前部分）。#P1-final closure 12
        # 保留 ``${RUN_NAMESPACE}`` 占位符，authorized_root 白名单在消费时解析。
        expanded = expand_env(str(static_prefix))
        return canonicalize(expanded.rstrip("/") or "/")
    if not isinstance(ar, str) or not ar.strip():
        raise ValidationError(f"{context}: authorized_root 必须是非空字符串")
    return canonicalize(expand_env(ar.strip()))


_UNRESOLVED_ENV_RE = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}")
_UNRESOLVED_DOLLAR_RE = re.compile(r"(?<!\\)\$[A-Za-z_][A-Za-z0-9_]*")


def _assert_no_unresolved_env(fields: Mapping[str, str], *, context: str) -> None:
    """#P0-47 env 展开后的字段里仍残留 ``${...}`` / ``$VAR`` → 启动失败。

    ``expand_env`` 未设 env 且无 default 时保留原样——不能把配置问题变成奇怪的
    路径问题，生产必须启动即失败。（在 ``expand_env`` **之后**检查，不能检查原始
    YAML——那里 ${ENV} 是合法待展开 token。）
    """

    def walk(node: Any, path: str) -> None:
        if isinstance(node, str):
            m = _UNRESOLVED_ENV_RE.search(node) or _UNRESOLVED_DOLLAR_RE.search(node)
            if m and "${RUN_NAMESPACE}" not in m.group(0):
                raise ValidationError(
                    f"{context}: 字段 {path} 含未解析环境变量 {m.group(0)!r}。"
                    "请设置该 env，或在 YAML 里提供 ${VAR:-default} 默认值。"
                )
        elif isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(dict(fields), "$")


def _parse_dataset(name: str, raw: dict) -> Dataset:
    """解析单个数据集条目。raw 是 YAML 里该数据集 name 下的字典。"""
    context = f"数据集 '{name}'"

    kind = raw.get("kind", "static")
    if kind not in {"static", "parametric"}:
        raise ValidationError(f"{context}: kind 必须是 static/parametric，收到 {kind!r}")

    # #P0-final closure 2：顶层 Dataset schema 是 strict schema——任何未知 key
    # 启动即失败。``time_colum:`` / ``query_polcy:`` / ``formatt:`` 这类 typo
    # 之前会被静默忽略然后走默认值；registry 是单一事实源，配置写错必须暴露。
    # 嵌套 storage/partitioning/engine 已有各自 strict key 检查（见下）。
    if kind == "static":
        _validate_known_keys(raw, allowed=_STATIC_DATASET_KEYS, context=context)
    else:
        _validate_known_keys(raw, allowed=_PARAMETRIC_DATASET_KEYS, context=context)

    access_mode = _require_str(raw, "access_mode", context=context)
    if access_mode not in _VALID_ACCESS_MODES:
        raise ValidationError(
            f"{context}: access_mode 必须是 {_VALID_ACCESS_MODES}，收到 {access_mode!r}"
        )

    layout = raw.get("layout", "plain")
    if layout not in _VALID_LAYOUTS:
        raise ValidationError(f"{context}: layout 必须是 {_VALID_LAYOUTS}，收到 {layout!r}")

    # time_column / instrument_column 可选（generic table）：优先显式字段，其次语义角色。
    roles_raw = raw.get("roles")
    roles: dict[str, str] = {}
    if roles_raw is not None:
        if not isinstance(roles_raw, dict):
            raise ValidationError(
                f"{context}: roles 必须是 mapping（角色名→列名），收到 {type(roles_raw).__name__}"
            )
        roles = {str(k): str(v) for k, v in roles_raw.items() if v is not None}

    time_column = raw.get("time_column", roles.get("event_time"))
    instrument_column = raw.get("instrument_column", roles.get("instrument"))
    if time_column is not None and (not isinstance(time_column, str) or not time_column):
        raise ValidationError(f"{context}: time_column 必须是非空字符串或省略")
    if instrument_column is not None and (
        not isinstance(instrument_column, str) or not instrument_column
    ):
        raise ValidationError(f"{context}: instrument_column 必须是非空字符串或省略")
    time_column = str(time_column) if time_column is not None else None
    instrument_column = str(instrument_column) if instrument_column is not None else None

    hive_partitioning = _strict_bool(
        raw.get("hive_partitioning", layout == "hive"),
        key="hive_partitioning",
        context=context,
    )
    union_by_name = _strict_bool(
        raw.get("union_by_name", False), key="union_by_name", context=context
    )

    # ---- 通用 Data IO Layer 字段解析 ----
    format_spec = FormatSpec.from_yaml(raw.get("format", "parquet"), context=context)
    # #P0-final closure 4：storage 统一走 ``StorageSpec.from_yaml`` 编译成 typed
    # ``StorageSpec`` 保存——任何模块不再各自解析 raw dict。``{source:{type:cos}}``
    # 合法配置不再被静默当 local；unknown key 启动即失败。
    storage = None
    if raw.get("storage") is not None:
        storage = StorageSpec.from_yaml(raw.get("storage"), context=f"{context}: storage")
    # #P0-final closure 3：partitioning 统一走 ``parse_partitioning`` 编译成 typed
    # ``PartitionSpec``（planner 与 loader 读同一份 schema，不再 split-brain）。
    partitioning = None
    if raw.get("partitioning") is not None:
        partitioning = parse_partitioning(raw.get("partitioning"), context=f"{context}: partitioning")
    semantic = raw.get("semantic")
    if semantic is not None and not isinstance(semantic, str):
        raise ValidationError(f"{context}: semantic 必须是字符串")
    engine = raw.get("engine")
    if engine is not None:
        if not isinstance(engine, dict):
            raise ValidationError(f"{context}: engine 必须是 mapping，收到 {type(engine).__name__}")
        _validate_known_keys(
            engine,
            allowed={"preferred", "fallback", "backend", "options"},
            context=f"{context}: engine",
        )

    # PR8：解析可选 schema（列名 -> 类型字符串）。不填 = 不做首访自检。
    schema_raw = raw.get("schema", {}) or {}
    if not isinstance(schema_raw, dict):
        raise ValidationError(
            f"{context}: schema 必须是 mapping（列名→类型字符串），收到 {type(schema_raw).__name__}"
        )
    schema_decl: dict[str, str] = {}
    for col, typ in schema_raw.items():
        if not isinstance(col, str) or not col:
            raise ValidationError(f"{context}: schema 列名必须是非空字符串，收到 {col!r}")
        if not isinstance(typ, str) or not typ:
            raise ValidationError(
                f"{context}: schema 列 '{col}' 类型必须是非空字符串，收到 {typ!r}"
            )
        schema_decl[col] = typ

    query_policy = parse_dataset_query_policy(raw.get("query_policy"), context=context)
    partition_columns = _parse_partition_columns(raw.get("partition_columns"), context=context)
    storage_format = str(raw.get("storage_format", "long")).lower()
    if storage_format not in {"long", "wide"}:
        raise ValidationError(
            f"{context}: storage_format 必须是 long|wide，收到 {storage_format!r}"
        )
    layout_policy = parse_layout_policy(raw.get("layout_policy"))
    specialized_only = _strict_bool(
        raw.get("specialized_only", False), key="specialized_only", context=context
    )
    # R14 #1：generation 指针数据集（factor_matrix）——读路径只解析 manifest
    # 的 ``generation`` 指向那一代（见 ``current_generation`` / ``resolve_paths``）。
    generation_pointer = _strict_bool(
        raw.get("generation_pointer", False),
        key="generation_pointer",
        context=context,
    )
    # R29-P0 #200：generation_required=true 要求 manifest.json 必须有 generation
    # 指针——manifest 缺失时 resolve_paths fail-closed（禁止 legacy fallback）。
    generation_required = _strict_bool(
        raw.get("generation_required", False),
        key="generation_required",
        context=context,
    )
    specialized_reason = raw.get("specialized_only_reason")
    if specialized_reason is not None and not isinstance(specialized_reason, str):
        raise ValidationError(
            f"{context}: specialized_only_reason 必须是字符串，收到 {specialized_reason!r}"
        )
    if specialized_only and not specialized_reason:
        raise ValidationError(
            f"{context}: specialized_only=true 时必须同时给出 specialized_only_reason"
            "（说明为什么不能 generic 读、该用什么专用入口）"
        )
    schema_migrations = _parse_schema_migrations(raw, context=context)
    # R29-P0 #206：mutation_owner 严格枚举（未知值 fail-closed）。
    mutation_owner = str(raw.get("mutation_owner", "dataaccess") or "dataaccess").strip().lower()
    if mutation_owner not in {"dataaccess", "external_versioned", "external_mutable", "immutable"}:
        raise ValidationError(
            f"{context}: mutation_owner 必须是 dataaccess / external_versioned / "
            f"external_mutable / immutable 之一，收到 {mutation_owner!r}"
        )

    if kind == "static":
        # published 数据集的 root 是直接路径；namespaced 数据集在 static 里
        # 也允许，但要手动把 ${RUN_NAMESPACE} 展开
        root_raw = _require_str(raw, "root", context=context)
        root_expanded = expand_env(root_raw)
        # #P1-final closure 12：namespaced 数据集的 root 里可能含
        # ``${RUN_NAMESPACE}``——**保留占位符**，read/write 时由
        # ``resolve_namespace_path`` 按当前 context 解析（load 不再烘焙，
        # 长期 worker 换 DataAccessSession 后目录不再串）。
        root = canonicalize(root_expanded)
        glob = raw.get("glob", default_glob_for_format(format_spec.type))
        # #P0-47 env 展开后仍有残留 → 启动失败
        _assert_no_unresolved_env(
            {"root": root_expanded, "glob": glob}, context=context
        )
        schema_version = _parse_schema_version(raw, context=context)
        authorized_root = _parse_authorized_root(raw, static_prefix=None, context=context)
        return StaticDataset(
            name=name,
            access_mode=access_mode,
            layout=layout,
            time_column=time_column,
            instrument_column=instrument_column,
            hive_partitioning=hive_partitioning,
            union_by_name=union_by_name,
            format_spec=format_spec,
            storage=storage,
            roles=roles,
            partitioning=partitioning,
            semantic=semantic,
            engine=engine,
            schema_version=schema_version,
            authorized_root=authorized_root,
            root=root,
            glob=glob,
            schema=schema_decl,
            query_policy=query_policy,
            partition_columns=partition_columns,
            storage_format=storage_format,
            layout_policy=layout_policy,
            specialized_only=specialized_only,
            specialized_only_reason=specialized_reason,
            schema_migrations=schema_migrations,
            mutation_owner=mutation_owner,
        )
    root_template_raw = _require_str(raw, "root_template", context=context)
    root_template = expand_env(root_template_raw)
    # #P1-final closure 12：``${RUN_NAMESPACE}`` **保留占位符**，read/write 时
    # 由 ``resolve_namespace_path`` 按当前 context 解析（load 不再烘焙）。
    glob_template = raw.get("glob_template", default_glob_for_format(format_spec.type))
    # #P0-47 env 展开后仍有残留 → 启动失败（含 authorized_root 模板）
    _assert_no_unresolved_env(
        {"root_template": root_template, "glob_template": glob_template},
        context=context,
    )
    params_schema_raw = raw.get("params_schema", {})
    if not isinstance(params_schema_raw, dict) or not params_schema_raw:
        raise ValidationError(f"{context}: parametric 数据集必须声明非空 params_schema")
    param_specs = parse_params_schema(params_schema_raw)
    params_schema = {k: spec.type for k, spec in param_specs.items()}

    # #P0-49 算出静态前缀用于白名单：优先显式 ``authorized_root``；缺省时才用
    # 「模板里第一个 { 之前的部分」（保守但过宽——如 /data/{market}/{factor_id}
    # 会授权整个 /data）。新数据集请显式声明 authorized_root；本层保留静态前缀
    # 作为向后兼容回退。
    # #P1-final closure 12：``${RUN_NAMESPACE}`` 里的 ``{`` 不能当参数占位符——
    # 先屏蔽成无花括号的哨兵再找第一个 ``{``，否则 ``/staging/${RUN_NAMESPACE}/f/``
    # 的静态前缀会被切成 ``/staging/$``（namespace 被吞）。
    _NS_SENTINEL = "__DA_NS_PLACEHOLDER__"
    _ns_probe = root_template.replace("${RUN_NAMESPACE}", _NS_SENTINEL)
    _first_brace = _ns_probe.find("{")
    _static_prefix = _ns_probe[:_first_brace] if _first_brace >= 0 else _ns_probe
    static_prefix = _static_prefix.replace(_NS_SENTINEL, "${RUN_NAMESPACE}")
    static_root = canonicalize(static_prefix.rstrip("/") or "/")
    authorized_root = _parse_authorized_root(raw, static_prefix=static_prefix, context=context)

    schema_version = _parse_schema_version(raw, context=context)
    return ParametricDataset(
        name=name,
        access_mode=access_mode,
        layout=layout,
        time_column=time_column,
        instrument_column=instrument_column,
        hive_partitioning=hive_partitioning,
        union_by_name=union_by_name,
        format_spec=format_spec,
        storage=storage,
        roles=roles,
        partitioning=partitioning,
        semantic=semantic,
        engine=engine,
        schema_version=schema_version,
        authorized_root=authorized_root,
        root_template=root_template,
        glob_template=glob_template,
        params_schema=params_schema,
        param_specs=param_specs,
        static_root=static_root,
        schema=schema_decl,
        query_policy=query_policy,
        partition_columns=partition_columns,
        storage_format=storage_format,
        generation_required=generation_required,
        layout_policy=layout_policy,
        specialized_only=specialized_only,
        specialized_only_reason=specialized_reason,
        schema_migrations=schema_migrations,
        mutation_owner=mutation_owner,
        generation_pointer=generation_pointer,
    )


class DatasetRegistry:
    """数据集登记表容器。由 store.get_store() 加载一次，全进程共用。"""

    def __init__(self, datasets: Mapping[str, Dataset]) -> None:
        self._datasets = dict(datasets)
        # R27-L：frozen dataclass ≠ immutable——Dataset 的 Mapping 字段仍是
        # 可变 dict，`store.get_dataset('x').schema['k'] = v` 能改共享 registry。
        # 统一冻结成 MappingProxyType（只读；dict()/keys()/get()/in 全部兼容）。
        # engine 的嵌套 options 仍是 dict（residual），顶层字段 mutation 已封。
        for ds in self._datasets.values():
            for fname in (
                "schema",
                "roles",
                "engine",
                "params_schema",
                "param_specs",
            ):
                val = getattr(ds, fname, None)
                if isinstance(val, dict) and not isinstance(val, MappingProxyType):
                    object.__setattr__(ds, fname, MappingProxyType(val))

    def get(self, name: str) -> Dataset:
        if name not in self._datasets:
            available = sorted(self._datasets)
            raise ValidationError(
                f"数据集 '{name}' 未注册。已注册：{available}\n"
                f"如需新增，请编辑 data_access/config/datasets.yaml"
            )
        return self._datasets[name]

    def __contains__(self, name: str) -> bool:
        return name in self._datasets

    def __iter__(self):
        return iter(self._datasets.values())

    def names(self) -> list[str]:
        return sorted(self._datasets)

    def allowed_roots(self) -> list[Path]:
        """收集所有数据集的可白名单化根目录，供 PathAuthorizer 使用。

        #P0-49 ParametricDataset 优先用显式 ``authorized_root``（比静态前缀更窄、
        更安全），缺省回退 static_root。
        按**当前 context** namespace 解析后返回（诊断/测试用）；PathAuthorizer
        构造用 ``allowed_root_templates``（保留占位符、延迟解析）。
        """
        roots: list[Path] = []
        for ds in self._datasets.values():
            if isinstance(ds, StaticDataset):
                roots.append(Path(resolve_namespace_path(str(ds.root))))
            elif isinstance(ds, ParametricDataset):
                raw = ds.authorized_root or ds.static_root
                roots.append(Path(resolve_namespace_path(str(raw))))
        return roots

    def allowed_root_templates(self) -> list[str]:
        """unresolved root template（``${RUN_NAMESPACE}`` 占位符原样保留）。

        #7 PathAuthorizer 需要这个：store 级对象不再把 namespace 烘焙进白名单，
        ``resolve_and_authorize`` 时按**当前请求**的 namespace 解析——长期 worker
        换 ``DataAccessSession`` 后白名单随之生效，不再与数据集路径错配。
        """
        roots: list[str] = []
        for ds in self._datasets.values():
            if isinstance(ds, StaticDataset):
                roots.append(str(ds.root))
            elif isinstance(ds, ParametricDataset):
                raw = ds.authorized_root or ds.static_root
                roots.append(str(raw))
        return roots


def load_registry(config_path: str | Path | None = None) -> DatasetRegistry:
    """从 YAML 文件加载登记表。

    如果 config_path 为 None，按以下顺序找：
        1. 环境变量 DATA_ACCESS_CONFIG
        2. data_access/config/datasets.yaml（本模块目录下）
    """
    if config_path is None:
        env_path = os.environ.get("DATA_ACCESS_CONFIG")
        if env_path:
            config_path = env_path
        else:
            config_path = Path(__file__).resolve().parent.parent / "config" / "datasets.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise ValidationError(f"datasets.yaml 不存在：{config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = strict_yaml_load(f.read(), context=str(config_path)) or {}

    if not isinstance(raw, dict):
        raise ValidationError(f"{config_path}: 顶层必须是 mapping（数据集名 → 配置）")

    datasets = {
        name: _parse_dataset(name, body)
        for name, body in raw.items()
        if not name.startswith("_")  # 跳过 YAML 锚点模板键（如 _ashare_lqtp_defaults）
    }
    return DatasetRegistry(datasets)
