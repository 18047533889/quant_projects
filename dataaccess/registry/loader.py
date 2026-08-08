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
from typing import Any, Mapping

from data_access.core.exceptions import ValidationError
from .yaml_loader import strict_yaml_load
from .layout_policy import LayoutPolicy, parse_layout_policy
from .params_validation import ParamSpec, parse_params_schema, validate_params
from data_access.read.query_budget import DatasetQueryPolicy, parse_dataset_query_policy
from data_access.read.formats import FormatSpec, default_glob_for_format, normalize_format_name
from data_access.core.namespace import resolve_namespace
from .paths import canonicalize, expand_env


# access_mode 三态：
#   published   —— 全员只读的正式数据（massive_parquet、factor_lake 快照等）
#   namespaced  —— 个人实验产物，路径自动带 ${RUN_NAMESPACE}，互不干扰
#   staging     —— 发布前的暂存区，写入仍按 namespace 隔离，校验通过后由 publish()
#                  原子切换到 published_root（PR1 不实现 publish，只登记字段占位）
_VALID_ACCESS_MODES = {"published", "namespaced", "staging"}
_VALID_LAYOUTS = {"plain", "hive"}


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
    # ---- 通用 Data IO Layer 新增字段（全部带默认值，向后兼容） ----
    format_spec: FormatSpec = field(default_factory=FormatSpec)   # 物理文件格式（parquet/csv/...）
    storage: dict[str, Any] | None = None                         # storage backend 声明
    roles: Mapping[str, str] = field(default_factory=dict)        # 语义角色（event_time/instrument/...）
    partitioning: dict[str, Any] | None = None                    # 时间/分区裁剪声明
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
        return [str(self.root / self.glob)]


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
    # ---- 通用 Data IO Layer 新增字段（全部带默认值，向后兼容） ----
    format_spec: FormatSpec = field(default_factory=FormatSpec)   # 物理文件格式（parquet/csv/...）
    storage: dict[str, Any] | None = None                         # storage backend 声明
    roles: Mapping[str, str] = field(default_factory=dict)        # 语义角色（event_time/instrument/...）
    partitioning: dict[str, Any] | None = None                    # 时间/分区裁剪声明
    semantic: str | None = None                                   # panel/event/factor/...
    engine: dict[str, Any] | None = None                          # 优先/兜底执行引擎
    schema_version: str | None = None                             # #36 schema 版本号
    authorized_root: Path | None = None                           # #P0-49 显式授权根

    @property
    def kind(self) -> str:
        return "parametric"

    def resolve_paths(self, **params: Any) -> list[str]:
        """用 params 填模板，返回给 DuckDB 的 glob 表达式。"""
        specs = self.param_specs or {
            k: ParamSpec(name=k, type=t) for k, t in self.params_schema.items()
        }
        validated = validate_params(self.name, specs, params)
        try:
            root = self.root_template.format(**validated)
            glob_part = self.glob_template.format(**validated)
        except KeyError as exc:
            raise ValidationError(f"模板变量缺失：{exc}") from exc
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
        # 向后兼容：静态前缀（模板里第一个 { 之前部分）
        expanded = expand_env(str(static_prefix))
        cleaned = expanded.replace("${RUN_NAMESPACE}", resolve_namespace())
        return canonicalize(cleaned.rstrip("/") or "/")
    if not isinstance(ar, str) or not ar.strip():
        raise ValidationError(f"{context}: authorized_root 必须是非空字符串")
    return canonicalize(expand_env(ar.strip()))


_UNRESOLVED_ENV_RE = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}")
_UNRESOLVED_DOLLAR_RE = re.compile(r"(?<!\\)\$[A-Za-z_][A-Za-z0-9_]*")


def _assert_no_unresolved_env(payload: Any, *, context: str) -> None:
    """#P0-47 registry compile 后任何残留 ``${...}`` / ``$VAR`` → 启动失败。

    ``expand_env`` 未设 env 且无 default 时保留原样——不能把配置问题变成奇怪的
    路径问题，生产必须启动即失败。
    """

    def walk(node: Any, path: str) -> None:
        if isinstance(node, str):
            m = _UNRESOLVED_ENV_RE.search(node) or _UNRESOLVED_DOLLAR_RE.search(node)
            if m:
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

    walk(payload, "$")


def _parse_dataset(name: str, raw: dict) -> Dataset:
    """解析单个数据集条目。raw 是 YAML 里该数据集 name 下的字典。"""
    context = f"数据集 '{name}'"

    kind = raw.get("kind", "static")
    if kind not in {"static", "parametric"}:
        raise ValidationError(f"{context}: kind 必须是 static/parametric，收到 {kind!r}")

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
    storage = raw.get("storage")
    if storage is not None and not isinstance(storage, dict):
        raise ValidationError(
            f"{context}: storage 必须是 mapping，收到 {type(storage).__name__}"
        )
    partitioning = raw.get("partitioning")
    if partitioning is not None and not isinstance(partitioning, dict):
        raise ValidationError(
            f"{context}: partitioning 必须是 mapping，收到 {type(partitioning).__name__}"
        )
    semantic = raw.get("semantic")
    if semantic is not None and not isinstance(semantic, str):
        raise ValidationError(f"{context}: semantic 必须是字符串")
    engine = raw.get("engine")
    if engine is not None and not isinstance(engine, dict):
        raise ValidationError(f"{context}: engine 必须是 mapping，收到 {type(engine).__name__}")

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

    if kind == "static":
        # published 数据集的 root 是直接路径；namespaced 数据集在 static 里
        # 也允许，但要手动把 ${RUN_NAMESPACE} 展开
        root_raw = _require_str(raw, "root", context=context)
        root_expanded = expand_env(root_raw)
        # namespaced 数据集的 root 里可能含 ${RUN_NAMESPACE}，展开一次
        if "${RUN_NAMESPACE}" in root_expanded or "{run_namespace}" in root_expanded.lower():
            ns = resolve_namespace()
            root_expanded = root_expanded.replace("${RUN_NAMESPACE}", ns)
        root = canonicalize(root_expanded)
        glob = raw.get("glob", default_glob_for_format(format_spec.type))
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
        )
    root_template_raw = _require_str(raw, "root_template", context=context)
    root_template = expand_env(root_template_raw)
    if "${RUN_NAMESPACE}" in root_template:
        root_template = root_template.replace("${RUN_NAMESPACE}", resolve_namespace())
    glob_template = raw.get("glob_template", default_glob_for_format(format_spec.type))
    params_schema_raw = raw.get("params_schema", {})
    if not isinstance(params_schema_raw, dict) or not params_schema_raw:
        raise ValidationError(f"{context}: parametric 数据集必须声明非空 params_schema")
    param_specs = parse_params_schema(params_schema_raw)
    params_schema = {k: spec.type for k, spec in param_specs.items()}

    # #P0-49 算出静态前缀用于白名单：优先显式 ``authorized_root``；缺省时才用
    # 「模板里第一个 { 之前的部分」（保守但过宽——如 /data/{market}/{factor_id}
    # 会授权整个 /data）。新数据集请显式声明 authorized_root；本层保留静态前缀
    # 作为向后兼容回退。
    first_brace = root_template.find("{")
    static_prefix = root_template[:first_brace] if first_brace >= 0 else root_template
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
        layout_policy=layout_policy,
    )


class DatasetRegistry:
    """数据集登记表容器。由 store.get_store() 加载一次，全进程共用。"""

    def __init__(self, datasets: Mapping[str, Dataset]) -> None:
        self._datasets = dict(datasets)

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
        """收集所有数据集的可白名单化根目录，供 PathAuthorizer 使用。"""
        roots: list[Path] = []
        for ds in self._datasets.values():
            if isinstance(ds, StaticDataset):
                roots.append(ds.root)
            elif isinstance(ds, ParametricDataset):
                roots.append(ds.static_root)
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
    _assert_no_unresolved_env(raw, context=str(config_path))

    datasets = {
        name: _parse_dataset(name, body)
        for name, body in raw.items()
        if not name.startswith("_")  # 跳过 YAML 锚点模板键（如 _ashare_lqtp_defaults）
    }
    return DatasetRegistry(datasets)
