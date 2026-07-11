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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from .exceptions import ValidationError
from .layout_policy import LayoutPolicy, parse_layout_policy
from .params_validation import ParamSpec, parse_params_schema, validate_params
from .query_budget import DatasetQueryPolicy, parse_dataset_query_policy
from .namespace import resolve_namespace
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
    """所有数据集的共同字段。"""
    name: str
    access_mode: str                  # published / namespaced / staging
    layout: str                       # plain / hive
    time_column: str                  # 业务约定的时间列名，供结构化谓词用
    instrument_column: str            # 业务约定的标的列名
    hive_partitioning: bool           # 传给 DuckDB read_parquet(hive_partitioning=)
    union_by_name: bool               # 传给 DuckDB read_parquet(union_by_name=)
    # PR8 注：schema 字段放在 StaticDataset / ParametricDataset 上而不是这里，
    # 因为 dataclass + frozen + 继承对字段默认值有严格顺序要求（非默认字段不能
    # 跟在默认字段之后）。放 base 上会污染子类的非默认字段，所以两个子类各自带。


@dataclass(frozen=True)
class StaticDataset(DatasetBase):
    """「根一次定死」的数据集：启动时算出 glob，read 时直接用。"""
    root: Path                        # 已 canonicalize 的绝对路径
    glob: str                         # 相对 root 的 glob，如 "**/*.parquet"
    # PR8：可选 schema，格式 {列名: 类型字符串}，首访自检用。空 = 不校验。
    schema: Mapping[str, str] = field(default_factory=dict)
    query_policy: DatasetQueryPolicy = field(default_factory=DatasetQueryPolicy)
    partition_columns: tuple[str, ...] = field(default_factory=lambda: ("year",))
    storage_format: str = "long"
    layout_policy: LayoutPolicy | None = None

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
    partition_columns: tuple[str, ...] = field(default_factory=lambda: ("year",))
    storage_format: str = "long"
    layout_policy: LayoutPolicy | None = None

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


def _parse_partition_columns(raw: Any, *, context: str) -> tuple[str, ...]:
    if raw is None:
        return ("year",)
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        raise ValidationError(
            f"{context}: partition_columns 必须是字符串列表，收到 {type(raw).__name__}"
        )
    cols = tuple(str(c) for c in raw if str(c))
    return cols or ("year",)


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

    time_column = _require_str(raw, "time_column", context=context)
    instrument_column = _require_str(raw, "instrument_column", context=context)

    hive_partitioning = bool(raw.get("hive_partitioning", layout == "hive"))
    union_by_name = bool(raw.get("union_by_name", False))

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
        glob = raw.get("glob", "**/*.parquet")
        return StaticDataset(
            name=name,
            access_mode=access_mode,
            layout=layout,
            time_column=time_column,
            instrument_column=instrument_column,
            hive_partitioning=hive_partitioning,
            union_by_name=union_by_name,
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
    glob_template = raw.get("glob_template", "**/*.parquet")
    params_schema_raw = raw.get("params_schema", {})
    if not isinstance(params_schema_raw, dict) or not params_schema_raw:
        raise ValidationError(f"{context}: parametric 数据集必须声明非空 params_schema")
    param_specs = parse_params_schema(params_schema_raw)
    params_schema = {k: spec.type for k, spec in param_specs.items()}

    # 算出静态前缀用于白名单：把模板里第一个 { 之前的部分当做可白名单化的根。
    # 这是一个保守但够用的做法；如果以后需要更细的路径鉴权再升级。
    first_brace = root_template.find("{")
    static_prefix = root_template[:first_brace] if first_brace >= 0 else root_template
    static_root = canonicalize(static_prefix.rstrip("/") or "/")

    return ParametricDataset(
        name=name,
        access_mode=access_mode,
        layout=layout,
        time_column=time_column,
        instrument_column=instrument_column,
        hive_partitioning=hive_partitioning,
        union_by_name=union_by_name,
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
            config_path = Path(__file__).parent / "config" / "datasets.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise ValidationError(f"datasets.yaml 不存在：{config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValidationError(f"{config_path}: 顶层必须是 mapping（数据集名 → 配置）")

    datasets = {
        name: _parse_dataset(name, body)
        for name, body in raw.items()
        if not name.startswith("_")  # 跳过 YAML 锚点模板键（如 _ashare_lqtp_defaults）
    }
    return DatasetRegistry(datasets)
