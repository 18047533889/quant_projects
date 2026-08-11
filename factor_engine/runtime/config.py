"""Strongly typed FactorEngine YAML configuration."""
from __future__ import annotations
import json  # R40 #76: canonical_config_hash 依赖 json.dumps，此前模块级缺 import
import re  # R40 #83: canonical hash 的 DSN 凭据脱敏
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml
from workspace_paths import default_factor_lake_root, resolve_path

_PROFILES_DIR = Path(__file__).resolve().parent.parent / "examples" / "profiles"

@dataclass(frozen=True)
class FactorDefinitionConfig:
    name: str
    expr: str
    freq: str = "1d"
    universe: str | None = None
    description: str | None = None
    surface: str = "daily"
    dialect: str = "native"
    dialect_version: str | None = None

@dataclass(frozen=True)
class DataSourceConfig:
    type: str
    options: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class BackendConfig:
    type: str = "auto"

@dataclass(frozen=True)
class EngineConfig:
    enable_cache: bool = True
    plan_cache_dir: str | None = None

@dataclass(frozen=True)
class MarketExecutionSpec:
    """Independent market-execution spec (R10 #12).

    One ``calendar`` string can no longer carry every meaning: ``market`` is the
    MarketID (ashare/us), ``calendar_id`` the trading-calendar identity
    (SSE/SZSE/NYSE/NASDAQ), ``timezone`` the exchange timezone
    (Asia/Shanghai / America/New_York), ``decision_time`` when the bar's
    decision is taken (close), and ``bar_timestamp_role`` whether the bar index
    is bar-open or bar-end.  Used by PIT visibility / session mapping — never
    conflated into the calendar string.
    """

    market: str | None = None
    calendar_id: str | None = None
    timezone: str | None = None
    decision_time: str = "close"
    bar_timestamp_role: str = "bar_end"

    def as_market_execution(self) -> "MarketExecutionSpec":
        return self


@dataclass(frozen=True)
class RunConfig:
    mode: str = "research"
    auto_warmup: bool = False
    trim_warmup: bool = True
    # R10-P0-025: ``market`` is the MarketID (ashare/us); ``calendar`` is the
    # trading-calendar identity (SSE/SZSE/NYSE/NASDAQ).  They are resolved
    # separately (``runtime.config_runtime._resolve_market``) and never merged
    # into one variable.
    market: str | None = None
    calendar: str | None = None
    # R10 #12: independent timezone / decision-time / bar-timestamp-role, kept
    # out of the single ``calendar`` string.  ``market_execution`` is derived
    # from this run section.
    timezone: str | None = None
    decision_time: str = "close"
    bar_timestamp_role: str = "bar_end"

    @property
    def market_execution(self) -> MarketExecutionSpec:
        return MarketExecutionSpec(
            market=self.market,
            calendar_id=self.calendar,
            timezone=self.timezone,
            decision_time=self.decision_time,
            bar_timestamp_role=self.bar_timestamp_role,
        )

@dataclass(frozen=True)
class DQConfig:
    profile: str | None = None
    strict: bool = False
    input_strict: bool = True

@dataclass(frozen=True)
class PITConfig:
    enforce: bool = False
    forbid_forward_fill: bool = False

@dataclass(frozen=True)
class IncrementalConfig:
    since: str | None = None
    end_date: str | None = None
    lookback_extra: int = 5
    recompute_tail_bars: int | None = None

@dataclass(frozen=True)
class MaterializationConfig:
    lake_root: str | None = None
    factor_id: str | None = None
    author: str | None = None
    frequency: str | None = None
    description: str | None = None
    expression: str | None = None
    target: str = "local"
    preserve_invalid_rows: bool = False
    value_dtype: str = "float32"
    isolate_partition_failures: bool = True
    resume_materialize: bool = False
    ch_ensure_table: bool = True
    incremental: IncrementalConfig | None = None
    clickhouse_table: str | None = None
    clickhouse_host: str | None = None
    clickhouse_port: int | None = None
    clickhouse_database: str | None = None
    clickhouse_username: str | None = None
    clickhouse_password: str | None = None
    clickhouse_secure: bool | None = None
    staging_dataset: str = "factor_lake_staging"
    storage_format: str = "long"
    partition_columns: tuple[str, ...] | None = None

@dataclass(frozen=True)
class PipelineConfig:
    batched_engine: bool = False

@dataclass(frozen=True)
class FactorEngineConfig:
    factor: FactorDefinitionConfig
    data_source: DataSourceConfig
    backend: BackendConfig = field(default_factory=BackendConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    materialization: MaterializationConfig | None = None
    run: RunConfig = field(default_factory=RunConfig)
    dq: DQConfig = field(default_factory=DQConfig)
    pit: PITConfig = field(default_factory=PITConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

def _parse_partition_columns(raw: Any) -> tuple[str, ...] | None:
    if raw is None: return None
    if isinstance(raw, str): raw=[raw]
    if not isinstance(raw,(list,tuple)): raise ValueError("materialization.partition_columns 必须是字符串列表")
    cols=tuple(str(c) for c in raw if str(c)); return cols or None

def _resolve_optional_path(value: Any, *, base_dir: Path) -> str | None:
    if value is None: return None
    text=str(value).strip()
    return str(resolve_path(text,base_dir=base_dir)) if text else None

_TRUE_STRINGS = {"true", "1", "yes", "on"}
_FALSE_STRINGS = {"false", "0", "no", "off"}


def _strict_bool(value: Any, *, name: str) -> bool:
    """Strict boolean coercion (R10 #10).

    ``bool("false")`` is True in Python — a YAML ``auto_warmup: "false"`` string
    would silently enable a gate.  Accept real bools, 0/1, and the canonical
    string set; anything else raises instead of Python-truthiness coercion.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, float) and value in (0.0, 1.0):
        return bool(value)
    if isinstance(value, str):
        low = value.strip().lower()
        if low in _TRUE_STRINGS:
            return True
        if low in _FALSE_STRINGS:
            return False
    raise ValueError(
        f"config field {name!r}: expected a boolean (true/false/0/1), got {value!r}"
    )


def _strict_int(value: Any, *, name: str) -> int:
    """Strict integer coercion (R10 #11).

    ``int(5.9) -> 5`` silently truncates a fractional config value.  Reject
    non-integral values instead; accept ints, integral floats, and integral
    strings.
    """
    if isinstance(value, bool):
        raise ValueError(f"config field {name!r}: bool is not an integer: {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise ValueError(
            f"config field {name!r}: fractional integer {value!r} would be "
            "truncated — provide an integral value"
        )
    if isinstance(value, str):
        text = value.strip()
        try:
            return int(text)
        except ValueError as exc:
            raise ValueError(
                f"config field {name!r}: expected an integer, got {value!r}"
            ) from exc
    raise ValueError(f"config field {name!r}: expected an integer, got {value!r}")


def _string_to_string_dict(value: Any, *, name: str) -> dict[str, str]:
    """严格校验 string→string 映射（fields/joins/aliases）。"""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping, got {type(value).__name__}")
    out: dict[str, str] = {}
    for k, v in value.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError(
                f"{name} entries must be string→string, got {k!r}→{v!r}"
            )
        out[k] = v
    return out


@dataclass(frozen=True)
class TypedDataSourceOptions:
    """R40 #135: nested source config（composite/production options）的 typed coercion。

    ``load_config`` 里 data_source 的 options（含 composite ``sources`` 下每个
    子源的 options）统一经 :meth:`from_dict` 严格类型化：布尔走 ``_strict_bool``、
    整数走 ``_strict_int``、已知结构字段做类型校验，未知键透传进 ``extra``
    （避免过度拒绝破坏既有配置）。``to_dict()`` 还原普通 dict —— 下游
    ``build_data_source_config`` 仍按 dict 展开（config.data_source.options 保持
    dict 契约）。
    """

    dataset: str | None = None
    fields: dict[str, str] | None = None
    root: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    timestamp_col: str | None = None
    instrument_col: str | None = None
    anchor: str | None = None
    anchor_column: str | None = None
    sources: dict[str, "TypedDataSourceOptions"] | None = None
    joins: dict[str, str] | None = None
    aliases: dict[str, str] | None = None
    max_files: int | None = None
    read_auto: bool = False
    snapshot_only: bool = False
    normalize_timestamp: bool = False
    join_policy: str | None = None
    usage: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Any) -> "TypedDataSourceOptions":
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise ValueError(
                f"data_source options must be a mapping, got {type(raw).__name__}"
            )
        out: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        for k, v in raw.items():
            if v is None:
                continue
            key = str(k)
            if key in {
                "dataset", "root", "start_date", "end_date", "timestamp_col",
                "instrument_col", "anchor", "anchor_column", "join_policy", "usage",
            }:
                out[key] = str(v)
            elif key in {"read_auto", "snapshot_only", "normalize_timestamp"}:
                out[key] = _strict_bool(v, name=f"data_source.options.{key}")
            elif key == "max_files":
                out[key] = _strict_int(v, name="data_source.options.max_files")
            elif key in {"fields", "joins", "aliases"}:
                out[key] = _string_to_string_dict(
                    v, name=f"data_source.options.{key}"
                )
            elif key == "sources":
                if not isinstance(v, dict):
                    raise ValueError(
                        "data_source.options.sources must be a mapping of nested sources"
                    )
                out[key] = {
                    str(name): cls.from_dict(sub) for name, sub in v.items()
                }
            else:
                extra[key] = v
        return cls(**out, extra=extra)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = dict(self.extra)
        for k in self.__dataclass_fields__:
            if k == "extra":
                continue
            v = getattr(self, k)
            if v is None:
                continue
            if k == "sources":
                out[k] = {name: sub.to_dict() for name, sub in v.items()}
            else:
                out[k] = v
        return out


def _forbid_unknown(section: dict[str, Any], allowed: set[str], *, name: str) -> None:
    """Fail-fast on unknown keys (R10 #9).

    ``extra="forbid"`` semantics: a typo like ``run: {mdoe: production}`` is
    silently ignored today (``mode`` falls back to research — dangerous for a
    production intent).  Any key not in ``allowed`` raises instead.
    """
    unknown = sorted(k for k in section if k not in allowed)
    if unknown:
        raise ValueError(
            f"config section {name!r} has unknown key(s): {', '.join(unknown)}. "
            f"Allowed: {', '.join(sorted(allowed))}"
        )


def _deep_merge(base: dict[str,Any], override: dict[str,Any]) -> dict[str,Any]:
    merged=dict(base)
    for key,value in override.items():
        merged[key]=_deep_merge(merged[key],value) if key in merged and isinstance(merged[key],dict) and isinstance(value,dict) else value
    return merged

def _resolve_profile_path(profile: str) -> Path:
    name=str(profile).strip()
    if not name: raise ValueError("profile name must be non-empty")
    path=_PROFILES_DIR/f"{name}.yaml"
    if not path.is_file(): raise FileNotFoundError(f"Unknown profile '{name}': {path}")
    return path

def load_profile(profile: str) -> dict[str,Any]:
    path=_resolve_profile_path(profile); payload=yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload,dict): raise ValueError(f"Profile file must be a mapping: {path}")
    return payload

def _factor_dialect(factor_payload: dict[str,Any]) -> tuple[str,str,str|None]:
    surface=str(factor_payload.get("surface") or factor_payload.get("dsl_surface") or "daily")
    dialect=str(factor_payload.get("dialect") or "native").lower(); version=factor_payload.get("dialect_version")
    if surface=="lqtp" and dialect=="native": dialect="lqtp"
    if dialect not in {"native","lqtp"}: raise ValueError(f"unsupported factor.dialect={dialect!r}")
    if dialect=="lqtp":
        from api.lqtp_compat import DEFAULT_LQTP_DIALECT_VERSION
        version=str(version or DEFAULT_LQTP_DIALECT_VERSION)
        if version!=DEFAULT_LQTP_DIALECT_VERSION: raise ValueError(f"unsupported factor.dialect_version={version!r}; supported={DEFAULT_LQTP_DIALECT_VERSION!r}")
    elif version is not None: raise ValueError("factor.dialect_version is only valid when factor.dialect='lqtp'")
    return surface,dialect,version

CONFIG_SCHEMA_VERSION = 1


#: R40 #78(d): 缺失 ``schema_version`` 的显式 legacy 策略。缺省允许（向后兼容
#: 历史配置文件），但这是**显式策略决策**，不是「静默当 current」——缺失时按
#: legacy(schema_version=0) 处理；``allow_missing=False`` 时缺失直接拒绝。
_ALLOW_MISSING_SCHEMA_VERSION = True


def _validate_config_schema_version(
    payload: dict[str, Any], *, allow_missing: bool = True
) -> None:
    """R21-217..219 + R40 #78: config schema_version 严格校验。

    - ``schema_version`` 缺失 => 显式 legacy 策略（``allow_missing`` 缺省放行，
      但绝不会被当作 current）；``allow_missing=False`` 时缺失直接报错；
    - bool 值（``True``/``False`` 在 Python 是 int）必须拒绝 —— ``True`` 曾
      被 ``int(True)=1`` 误判为 current；
    - 负数 / 0 拒绝；
    - 更新于支持版本 => 拒绝（前向迁移必须显式）；
    - ``0 < version < CONFIG_SCHEMA_VERSION`` => 拒绝并给出迁移指引（本 repo
      尚无迁移链，绝不静默跳过）。
    """
    version = payload.get("schema_version")
    if version is None:
        if allow_missing:
            return  # 显式 legacy 策略：按 schema_version=0（无迁移链）处理
        raise ValueError(
            "config schema_version is missing; set schema_version=<current> or "
            "declare an explicit legacy-schema policy"
        )
    if isinstance(version, bool):
        raise ValueError(
            f"config schema_version must be an integer, got bool {version!r} "
            "(True/False are ints in Python and would silently pass as 1/0)"
        )
    try:
        version_int = int(version)
    except (TypeError, ValueError):
        raise ValueError(f"config schema_version must be an integer, got {version!r}") from None
    if version_int < 0:
        raise ValueError(
            f"config schema_version={version_int} is negative; must be >= 1"
        )
    if version_int == 0:
        raise ValueError(
            f"config schema_version=0 is invalid; use schema_version="
            f"{CONFIG_SCHEMA_VERSION} or declare an explicit legacy-schema policy"
        )
    if version_int > CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"config schema_version={version_int} is newer than supported "
            f"{CONFIG_SCHEMA_VERSION}; run an explicit config migration"
        )
    if version_int < CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"config schema_version={version_int} is older than supported "
            f"{CONFIG_SCHEMA_VERSION}; a migration chain is required — run the "
            "explicit config migration before loading"
        )


#: R40 #83: canonical hash 前必须剥除的敏感字段名子串（与 ``service.errors``
#: 的边界脱敏保持同一策略：password/token/secret/api_key/… 绝不进 hash 依赖）。
_SECRET_KEY_SUBSTRINGS = (
    "password", "token", "secret", "api_key", "apikey", "authorization",
    "dsn", "access_key", "private_key",
)

_DSN_CREDENTIAL_RE = re.compile(r"(?i)([a-z0-9+]+://)([^/\s:@]+)(:[^/\s@]+)?@")


def _redact_dsn_credentials(text: str) -> str:
    """剥掉 URL/DSN 字符串里的 ``user:password@`` 凭据段。"""
    return _DSN_CREDENTIAL_RE.sub(r"\1<redacted>@", str(text))


def _is_secret_key(key: Any) -> bool:
    lowered = str(key).lower()
    return any(seg in lowered for seg in _SECRET_KEY_SUBSTRINGS)


def _redact_secrets_for_hash(obj: Any) -> Any:
    """递归脱敏后返回**可哈希 canonical** 结构（R40 #83）。

    - 字段名含 ``*password*``/``*token*``/``*secret*``/``*api_key*`` 等 => 值替换
      为 ``"<redacted>"``（不同凭据 → 相同 hash）；
    - 字符串里的 DSN 凭据（``scheme://user:pass@``）=> 剥除；
    - 其余结构原样递归，保持确定性排序。
    """
    from dataclasses import asdict, is_dataclass

    if is_dataclass(obj) and not isinstance(obj, type):
        return {
            k: _redact_secrets_for_hash(v)
            for k, v in asdict(obj).items()
            if v is not None
        }
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in sorted(obj.items()):
            if v is None:
                continue
            out[k] = "<redacted>" if _is_secret_key(k) else _redact_secrets_for_hash(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_redact_secrets_for_hash(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, str):
        return _redact_dsn_credentials(obj)
    return obj


def canonical_config_hash(config: "FactorEngineConfig") -> str:
    """R21-220 + R40 #83: reproducible canonical hash of a parsed config.

    先经 :func:`_redact_secrets_for_hash` 剥除密码/token/secret/api_key/DSN
    凭据再 hashing —— 同一配置仅凭据不同 ⇒ 相同 hash（hash 永远不携带机密）。
    """
    import hashlib

    canonical = json.dumps(
        _redact_secrets_for_hash(config),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_config(path: str|Path, *, profile: str|None=None) -> FactorEngineConfig:
    config_path=Path(path); payload=yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload,dict): raise ValueError(f"Config must be a mapping: {config_path}")
    profile_name=profile or payload.get("profile")
    if profile_name:
        profile_payload = load_profile(str(profile_name))
        # R40 #79: profile 文件可能携带「documented, not consumed」的
        # data_access/label 段 —— 合并前剥掉，让 merged payload 通过严格
        # top-level allowed set；用户自己配置里的 data_access/label 仍会被拒。
        for _k in ("data_access", "label"):
            profile_payload.pop(_k, None)
        payload=_deep_merge(profile_payload,payload)
    # R40 #77: schema_version 校验必须在 **merge 之后** —— 否则 profile 注入的
    # 更高 schema_version 字段可绕过 schema 门（raw payload 校验时还没有该字段）。
    _validate_config_schema_version(payload)
    base_dir=config_path.parent.resolve()
    # R10 #9: extra="forbid" semantics — a typo like ``run: {mdoe: production}``
    # must fail fast instead of silently falling back to research.
    # R40 #79: data_access/label 从 allowed set 移除（DataAccess 由独立
    # dataaccess/ 包自管，loader 不消费这两个段）→ 出现即 fail-fast。
    _forbid_unknown(
        payload,
        {
            "factor","data_source","backend","engine","run","dq","pit",
            "pipeline","materialization","materialize","profile",
            "schema_version",
        },
        name="config",
    )
    fp=payload.get("factor",{}); dsp=payload.get("data_source",{}); bp=payload.get("backend",{}); ep=payload.get("engine",{}); rp=payload.get("run",{}); dqp=payload.get("dq",{}); pp=payload.get("pit",{}); pipe=payload.get("pipeline",{}); mp=payload.get("materialization",payload.get("materialize"))
    if not isinstance(fp,dict) or "name" not in fp or "expr" not in fp: raise ValueError("Config must include factor.name and factor.expr")
    if not isinstance(dsp,dict) or "type" not in dsp: raise ValueError("Config must include data_source.type")
    _forbid_unknown(fp, {"name","expr","freq","universe","description","surface","dsl_surface","dialect","dialect_version"}, name="factor")
    _forbid_unknown(bp, {"type"}, name="backend")
    _forbid_unknown(ep, {"enable_cache","plan_cache_dir"}, name="engine")
    _forbid_unknown(rp, {"mode","auto_warmup","trim_warmup","market","calendar","timezone","decision_time","bar_timestamp_role"}, name="run")
    _forbid_unknown(dqp, {"profile","strict","input_strict"}, name="dq")
    _forbid_unknown(pp, {"enforce","forbid_forward_fill"}, name="pit")
    _forbid_unknown(pipe, {"batched_engine"}, name="pipeline")
    canonical_surface,dialect,version=_factor_dialect(fp)
    # FactorEngine.from_loaded_config in the current public API still consumes
    # one parser-surface argument. Preserve the orthogonal metadata on the config
    # while bridging LQTP configs through the legacy surface token until that API
    # is removed.
    parser_surface="lqtp" if dialect=="lqtp" else canonical_surface
    factor_config=FactorDefinitionConfig(name=str(fp["name"]),expr=str(fp["expr"]),freq=str(fp.get("freq","1d")),universe=fp.get("universe"),description=fp.get("description"),surface=parser_surface,dialect=dialect,dialect_version=version)
    # R40 #135: nested source options（含 composite sources 下每个子源）统一走
    # TypedDataSourceOptions typed coercion（严格布尔/整数/结构校验），再还原为
    # dict —— 下游 build_data_source_config 按 dict 展开的契约不变。
    data_source_config=DataSourceConfig(
        type=str(dsp["type"]),
        options=TypedDataSourceOptions.from_dict(
            {k:v for k,v in dsp.items() if k!="type"}
        ).to_dict(),
    )
    backend_config=BackendConfig(type=str(bp.get("type","auto")))
    engine_config=EngineConfig(enable_cache=_strict_bool(ep.get("enable_cache",True),name="engine.enable_cache"),plan_cache_dir=_resolve_optional_path(ep.get("plan_cache_dir"),base_dir=base_dir))
    materialization_config=None
    if mp is not None:
        if not isinstance(mp,dict): raise ValueError("Config materialization section must be a mapping")
        _forbid_unknown(
            mp,
            {
                "lake_root","factor_id","author","frequency","description",
                "expression","target","preserve_invalid_rows","value_dtype",
                "isolate_partition_failures","resume_materialize","ch_ensure_table",
                "incremental","clickhouse_table","clickhouse_host","clickhouse_port",
                "clickhouse_database","clickhouse_username","clickhouse_password",
                "clickhouse_secure","staging_dataset","storage_format",
                "partition_columns",
            },
            name="materialization",
        )
        inc=mp.get("incremental")
        if inc is not None and not isinstance(inc,dict):
            raise ValueError("Config materialization.incremental must be a mapping")
        if isinstance(inc,dict):
            _forbid_unknown(inc, {"since","end_date","lookback_extra","recompute_tail_bars"}, name="materialization.incremental")
        materialization_config=MaterializationConfig(
            lake_root=_resolve_optional_path(mp.get("lake_root"),base_dir=base_dir) or str(default_factor_lake_root()),factor_id=mp.get("factor_id"),author=mp.get("author"),frequency=mp.get("frequency"),description=mp.get("description"),expression=mp.get("expression"),target=str(mp.get("target","local")),preserve_invalid_rows=_strict_bool(mp.get("preserve_invalid_rows",False),name="materialization.preserve_invalid_rows"),value_dtype=str(mp.get("value_dtype","float32")),isolate_partition_failures=_strict_bool(mp.get("isolate_partition_failures",True),name="materialization.isolate_partition_failures"),resume_materialize=_strict_bool(mp.get("resume_materialize",False),name="materialization.resume_materialize"),ch_ensure_table=_strict_bool(mp.get("ch_ensure_table",True),name="materialization.ch_ensure_table"),
            incremental=IncrementalConfig(since=inc.get("since"),end_date=inc.get("end_date"),lookback_extra=_strict_int(inc.get("lookback_extra",5),name="materialization.incremental.lookback_extra"),recompute_tail_bars=inc.get("recompute_tail_bars")) if isinstance(inc,dict) else None,
            clickhouse_table=mp.get("clickhouse_table"),clickhouse_host=mp.get("clickhouse_host"),clickhouse_port=mp.get("clickhouse_port"),clickhouse_database=mp.get("clickhouse_database"),clickhouse_username=mp.get("clickhouse_username"),clickhouse_password=mp.get("clickhouse_password"),clickhouse_secure=mp.get("clickhouse_secure"),staging_dataset=str(mp.get("staging_dataset","factor_lake_staging")),storage_format=str(mp.get("storage_format","long")),partition_columns=_parse_partition_columns(mp.get("partition_columns")))
    return FactorEngineConfig(factor=factor_config,data_source=data_source_config,backend=backend_config,engine=engine_config,materialization=materialization_config,
        run=RunConfig(mode=str(rp.get("mode","research")),auto_warmup=_strict_bool(rp.get("auto_warmup",False),name="run.auto_warmup"),trim_warmup=_strict_bool(rp.get("trim_warmup",True),name="run.trim_warmup"),market=rp.get("market"),calendar=rp.get("calendar"),timezone=rp.get("timezone"),decision_time=str(rp.get("decision_time","close")),bar_timestamp_role=str(rp.get("bar_timestamp_role","bar_end"))),
        dq=DQConfig(profile=dqp.get("profile"),strict=_strict_bool(dqp.get("strict",False),name="dq.strict"),input_strict=_strict_bool(dqp.get("input_strict",True),name="dq.input_strict")),
        pit=PITConfig(enforce=_strict_bool(pp.get("enforce",False),name="pit.enforce"),forbid_forward_fill=_strict_bool(pp.get("forbid_forward_fill",False),name="pit.forbid_forward_fill")),pipeline=PipelineConfig(batched_engine=_strict_bool(pipe.get("batched_engine",False),name="pipeline.batched_engine")))
