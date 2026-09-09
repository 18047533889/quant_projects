"""FactorEngine ↔ cleaned_operators execution bridge.

The bridge converts engine Series/panels to the selected physical operator
backend and normalises results back to the FactorEngine panel contract.
Production backend selection is evidence constrained and workload-cost aware.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .pandas_compat import pd
from factor_engine.planner.logical_plan import PlanNode
from .context import ExecutionContext
from .panel_native import panel_native_enabled, to_panel
from factor_engine.cache.panel_cache import series_panel_cache_key


@dataclass
class _RecipePanelSource:
    """Request-local source used by the canonical plan executor."""

    panel: Any
    column_name: str = "__recipe_input__"
    load_count: int = 0

    def load_column(self, name: str):
        if name != self.column_name:
            raise KeyError(name)
        self.load_count += 1
        series = self.panel.stack(future_stack=True)
        series.index = series.index.set_names(("timestamp", "instrument"))
        series.name = name
        return series

    def load_columns(self, names):
        return {name: self.load_column(name) for name in names}


def compile_operator_recipe(steps) -> PlanNode:
    """Compile ordered canonical operators into one existing logical-plan DAG.

    ``steps`` contains ``(canonical_operator, scalar_parameters)`` pairs. The
    input column is loaded once and every subsequent node consumes the prior
    node. Unknown operators fail before execution.
    """
    ensure_cleaned_loaded()
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    node = PlanNode("column", attrs={"name": "__recipe_input__"})
    for canonical, parameters in steps:
        resolved = OperatorRegistry.resolve_canonical_strict(str(canonical))
        if OperatorRegistry.get(resolved) is None:
            raise ValueError(f"unknown recipe operator {canonical!r}")
        node = PlanNode(resolved, inputs=(node,), attrs=dict(parameters))
    return node


def execute_operator_recipe(panel, steps, *, runtime_stats=None):
    """Execute a multi-step recipe through one PlanNode/PandasBackend session."""
    if not isinstance(panel, pd.DataFrame):
        raise TypeError("recipe panel must be a pandas DataFrame")
    source = _RecipePanelSource(panel)
    stats = runtime_stats if runtime_stats is not None else {}
    plan = compile_operator_recipe(steps)
    from factor_engine.backend.pandas_backend import PandasBackend

    ctx = ExecutionContext(data_source=source, run_mode="research", runtime_stats=stats)
    result = PandasBackend().execute(plan, ctx)
    stats["recipe_input_load_count"] = source.load_count
    stats["recipe_operator_count"] = len(tuple(steps))
    if isinstance(result, pd.Series) and isinstance(result.index, pd.MultiIndex):
        return result.unstack(level=-1)
    if isinstance(result, pd.DataFrame):
        return result
    raise TypeError(f"recipe execution returned unsupported type {type(result).__name__}")


class NoCertifiedParameterRegionError(ValueError):
    """R40 #156: a production operator has NO certified parameter region for the
    selected backend — fail closed instead of the legacy ``coverage_skip``
    pass-through."""


class ParameterCertificationInfrastructureError(RuntimeError):
    """R40 #157/#159: a certification phase failed for an infrastructure reason
    (store load / bound-call / semantic-version resolver / identity
    serialization).  Production hard-fails; research degrades with a reason."""


class BoundOperatorCallIncompleteError(ParameterCertificationInfrastructureError):
    """R40 #158: the bound operator call is incomplete (defaults/positional/
    alias-normalized parameters missing) in a production path."""


# ---------------------------------------------------------------------------
# R40 #158: complete bound scalar parameters.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _BoundScalarParams:
    """A fully bound scalar parameter point + completeness flag (R40 #158).

    Built by a single ``bind_operator_call`` (no broad ``except`` fallback): the
    certification query and the kernel call share the SAME bound values, and
    ``complete`` is ``True`` only when every declared scalar parameter resolves
    in the normalized bound (defaults/positional/alias-normalized all included).
    """

    normalized: dict[str, Any]
    complete: bool
    missing: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# R40 #160: typed input dtype signature for the certification key.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class InputDType:
    """One input's ordered dtype signature (R40 #160)."""

    name: str
    dtype: str
    nullable: bool


@dataclass(frozen=True)
class InputDTypeSignature:
    """Ordered (param role, dtype, nullable) signature of every panel input.

    The certification key binds the FULL ordered input signature (not just the
    first non-object column), so two calls with different input dtypes/roles do
    not share a certification space.
    """

    inputs: tuple[InputDType, ...] = ()

    def to_key(self) -> str:
        return ";".join(
            f"{d.name}={d.dtype}:{'n' if d.nullable else 'o'}" for d in self.inputs
        ) or "no_inputs"


# ---------------------------------------------------------------------------
# R40 #161: execution-variant identity from the ACTUALLY SELECTED implementation.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExecutionVariantIdentity:
    """Which physical implementation the certified execution used (R40 #161).

    Replaces the hard-coded ``"reference"``: certification/lineage/runtime-event
    consumers bind on the real backend + implementation id + kernel variant +
    fusion flag + code hash so a fast/Numba path certifies its own space.
    """

    backend: str
    implementation_id: str
    kernel_variant: str
    fused: bool
    code_hash: str

    @classmethod
    def of(cls, operator: Any, backend: str, canonical: str) -> "ExecutionVariantIdentity":
        impl_id = f"{type(operator).__module__}.{type(operator).__qualname__}"
        if str(backend).lower() == "polars":
            kernel_variant = "native_polars"
        elif "numba" in str(type(operator).__module__).lower():
            kernel_variant = "numba"
        else:
            kernel_variant = "reference"
        code_hash = hashlib.sha256(
            f"{impl_id}|{canonical}|{kernel_variant}".encode("utf-8")
        ).hexdigest()[:16]
        return cls(
            backend=str(backend),
            implementation_id=impl_id,
            kernel_variant=kernel_variant,
            fused=False,
            code_hash=code_hash,
        )

    def to_key(self) -> str:
        return (
            f"{self.backend}:{self.implementation_id}:{self.kernel_variant}:"
            f"{'f' if self.fused else 'u'}:{self.code_hash}"
        )


def ensure_cleaned_loaded() -> None:
    """Load the operator registry via the thread-safe bootstrap (R40 #155).

    The legacy ``_CLEANED_LOADED`` unlocked global is removed: every call
    delegates to ``cleaned_operators.REGISTRY_BOOTSTRAP.ensure_ready(...)`` so
    concurrent callers WAIT for a fully-loaded registry and never return a
    half-loaded one.
    """
    from factor_engine.cleaned_operators import REGISTRY_BOOTSTRAP

    REGISTRY_BOOTSTRAP.ensure_ready(include_research=True)


def ensure_operator_registry(*, surface: str = "production") -> None:
    """Surface-aware registry bootstrap (R40 #155).

    Production callers MUST use this entry point: it freezes the surface on the
    first bootstrap call (``include_research=False`` for a research-free
    production surface) and enforces the production signature authority.  A
    bare ``ensure_cleaned_loaded()`` (research default) from an analyzer or
    bridge helper never silently pre-loads a research surface for a production
    caller that uses this entry.
    """
    from factor_engine.cleaned_operators import REGISTRY_BOOTSTRAP, check_signature_authority

    surface = str(surface or "production").lower()
    if surface == "production":
        REGISTRY_BOOTSTRAP.ensure_ready(include_research=False)
        check_signature_authority(production=True)
    else:
        REGISTRY_BOOTSTRAP.ensure_ready(include_research=True)


def series_to_panel(s: pd.Series, ctx: ExecutionContext) -> pd.DataFrame:
    if not isinstance(s.index, pd.MultiIndex):
        raise TypeError("cleaned_bridge expects MultiIndex (timestamp, instrument) Series")
    cache = ctx.panel_cache
    cache_key = series_panel_cache_key(s)
    if cache is not None:
        hit = cache.get(cache_key)
        if hit is not None:
            return hit
    tcol, icol = ctx.timestamp_col, ctx.instrument_col
    if s.index.names != [tcol, icol]:
        s = s.copy()
        s.index = s.index.set_names([tcol, icol])
    panel = s.unstack(level=icol)
    if not panel.index.is_monotonic_increasing:
        panel = panel.sort_index()
    if cache is not None:
        cache[cache_key] = panel
    return panel


def panel_to_series(
    panel: pd.DataFrame,
    ctx: ExecutionContext,
    *,
    template: "pd.Series | pd.Index | None" = None,
) -> pd.Series:
    # The common dense numeric root need not rebuild pandas' general stack
    # machinery for every factor. Keep unusual axes/dtypes/metadata on that
    # reference path, and copy values so a caller cannot mutate cached panels.
    simple_columns = (
        type(panel.columns) in (pd.Index, pd.RangeIndex)
        and panel.columns.is_unique and not panel.columns.hasnans
        and (panel.columns.dtype.kind in "iu" or (
            panel.columns.dtype == np.dtype("object")
            and all(type(column) is str for column in panel.columns)
        ))
    )
    if (
        type(panel) is pd.DataFrame and not panel.empty
        and type(panel.index) is pd.DatetimeIndex
        and panel.index.is_unique and not panel.index.hasnans
        and simple_columns and not panel.attrs and panel.flags.allows_duplicate_labels
        and all(dtype == np.dtype("float64") for dtype in panel.dtypes)
    ):
        axis = pd.MultiIndex.from_product([panel.index, panel.columns])
        values = np.array(panel.to_numpy(copy=False), order="C", copy=True).reshape(-1)
        stacked = pd.Series(values, index=axis)
    else:
        stacked = panel.stack(future_stack=True)
    stacked.index.names = [ctx.timestamp_col, ctx.instrument_col]
    if template is None:
        return stacked
    if isinstance(template, pd.Index):
        target = template
    else:
        target = template.index
    if len(stacked) == len(target) and stacked.index.equals(target):
        return stacked
    return stacked.reindex(target)


def _resolve_canonical(op: str) -> str:
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    return OperatorRegistry.resolve_canonical_strict(op)


def _call_cleaned_operator(canonical: str, operator: Any, call_args: list[Any], kw: dict[str, Any]) -> Any:
    from factor_engine.backend.parameter_aliases import reject_runtime_parameter_aliases
    reject_runtime_parameter_aliases(canonical, kw)
    # R5 P0-01 (defense-in-depth): the dispatch layer runs the central logical
    # call validator for every cleaned operator, so an operator that overrides
    # ``calculate`` directly (bypassing ``SeriesOperator._prepare_call``) still
    # cannot escape the integer ParamSpec / panel-axis / validate_params gate.
    # SeriesOperator-derived operators re-validate inside ``calculate``; that
    # second pass is pure and cheap.  Polars panels are left untouched by the
    # pandas-axis validator (only ``pd.DataFrame`` frames are checked).
    from factor_engine.cleaned_operators.base import validate_operator_call
    metadata = getattr(operator, "metadata", None)
    if metadata is not None:
        validate_operator_call(operator, tuple(call_args), dict(kw))
    return operator.calculate(*call_args, **kw)


def _operator_backend_preference(ctx: ExecutionContext) -> str:
    perf = getattr(ctx, "perf", None)
    if perf is not None and getattr(perf, "operator_backend", None):
        return str(perf.operator_backend)
    return "auto"


def _data_source_kind(ctx: ExecutionContext) -> str:
    ds = getattr(ctx, "data_source", None)
    seen: set[int] = set()
    while ds is not None and id(ds) not in seen:
        seen.add(id(ds))
        name = type(ds).__name__.lower()
        if "clickhouse" in name:
            return "clickhouse"
        if "duckdb" in name or "data_access" in name or "parquet" in name:
            return "duckdb"
        ds = getattr(ds, "inner", None) or getattr(ds, "_inner", None)
    return "memory"


def _bound_scalar_parameters(operator: Any, call_args: list[Any], kw: dict[str, Any]) -> _BoundScalarParams:
    """R39 #27 + R40 #158：single ``bind_operator_call`` → complete BoundOperatorCall.

    显式参数 + kernel 默认参数合成为**完整 canonical BoundParameterPoint**
    （alias 解析到 canonical 名、类型归一），供参数域认证查询。这样
    ``ts_mean(x)``（window 走默认值、attrs 为空）不再跳过认证——它和
    ``ts_mean(x, window=20)`` 一样，用完整 ``{window: 20}`` 参数点查询。

    R40 #158：删除 ``except Exception:`` 回退到显式 kwargs 的分支——bind 失败
    （非法参数、alias 归一错误、类型转换错误）直接上抛，由调用方按
    production/research 判定 hard-fail 或降级。``complete`` 为 ``True`` 仅当每个
    声明的 scalar 参数都在归一化 bound 中解析（默认/位置/alias 全含）。
    """
    from factor_engine.cleaned_operators.base import bind_operator_call, _kernel_param_defaults

    meta = getattr(operator, "metadata", None)
    names = list(getattr(meta, "param_names", None) or [])
    defaults = _kernel_param_defaults(operator)
    call = bind_operator_call(operator, tuple(call_args), dict(kw), defaults=defaults)
    nv = call.bound.normalized_values
    inactive = set(getattr(call.bound, "inactive_params", frozenset()))
    normalized: dict[str, Any] = {
        name: nv[name]
        for name in names
        if name in nv and not isinstance(nv[name], (pd.Series, pd.DataFrame))
    }
    missing = tuple(
        name for name in names
        if name not in nv
        and name not in inactive
        and not isinstance(nv.get(name), (pd.Series, pd.DataFrame))
    )
    return _BoundScalarParams(normalized=normalized, complete=not missing, missing=missing)


def _operator_semantic_version(canonical: str) -> str:
    """R39 #28 + R40 #159：认证 key 的 semantic_version 维度。

    ``versioned_name`` 反映算子语义版本（例如 ``ts_corr@v2``）。R40 #159：空串
    不再是合法 sentinel——production 解析失败直接抛
    ``ParameterCertificationInfrastructureError``（不是 ``""`` 混进认证空间）。
    """
    from factor_engine.backend.operator_semantic_version import versioned_name

    try:
        return versioned_name(canonical)
    except Exception as exc:
        raise ParameterCertificationInfrastructureError(
            f"semantic-version resolver failed for canonical {canonical!r}: "
            f"{type(exc).__name__}: {exc} (R40 #159 — empty string is not a "
            "valid certification identity)"
        ) from exc


def _execution_variant(operator: Any, backend: str, canonical: str) -> ExecutionVariantIdentity:
    """R40 #161：认证 key 的 execution_variant 维度——从实际选定实现产生。

    R37 独立 oracle 认证的是 reference 变体；R40 替换硬编码 ``"reference"``，
    由 ``ExecutionVariantIdentity.of(operator, backend, canonical)`` 从实际选定
    实现（implementation id / kernel variant / fusion / code hash）产生。
    """
    return ExecutionVariantIdentity.of(operator, backend, canonical)


def _input_dtype(evaluated: list[Any]) -> InputDTypeSignature:
    """R39 #28 + R40 #160：认证 key 的 dtype 维度——有序输入 dtype 签名。

    旧的实现只取第一个非 object/category 列。R40 #160：认证 key 绑定每个输入
    的 (param role, dtype, nullable) 有序签名——不同输入角色/dtype/nullable 不
    能共用同一认证空间。
    """
    from .panel_polars import is_polars_frame

    inputs: list[InputDType] = []
    for val in evaluated:
        if isinstance(val, pd.Series):
            inputs.append(InputDType(name=val.name or "", dtype=str(val.dtype),
                                     nullable=bool(val.isna().any())))
        elif isinstance(val, pd.DataFrame):
            for col in val.columns:
                dt = val[col].dtype
                inputs.append(InputDType(name=str(col), dtype=str(dt),
                                         nullable=bool(val[col].isna().any())))
        elif is_polars_frame(val):
            for col in val.columns:
                dt = val.schema[col]
                try:
                    nulls = int(val[col].null_count() or 0) > 0
                except Exception:
                    nulls = False
                inputs.append(InputDType(name=str(col), dtype=str(dt), nullable=nulls))
    return InputDTypeSignature(tuple(inputs))


def _estimate_row_count(evaluated: list[Any]) -> int | None:
    """Estimate scalar operator workload after child evaluation.

    A wide panel is costed by cells rather than dates because Pandas↔Polars
    conversion and elementwise/rolling work scale with date×instrument cells.
    """
    from .panel_polars import is_polars_frame

    estimates: list[int] = []
    for val in evaluated:
        if isinstance(val, pd.Series):
            estimates.append(int(len(val)))
        elif isinstance(val, pd.DataFrame):
            estimates.append(int(val.shape[0] * max(1, val.shape[1])))
        elif is_polars_frame(val):
            height = int(getattr(val, "height", 0) or 0)
            width = int(getattr(val, "width", 1) or 1)
            estimates.append(height * max(1, width))
    return max(estimates) if estimates else None


def _record_uncertified(canonical: str, reason: str, *, ctx: ExecutionContext) -> None:
    """Best-effort telemetry for a research-mode certification degradation."""
    try:
        # Resource snapshots belong to the run boundary/sampler, not every
        # operator invocation. Keep the degradation observable in the root's
        # own context; the old snapshot return value was discarded entirely.
        if ctx.runtime_stats is None:
            ctx.runtime_stats = {}
        stats = ctx.runtime_stats
        stats["param_domain_membership_skipped"] = canonical
        stats["parameter_certification_degradation_reason"] = reason
        stats["parameter_certification_degradation_count"] = int(
            stats.get("parameter_certification_degradation_count", 0)
        ) + 1
    except Exception:
        pass


def _record_backend_route(
    ctx: ExecutionContext,
    *,
    canonical: str,
    backend: str,
    row_count_estimate: int | None,
) -> None:
    """R40 #162: thread-safe backend-route telemetry.

    The old implementation did a dict read-modify-write on the shared
    ``ctx.runtime_stats`` dict — two concurrent jobs sharing the context dropped
    counts.  Counts now live in the request-scoped ``ExecutionPerfCounters``
    (atomic under a lock) under ``backend_route:<canonical>:<backend>``; the
    informational "last route" record is written through the same counter object
    so telemetry never mutates semantic context and never races.
    """
    from .panel_polars import get_request_perf_counters

    perf = get_request_perf_counters()
    perf.incr(f"backend_route:{canonical}:{backend}")
    perf.incr("backend_route_total")
    # NOTE: the legacy ``ctx.runtime_stats["last_operator_backend_route"]``
    # single-slot record is deliberately NOT written — a shared last-write-wins
    # slot overwrites another factor's route (audit C25 blocker).  Route counts
    # are request-scoped; per-canonical last-route data can be derived from the
    # count keys by a per-job collector.


def _resolve_operator(
    canonical: str,
    ctx: ExecutionContext,
    *,
    row_count_estimate: int | None = None,
):
    # R21-P022: if the cost optimizer has already selected a backend for this
    # whole plan, honor it and skip per-operator BackendRouter.select(auto).
    selected_backend = getattr(ctx, "selected_backend", None)
    if selected_backend is not None:
        from factor_engine.backend.operator_capability import resolve_canonical as _rc
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        name = _rc(canonical)
        try:
            operator = OperatorRegistry.get(name)
        except Exception:
            operator = None
        # Map whole-plan backend labels to operator-level backend keys.
        _backend_map = {
            "polars_panel": "polars",
            "polars_long": "polars",
            "duckdb_sql": "sql",
            "clickhouse_sql": "sql",
            "hybrid": "auto",
            "pandas_numpy": "pandas_numpy",
        }
        op_backend = _backend_map.get(selected_backend, selected_backend)
        _record_backend_route(
            ctx,
            canonical=name,
            backend=op_backend,
            row_count_estimate=row_count_estimate,
        )
        return operator, op_backend

    from factor_engine.backend.backend_router import BackendRouter

    prefer = _operator_backend_preference(ctx)
    run_mode = str(getattr(ctx, "run_mode", "research") or "research").lower()
    fallback = "allow" if getattr(ctx, "production_fallback_policy", "error") == "warn" else "error"
    selection = BackendRouter.select(
        canonical,
        requested_backend=prefer,
        run_mode=run_mode,
        fallback_policy=fallback,
        data_source_kind=_data_source_kind(ctx),
        row_count_estimate=row_count_estimate,
        allow_unverified_backend=run_mode != "production",
    )
    _record_backend_route(
        ctx,
        canonical=selection.canonical,
        backend=selection.backend,
        row_count_estimate=row_count_estimate,
    )
    return selection.operator, selection.backend


def _record_polars_op(ctx: ExecutionContext, op: str) -> None:
    runtime = getattr(ctx, "runtime_stats", None) or {}
    cache_stats = runtime.get("cache")
    if cache_stats is not None and hasattr(cache_stats, "record_polars_op"):
        cache_stats.record_polars_op(op)


def _prepare_call_args(
    evaluated: list[Any],
    ctx: ExecutionContext,
    *,
    backend: str,
    broadcast_scalars: bool = False,
) -> tuple[list[Any], pd.Series | None, pd.DataFrame | None]:
    from .panel_polars import is_polars_frame

    call_args: list[Any] = []
    template: pd.Series | None = None
    template_panel: pd.DataFrame | None = None
    for val in evaluated:
        if isinstance(val, (pd.Series, pd.DataFrame)) or is_polars_frame(val):
            panel = to_panel(val, ctx) if isinstance(val, (pd.Series, pd.DataFrame)) else val
            if is_polars_frame(panel):
                if template_panel is None and isinstance(val, pd.Series):
                    template_panel = val.unstack(level=ctx.instrument_col)
                call_args.append(panel)
            else:
                if template_panel is None:
                    template_panel = panel
                if backend == "polars":
                    from .panel_polars import panel_to_polars
                    call_args.append(panel_to_polars(panel))
                else:
                    call_args.append(panel)
            if template is None and isinstance(val, pd.Series):
                template = val
            elif template is None and _ctx_template(ctx) is not None:
                template = _ctx_template(ctx)
        else:
            if broadcast_scalars and template_panel is not None:
                scalar_panel = pd.DataFrame(
                    val, index=template_panel.index, columns=template_panel.columns
                )
                if backend == "polars":
                    # WS-B #245: keep every panel in a polars call a polars frame
                    # carrying the same ``__fe_time__`` metadata column so the
                    # central validator / PanelIdentity gate sees one consistent
                    # axis (a bare pandas frame would be column-misaligned with
                    # the ``__fe_time__``-carrying polars inputs).
                    from .panel_polars import panel_to_polars
                    call_args.append(panel_to_polars(scalar_panel))
                else:
                    call_args.append(scalar_panel)
            else:
                # Scalar literals used by panel operators (e.g. ts_delay(..., n))
                # remain scalar; cleaned operators validate them directly.
                call_args.append(val)
    return call_args, template, template_panel


def _target_index(template: Any) -> Any:
    """从 Series 或 MultiIndex 模板提取对齐用的 target index（R16）。"""
    if isinstance(template, pd.Index):
        return template
    return template.index


def _ctx_template(ctx: ExecutionContext) -> Any:
    """Phase 5 R16：优先取 axis-only ``template_index``，兼容旧 ``template_series``。"""
    idx = getattr(ctx, "template_index", None)
    if idx is not None:
        return idx
    return getattr(ctx, "template_series", None)


class GrainTransformCertificate:
    """R40 #163: certificate for a grain-changing (downsampling) operator result.

    ``input_grain`` / ``output_grain`` are the declared grains (e.g.
    ``minute`` -> ``daily``); ``calendar_id`` / ``calendar_version`` /
    ``timezone`` / ``session_id`` identify the session calendar used for the
    mapping; ``mapping_policy`` is the declared output-date policy
    (``session_end`` / ``session_open`` / ``last_bar_of_session`` / ...).
    The certificate is computed by :func:`validate_grain_transform` and is
    independent of the panel-content checks (instrument axis / monotonic /
    duplicate / future).
    """

    __slots__ = (
        "input_grain", "output_grain", "calendar_id", "calendar_version",
        "timezone", "session_id", "mapping_policy",
    )

    def __init__(
        self,
        *,
        input_grain: str,
        output_grain: str,
        calendar_id: str,
        calendar_version: str,
        timezone: str,
        session_id: str,
        mapping_policy: str,
    ) -> None:
        self.input_grain = input_grain
        self.output_grain = output_grain
        self.calendar_id = calendar_id
        self.calendar_version = calendar_version
        self.timezone = timezone
        self.session_id = session_id
        self.mapping_policy = mapping_policy

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_grain": self.input_grain,
            "output_grain": self.output_grain,
            "calendar_id": self.calendar_id,
            "calendar_version": self.calendar_version,
            "timezone": self.timezone,
            "session_id": self.session_id,
            "mapping_policy": self.mapping_policy,
        }


#: R40 #163: 8-check grain-transform validation.  Checks that do not need a
#: concrete calendar provider (monotonic / no-duplicate / no-future /
#: instrument-axis-preserved / declared-grain match / output-in-input coverage)
#: always run; market-session checks (each output date maps to a real market
#: session, calendar version consistency) run when a calendar provider is given.
def validate_grain_transform(
    result: pd.DataFrame,
    template_panel: pd.DataFrame,
    certificate: GrainTransformCertificate,
    *,
    calendar: Any = None,
) -> list[str]:
    """Return the list of grain-transform validation violations (R40 #163)."""
    from factor_engine.backend.operator_errors import OperatorShapeError

    errors: list[str] = []
    # 1. output index is a DatetimeIndex
    if not isinstance(result.index, pd.DatetimeIndex):
        errors.append(f"downsampled result index is {type(result.index).__name__}, not DatetimeIndex")
    else:
        # 2. monotonic (non-decreasing)
        if not result.index.is_monotonic_increasing:
            errors.append("downsampled result dates are not monotonic")
        # 3. no duplicates
        if not result.index.is_unique:
            errors.append("downsampled result dates are not unique")
        # 4. no future dates (relative to the input panel's last date)
        if template_panel is not None and len(template_panel) and len(result):
            if result.index.max() > template_panel.index.max():
                errors.append("downsampled result contains dates beyond the input panel")
        # 5. output dates within input session coverage
        if template_panel is not None and len(template_panel):
            if not result.index.isin(template_panel.index).all():
                errors.append(
                    "downsampled output dates are not a subset of the input session dates"
                )
    # 6. instrument axis preserved
    if template_panel is not None:
        if list(result.columns) != list(template_panel.columns):
            errors.append("downsampled result does not preserve the instrument axis")
    # 7. declared grain mapping (minute -> daily, no upsampling)
    if len(result) > len(template_panel) if template_panel is not None else False:
        errors.append("downsampled result has MORE rows than the input (undeclared upsampling)")
    # 8. market-session mapping (needs a calendar provider)
    if calendar is not None:
        session_dates = getattr(calendar, "session_dates", None)
        if callable(session_dates):
            try:
                valid = set(session_dates())
            except Exception:
                valid = None
            if valid is not None:
                for d in result.index:
                    if d.date() not in valid:
                        errors.append(f"output date {d.date()} is not a real market session")
        if not certificate.calendar_id or not certificate.calendar_version:
            errors.append("grain transform has no declared calendar id/version")
    return errors


def _validate_downsampled_result(
    result: pd.DataFrame, template_panel: pd.DataFrame
) -> None:
    """Validate a shape-changing (downsampled) operator result (R11 P0-04).

    The result is a DIFFERENT frequency than the input (e.g. minute -> daily),
    so the exact index-match contract does not apply.  It must still be a
    well-formed panel: same instrument columns as the input, a unique index, and
    — for a time downsampling — fewer or equal rows (never MORE than the input,
    which would be an upsampling the operator did not declare).

    R40 #163: the full 8-check grain-transform certificate validation is exposed
    as :func:`validate_grain_transform` (callers with a concrete calendar
    provider opt in); this entry keeps the original structural checks so the
    legacy minute->daily path is unchanged.
    """
    from factor_engine.backend.operator_errors import OperatorShapeError

    if not isinstance(result.index, (pd.DatetimeIndex,)):
        raise OperatorShapeError(
            f"downsampled operator result index {type(result.index).__name__} is "
            "not a DatetimeIndex (R11 P0-04 fail-closed)"
        )
    if not result.index.is_unique:
        raise OperatorShapeError(
            "downsampled operator result has duplicate dates (R11 P0-04)"
        )
    if list(result.columns) != list(template_panel.columns):
        raise OperatorShapeError(
            f"downsampled operator result columns {list(result.columns)} do not "
            f"match the input panel {list(template_panel.columns)} — a daily "
            "aggregation must preserve the instrument axis (R11 P0-04)"
        )
    if len(result) > len(template_panel):
        raise OperatorShapeError(
            f"downsampled operator produced {len(result)} rows from an input of "
            f"{len(template_panel)} — an undeclared UPSAMPLING is not allowed "
            "(R11 P0-04 fail-closed)"
        )


def _validate_no_extra_output_columns(result: Any, template_panel: pd.DataFrame) -> None:
    """R40 #207: an axis-preserving single-output operator must NOT return value
    columns beyond the input panel — a silent extra column would be dropped by
    the representation boundary and hide a kernel bug.

    Multi-output operators are exempt via an explicit ``OutputColumnContract``
    (declared on the operator metadata as ``output_column_contract``).
    """
    from factor_engine.backend.operator_errors import OperatorShapeError
    from factor_engine.cleaned_operators.common._polars_bridge import FE_TIME_COL, SKIP

    try:
        value_cols = [str(c) for c in result.columns if str(c) not in SKIP and str(c) != FE_TIME_COL]
    except AttributeError:
        return  # not a frame-like result; other shape checks cover it
    template_value_cols = {str(c) for c in template_panel.columns}
    extra = [c for c in value_cols if c not in template_value_cols]
    if extra:
        raise OperatorShapeError(
            f"operator returned {len(extra)} extra output column(s) {extra} beyond "
            f"the input panel's instrument columns — an axis-preserving "
            "single-output operator must preserve the instrument axis exactly "
            "(R40 #207)"
        )


def _normalize_operator_result(
    result: Any,
    *,
    backend: str,
    template: "pd.Series | pd.Index",
    template_panel: pd.DataFrame | None,
    ctx: ExecutionContext,
    operator: Any = None,
) -> Any:
    if backend == "polars":
        from .panel_polars import is_polars_frame, polars_to_panel
        if is_polars_frame(result):
            if template_panel is None:
                raise ValueError("polars operator requires a DataFrame template panel")
            # R40 #207: check the RAW polars frame for extra value columns BEFORE
            # the conversion silently drops them.
            _meta = getattr(operator, "metadata", None)
            _oc = getattr(_meta, "output_column_contract", None)
            if template_panel is not None and _oc is None:
                _ig = getattr(_meta, "input_grain", None)
                _og = getattr(_meta, "output_grain", None)
                if not (_ig and _og and _ig != _og):
                    _validate_no_extra_output_columns(result, template_panel)
            _run_mode_strict = str(getattr(ctx, "run_mode", "research") or "research").lower() == "production"
            result = polars_to_panel(result, template=template_panel, strict=_run_mode_strict)

    from factor_engine.backend.operator_errors import OperatorShapeError
    # R11 P0-04: a shape-changing operator (declared input_grain != output_grain,
    # e.g. minute -> daily) legitimately returns a DIFFERENT-frequency panel, so
    # the exact-index-match check below does not apply.  ``output_grain`` /
    # ``input_grain`` live on the operator metadata and are also mirrored on the
    # catalog contract by the registry.
    grain_changing = False
    if operator is not None:
        _meta = getattr(operator, "metadata", None)
        if _meta is not None:
            _ig = getattr(_meta, "input_grain", None)
            _og = getattr(_meta, "output_grain", None)
            grain_changing = bool(_ig and _og and _ig != _og)
    if isinstance(result, pd.DataFrame):
        if template_panel is None:
            raise OperatorShapeError("DataFrame result requires a panel template")
        if grain_changing:
            _validate_downsampled_result(result, template_panel)
            if panel_native_enabled(ctx):
                return result
            # A daily (or other downsampled) result must NOT be reindexed onto
            # the minute input template — that would blank every daily value
            # (no matching minute keys).  Stack on the result's OWN axis.
            return panel_to_series(result, ctx, template=None)
        if not result.index.equals(template_panel.index):
            raise OperatorShapeError("operator DataFrame index does not match input panel")
        if not result.columns.equals(template_panel.columns):
            raise OperatorShapeError("operator DataFrame columns do not match input panel")
        if panel_native_enabled(ctx):
            return result
        return panel_to_series(result, ctx, template=template)
    target = _target_index(template)
    if isinstance(result, pd.Series):
        if isinstance(result.index, pd.MultiIndex):
            if not result.index.equals(target):
                raise OperatorShapeError("operator result index does not match the input template")
            return result
        if len(result) != len(target):
            raise OperatorShapeError(f"operator result length {len(result)} does not match template {len(target)}")
        if not result.index.equals(target):
            raise OperatorShapeError("operator returned an indexless/misaligned Series")
        return result
    if pd.api.types.is_scalar(result):
        return pd.Series(result, index=target)
    array = np.asarray(result)
    if array.ndim == 1:
        if len(array) != len(target):
            raise OperatorShapeError(f"operator 1D result length {len(array)} does not match template {len(target)}")
        # R20-100..102: use the canonical ``_target_index(template)`` for the 1D
        # branch too — ``template`` may itself be a ``pd.Index`` (axis-only
        # template), in which case ``template.index`` is wrong.
        return pd.Series(array, index=target)
    if array.ndim == 2:
        if template_panel is None:
            raise OperatorShapeError("operator 2D result requires a panel template")
        if tuple(array.shape) != tuple(template_panel.shape):
            raise OperatorShapeError(f"operator 2D result shape {array.shape} does not match panel {template_panel.shape}")
        frame = pd.DataFrame(array, index=template_panel.index, columns=template_panel.columns)
        if panel_native_enabled(ctx):
            return frame
        return panel_to_series(frame, ctx, template=template)
    raise OperatorShapeError(f"unsupported operator result type/shape: {type(result).__name__} {array.shape}")


def make_cleaned_kernel(eval_fn: Callable[[PlanNode, ExecutionContext], Any], op: str):
    def _kernel(node: PlanNode, ctx: ExecutionContext) -> pd.Series:
        ensure_cleaned_loaded()
        canonical = _resolve_canonical(op)

        # Evaluate children first so routing sees the actual workload size rather
        # than a hard-coded global row estimate.
        evaluated: list[Any] = [eval_fn(child, ctx) for child in node.inputs]
        row_count_estimate = _estimate_row_count(evaluated)
        operator, backend = _resolve_operator(
            canonical,
            ctx,
            row_count_estimate=row_count_estimate,
        )
        if operator is None:
            raise NotImplementedError(f"cleaned operator not implemented: {op!r}")
        if backend == "polars":
            _record_polars_op(ctx, op)

        kw = dict(node.attrs)
        # Plan-inference metadata must never become operator kwargs (R5-06).
        # ``dtype`` is stamped by the analyzer from ``OPERATOR_SIGNATURES`` output
        # typing (e.g. Series[Bool] -> 'bool') and is not a declared parameter of
        # any operator — forwarding it makes an undeclared hidden kwarg that the
        # validator rejects.  ``semantic_attrs`` already carries field metadata
        # separately; ``dtype`` is the one inference key that leaked into attrs.
        kw.pop("dtype", None)

        call_args, template, template_panel = _prepare_call_args(
            evaluated, ctx, backend=backend,
            broadcast_scalars=canonical in {"maximum", "minimum"},
        )

        # R37-P0-009 + R39 #26/#27/#28/#29 + R40 #156-#161：production 执行前参数域
        # membership 门（typed phases —— BindCall → LoadCertificate →
        # ResolveCertificationIdentity → CheckMembership）。
        #   #26 run_mode 作为认证调用身份传入（认证层不再从 env 重猜）；
        #   #27 bound normalization——显式参数 + kernel 默认参数合成完整参数点；
        #   #28 认证 key 全维度（semantic_version/backend/variant/source_context/
        #       dtype/grain）真正用全；
        #   #29 production 强制装载证据（ensure_loaded），空 store / 旧 SHA store
        #       fail closed；
        #   #156 production 无认证区域 fail closed（不再 coverage_skip 放行）；
        #   #157 任一 infrastructure 阶段 production hard-fail（不再单一 broad except）；
        #   #158 单一 bound_call（complete flag）+ 认证与 kernel 共用同一 bound；
        #   #159 semantic_version 缺失/解析失败 hard-fail（不是空串）；
        #   #160 认证 key 绑定有序输入 dtype 签名；
        #   #161 execution_variant 从实际选定实现产生。
        _run_mode = str(getattr(ctx, "run_mode", "research") or "research").lower()
        _production = _run_mode == "production"
        try:
            from factor_engine.runtime.parameter_domain_store import (
                assert_parameter_point_certified,
                get_parameter_domain_store,
            )
            from factor_engine.runtime.exceptions import ParameterDomainError

            # Phase 1: BindCall (R40 #158) — single bind, no broad except.  An
            # incomplete bind hard-fails production (defaults/positional/alias
            # dropped) and degrades research.
            _bound = _bound_scalar_parameters(operator, call_args, kw)
            if _production and not _bound.complete:
                raise BoundOperatorCallIncompleteError(
                    f"production parameter bind incomplete for canonical "
                    f"{canonical!r}: missing {_bound.missing} (R40 #158 — an "
                    "incomplete parameter point cannot be certified)"
                )
            _point = _bound.normalized

            # Phase 2: LoadCertificate (R40 #157/#29).
            try:
                store = get_parameter_domain_store(ensure_loaded=_production)
            except Exception as exc:
                raise ParameterCertificationInfrastructureError(
                    f"parameter-domain store load failed for canonical {canonical!r}: "
                    f"{type(exc).__name__}: {exc} (R40 #157 — infrastructure "
                    "failure hard-fails production)"
                ) from exc

            # Phase 3: ResolveCertificationIdentity (R40 #159/#160/#161).
            semantic_version = _operator_semantic_version(canonical)
            variant = _execution_variant(operator, backend, canonical)
            dtype_sig = _input_dtype(evaluated)

            # Phase 4: CheckMembership.
            if store.operator_has_any_certified_region_by_backend(canonical, backend):
                assert_parameter_point_certified(
                    canonical, _point, backend=backend,
                    run_mode=_run_mode,
                    semantic_version=semantic_version,
                    execution_variant=variant.to_key(),
                    source_context=_data_source_kind(ctx),
                    dtype=dtype_sig.to_key(),
                    grain=str(getattr(ctx, "grain", "daily") or "daily"),
                    store=store,
                )
            elif _production:
                raise NoCertifiedParameterRegionError(
                    f"production operator {canonical!r} has no certified parameter "
                    f"region for backend {backend!r} (R40 #156 — no-certified-region "
                    "fails closed; a parameterless formal certification or a typed "
                    "ParameterCertificationExemption is required to skip)"
                )
            else:
                _record_uncertified(
                    canonical,
                    f"no certified region for backend {backend!r} (research degradation)",
                    ctx=ctx,
                )
        except (ParameterDomainError, NoCertifiedParameterRegionError,
                ParameterCertificationInfrastructureError):
            raise
        except BoundOperatorCallIncompleteError:
            raise  # production-only; re-raise typed path
        except Exception as exc:
            # Research-only degradation for unexpected infrastructure errors;
            # production already hard-failed above (every production error path
            # raises a typed exception that the except clause above re-raises).
            _record_uncertified(
                canonical, f"{type(exc).__name__}: {exc} (research degradation)", ctx=ctx
            )
        # M11: transport only caller-owned execution identity across the public
        # operator boundary.  Instrument/window coordinates remain the model
        # producer's responsibility; absent authority stays None.
        from factor_engine.cleaned_operators.ts_model._rolling_core import (
            FitScope,
            fit_failure_receipts,
            fit_receipt_scope,
        )

        receipt_scope = FitScope(
            canonical=canonical,
            backend=backend,
            profile=getattr(ctx, "profile_id", None),
            execution_id=getattr(ctx, "execution_id", None),
            run_id=getattr(ctx, "run_id", None),
            task_id=getattr(ctx, "task_id", None),
            factor_id=getattr(ctx, "factor_id", None),
        )
        with fit_receipt_scope(receipt_scope):
            sink = getattr(ctx, "fit_failure_sink", None)
            if sink is None:
                result = _call_cleaned_operator(canonical, operator, call_args, kw)
            else:
                with fit_failure_receipts(sink):
                    result = _call_cleaned_operator(canonical, operator, call_args, kw)
        # Keep literal-only arithmetic scalar.  Promoting an intermediate such as
        # ``floor(window / 2) + 1`` to a panel makes it an invalid lag/window
        # argument when it is later consumed by a time-series operator.
        if template is None and template_panel is None and not any(
            isinstance(value, (pd.Series, pd.DataFrame))
            for value in evaluated
        ):
            return result
        if template is None:
            template = _ctx_template(ctx)
        if template is None:
            raise ValueError(f"cleaned op {op!r} requires at least one Series input")
        return _normalize_operator_result(
            result,
            backend=backend,
            template=template,
            template_panel=template_panel,
            ctx=ctx,
            operator=operator,
        )
    return _kernel


def list_cleaned_ops_for_backend(skip: set[str]) -> list[str]:
    ensure_cleaned_loaded()
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    names: set[str] = set()
    for canon in OperatorRegistry.list_canonical():
        if canon not in skip and OperatorRegistry.get(canon) is not None:
            names.add(canon)
    for alias, canon in OperatorRegistry._aliases.items():
        if alias not in skip and OperatorRegistry.get(canon) is not None:
            names.add(alias)
    # Compiler-internal lowering nodes (protected_div et al.) carry an
    # ``internal`` surface so production ``OperatorRegistry.get`` returns None,
    # but they ARE registered backend implementations and must receive a kernel
    # for the pandas backend — otherwise ``_eval`` hits ``KeyError`` at runtime
    # on a legitimate plan node (e.g. alias ``protected_div`` reachable from the
    # DSL).  ``_read_state`` gives us the raw operators dict; include every
    # registered name, then drop the production-surface gate here too.
    _operators = OperatorRegistry._operators
    for canon in OperatorRegistry.list_canonical():
        if canon not in names and canon in _operators and _operators[canon]:
            names.add(canon)
    for alias, canon in OperatorRegistry._aliases.items():
        if alias not in names and canon in _operators and _operators.get(canon):
            names.add(alias)
    return sorted(names)


def build_cleaned_dsl_allowlist(skip: set[str] | None = None, *, surface: str = "daily") -> dict[str, Any]:
    ensure_cleaned_loaded()
    from factor_engine.cleaned_operators.operator_surface import is_dsl_name_allowed
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    from factor_engine.api.cleaned_ops import make_cleaned_call_factory

    skip = skip or set()
    out: dict[str, Any] = {}
    for canon in OperatorRegistry.list_canonical():
        if canon in skip or canon in out or OperatorRegistry.get(canon) is None:
            continue
        if is_dsl_name_allowed(canon, canon, surface=surface):
            out[canon] = make_cleaned_call_factory(canon)
    for alias, canon in OperatorRegistry._aliases.items():
        if alias in skip or alias in out or OperatorRegistry.get(canon) is None:
            continue
        if is_dsl_name_allowed(alias, canon, surface=surface):
            out[alias] = make_cleaned_call_factory(canon)
    return out


def build_production_dsl_allowlist(skip: set[str] | None = None) -> dict[str, Any]:
    """Expose every production-admitted factor operator, not only daily core."""
    from factor_engine.cleaned_operators.operator_spec import build_operator_spec
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    full = build_cleaned_dsl_allowlist(skip, surface="all")
    out: dict[str, Any] = {}
    for name, factory in full.items():
        canon = OperatorRegistry._aliases.get(name, name)
        spec = build_operator_spec(canon)
        if spec is not None and spec.allow_in_production:
            out[name] = factory
    return out


build_cleaned_dsl_extensions = build_cleaned_dsl_allowlist
