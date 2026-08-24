# -*- coding: utf-8 -*-
"""Shared helpers for audited operator overrides."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import Operator as PandasOperator
from factor_engine.cleaned_operators.base import OperatorMetadata as PandasMetadata
from factor_engine.cleaned_operators.base_polars import Operator as PolarsOperator
from factor_engine.cleaned_operators.base_polars import OperatorMetadata as PolarsMetadata
from factor_engine.cleaned_operators.registry import OperatorRegistry

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None  # type: ignore

SKIP_COLUMNS = frozenset({"date", "stock_code", "ts", "inst", "instrument", "timestamp"})
EPS = 1e-12


def positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        out = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a positive integer") from exc
    if out < 1 or float(value) != float(out):
        raise ValueError(f"{name} must be a positive integer")
    return out


def nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        out = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a non-negative integer") from exc
    if out < 0 or float(value) != float(out):
        raise ValueError(f"{name} must be a non-negative integer")
    return out


def window_params(window: Any, min_periods: Any | None, *, default_mp: int = 1) -> tuple[int, int]:
    w = positive_int(window, "window")
    mp = default_mp if min_periods is None else positive_int(min_periods, "min_periods")
    if mp > w:
        raise ValueError("min_periods must satisfy 1 <= min_periods <= window")
    return w, mp


def aligned_pd(*frames: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """Validate exact panel alignment instead of silently dropping/reindexing data."""
    if not frames:
        return ()
    base = frames[0]
    if not isinstance(base, pd.DataFrame):
        raise TypeError("aligned_pd requires pandas DataFrame inputs")
    if not base.index.is_unique or not base.columns.is_unique:
        raise ValueError("primary panel axes must be unique")
    for position, frame in enumerate(frames[1:], start=1):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"panel input {position} must be a pandas DataFrame")
        if not frame.index.is_unique or not frame.columns.is_unique:
            raise ValueError(f"panel input {position} axes must be unique")
        if not frame.index.equals(base.index):
            raise ValueError(f"panel input {position} index does not match the primary panel")
        if not frame.columns.equals(base.columns):
            raise ValueError(f"panel input {position} columns do not match the primary panel")
    return tuple(frames)


def finite_pd(x: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        np.isfinite(x.to_numpy(dtype=float, copy=False)),
        index=x.index,
        columns=x.columns,
    )


def frame_pd(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=template.index, columns=template.columns, dtype=float)


def pl_cols(frame: "pl.DataFrame") -> list[str]:
    return [c for c in frame.columns if c not in SKIP_COLUMNS]


def pl_base_with(frame: "pl.DataFrame", replacements: dict[str, "pl.Series"]) -> "pl.DataFrame":
    return frame.with_columns([series.alias(name) for name, series in replacements.items()])


def pl_finite(name: str) -> "pl.Expr":
    return pl.col(name).is_not_null() & pl.col(name).cast(pl.Float64, strict=False).is_finite()


def pl_unary_rolling_map(
    x: "pl.DataFrame",
    window: int,
    min_periods: int,
    func: Callable[[np.ndarray], float],
) -> "pl.DataFrame":
    replacements: dict[str, pl.Series] = {}
    for col in pl_cols(x):
        temp = pl.DataFrame({"v": x[col]})
        replacements[col] = temp.select(
            pl.col("v").rolling_map(
                func,
                window_size=window,
                min_samples=min_periods,
            ).alias("o")
        )["o"]
    return pl_base_with(x, replacements)


class PandasFunctionOperator(PandasOperator):
    # R5-02: direct ``calculate`` that routes through ``_prepare_call``.
    _HANDLES_CALL_CONTRACT = True

    def __init__(
        self,
        name: str,
        category: str,
        params: list[str],
        description: str,
        fn: Callable[..., pd.DataFrame],
        *,
        param_specs: dict[str, Any] | None = None,
    ):
        self._fn = fn
        # R13 NEW-P0-05: a backend implementation must NOT self-declare
        # ``pit_safe`` / ``audited`` — registration is not a semantic audit nor a
        # temporal certification.  PIT/certification is owned exclusively by the
        # canonical certification record (semantic_certification / evidence), so
        # those tags are stripped here.  The ``operator_overhaul_audited`` source
        # string remains (it is provenance, not a certification tag).
        self.metadata = PandasMetadata(
            name=name,
            category=category,
            description=description,
            param_names=params,
            return_type="series",
            param_specs={k: v for k, v in (param_specs or {}).items() if k in params},
            tags=["daily", "panel", category],
        )

    def calculate(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        processed_args, processed_kwargs = self._prepare_call(tuple(args), dict(kwargs))
        return self._fn(*processed_args, **processed_kwargs)


class PolarsFunctionOperator(PolarsOperator):
    # R5-02: direct ``calculate`` that routes through ``_prepare_call``.
    _HANDLES_CALL_CONTRACT = True

    def __init__(
        self,
        name: str,
        category: str,
        params: list[str],
        description: str,
        fn: Callable[..., "pl.DataFrame"],
        *,
        param_specs: dict[str, Any] | None = None,
    ):
        self._fn = fn
        # R13 NEW-P0-05: no self-declared ``pit_safe`` / ``audited`` tags — those
        # are owned by the canonical certification record, never by a backend
        # registration.
        self.metadata = PolarsMetadata(
            name=name,
            category=category,
            description=description,
            param_names=params,
            return_type="series",
            param_specs={k: v for k, v in (param_specs or {}).items() if k in params},
            tags=["daily", "panel", "polars_native", category],
        )

    def calculate(self, *args: Any, **kwargs: Any) -> "pl.DataFrame":
        if pl is None:
            raise ImportError("polars is required for the Polars backend")
        prepare = getattr(self, "_prepare_call", None)
        if prepare is not None:
            args, kwargs = prepare(tuple(args), dict(kwargs))
        return self._fn(*args, **kwargs)


@dataclass(frozen=True)
class Spec:
    category: str
    params: list[str]
    description: str
    pandas_fn: Callable[..., pd.DataFrame]
    polars_fn: Callable[..., "pl.DataFrame"] | None = None
    # Model-audit Phase 4 (search-space hygiene): explicit ParamSpec declarations
    # carried onto the operator metadata by register_specs (subset-filtered to the
    # declared ``params``).  ``None`` keeps the legacy empty contract.
    param_specs: dict[str, Any] | None = None


# R13 NEW-P0-04: logical-contract fields a replacement implementation may ONLY
# change via the canonical registration — never by rebuilding a reduced copy.
# A backend replacement may alter {implementation callable, backend capability,
# implementation hash} alone; every one of these dimensions must be inherited
# from (or verified equal to) the canonical contract.
_CONTRACT_FIELDS: tuple[tuple[str, Any], ...] = (
    ("param_specs", {}),
    ("param_aliases", {}),
    ("relational_specs", []),
    ("param_types", {}),
    ("input_units", {}),
    ("compatible_units", {}),
    ("output_unit", None),
    ("window_semantics", None),
    ("input_grain", None),
    ("output_grain", None),
    ("available_at", None),
    ("same_session_usable", None),
    ("role", None),
    ("input_arity", None),
    ("panel_arity", None),
    ("total_positional_arity", None),
    ("scalar_params", ()),
    ("panel_params", ()),
    ("input_fields", []),
    ("output_field", None),
    ("broadcast_specs", ()),
)


def _contract_field_equal(left: Any, right: Any) -> bool:
    """Structural equality for one logical-contract field (R13 NEW-P0-04).

    Mirrors the registry's ``_logical_field_equal``: containers compare
    element-wise (so a pandas ``param_specs`` dict and a parallel dict of the
    same specs compare equal despite different object identities); scalars and
    ``None`` compare directly.
    """
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left.keys()) != set(right.keys()):
            return False
        return all(_contract_field_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return False
        return all(_contract_field_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, float) and isinstance(right, float) and left != left and right != right:
        return True  # NaN == NaN for contract comparison
    return left == right


def _inherit_canonical_logical_contract(canonical: str, operator: Any) -> None:
    """R13 NEW-P0-04: replacement metadata inherits the canonical logical contract.

    Before an overhaul replacement is registered, resolve the canonical's
    logical contract (ParamSpec / aliases / units / compatible units / output
    unit / active_when-in-specs / grains / availability / role / arity /
    broadcast / semantic fields) from the existing catalog entry.  Every empty
    slot on the replacement is copied from the canonical; every non-empty slot
    must EQUAL the canonical or registration raises — a replacement may change
    the implementation, not the logical contract.

    The registry's ``_backfill_logical_contract`` already enforces this at
    ``register()`` time; this helper makes the overhaul layer honest up front and
    carries the canonical contract onto the operator *instance* metadata so the
    central validator sees the same contract on every backend.
    """
    meta = getattr(operator, "metadata", None)
    if meta is None:
        return
    catalog = OperatorRegistry._catalog.get(canonical) or {}
    for field, empty in _CONTRACT_FIELDS:
        canonical_value = catalog.get(field)
        if canonical_value in (None, "", (), [], {}):
            continue  # the canonical declares nothing here — nothing to inherit
        try:
            current = getattr(meta, field, None)
        except AttributeError:  # pragma: no cover - metadata is a dataclass
            continue
        if current in (None, "", (), [], {}):
            # Empty replacement slot -> inherit the canonical contract value so
            # the operator instance (not just the catalog dict) carries it.
            try:
                if isinstance(canonical_value, dict):
                    setattr(meta, field, dict(canonical_value))
                elif isinstance(canonical_value, (list, tuple)):
                    setattr(meta, field, type(canonical_value)(canonical_value))
                else:
                    setattr(meta, field, canonical_value)
            except (AttributeError, TypeError, ValueError):
                pass  # frozen metadata: the catalog still carries the contract
            continue
        # Non-empty replacement-declared value must MATCH the canonical.
        if not _contract_field_equal(current, canonical_value):
            raise ValueError(
                f"overhaul replacement of {canonical!r} diverges from the "
                f"canonical logical contract: field {field!r} declares "
                f"{current!r} but the canonical contract holds "
                f"{canonical_value!r} (R13 NEW-P0-04 — a replacement may change "
                "only the implementation callable / backend capability / "
                "implementation hash, never the logical contract)"
            )


def register_specs(specs: dict[str, Spec]) -> None:
    for name, spec in specs.items():
        prior_source = str(
            (OperatorRegistry._catalog.get(name, {}).get("backend_meta") or {})
            .get("pandas_numpy", {}).get("source", "")
            or ""
        )
        pandas_op = PandasFunctionOperator(
            name, spec.category, spec.params, spec.description, spec.pandas_fn,
            param_specs=spec.param_specs,
        )
        # R13 NEW-P0-04: a replacement may change only the implementation — the
        # canonical logical contract must be inherited / verified, never rebuilt.
        _inherit_canonical_logical_contract(name, pandas_op)
        OperatorRegistry.register(
            pandas_op,
            canonical=name,
            backend="pandas_numpy",
            source="operator_overhaul_audited",
            status="production",
            backend_explicit=True,
            # Round-7 P0 override-chain pinning: this bootstrap layer replaces an
            # earlier registration (e.g. ``gtja_compat``/``daily_panel``) — pin
            # the exact prior source so the chain is independent of import order.
            replace=True,
            replacement_reason="overhaul audit layer replaces prior bootstrap source",
            expected_old_source=prior_source,
        )
        if pl is not None and spec.polars_fn is not None:
            polars_op = PolarsFunctionOperator(
                name, spec.category, spec.params, spec.description, spec.polars_fn,
                param_specs=spec.param_specs,
            )
            _inherit_canonical_logical_contract(name, polars_op)
            OperatorRegistry.register(
                polars_op,
                canonical=name,
                backend="polars",
                source="operator_overhaul_native_polars",
                status="production",
                backend_explicit=True,
                replace=True,
                replacement_reason="overhaul audit layer replaces prior bootstrap source",
                expected_old_source=str(
                    (OperatorRegistry._catalog.get(name, {}).get("backend_meta") or {})
                    .get("polars", {}).get("source", "") or ""
                ),
            )
