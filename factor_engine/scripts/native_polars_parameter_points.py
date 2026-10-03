"""Fresh selected-Polars parameter evidence; never copy reference receipts.

This primitive runs a concrete panel against its Pandas authority. The caller
must supply the actual source context and persist a source-generation manifest;
it is not a production gate bypass or a broad parameter-region certificate.
"""
from __future__ import annotations
import hashlib
import json
import polars as pl
MAX_CERTIFICATION_CELLS = 200_000


def _certify_prepared_point(canonical, native_args, reference_args, kwargs, *,
                            context, store, evaluated, rtol, atol):
    from factor_engine.cleaned_operators.registry import OperatorRegistry as R
    from factor_engine.backend.cleaned_bridge import (
        _bound_scalar_parameters, _execution_variant, _input_dtype,
        _data_source_kind, _operator_semantic_version,
    )
    from factor_engine.backend.polars_backend_kind import get_physical_spec
    from factor_engine.backend.native_polars_evidence import _identity
    from factor_engine.runtime.parameter_domain_store import CertificationKey
    import numpy as np
    native = R.get(canonical, "polars", mode="any")
    reference = R.get(canonical, "pandas_numpy", mode="any")
    if native is None or reference is None:
        raise ValueError("selected native and reference owners are required")
    spec = get_physical_spec(native)
    if spec is None or spec.execution_kind.value != "polars_native_expr":
        raise ValueError("parameter evidence requires a selected native Expr owner")
    if not native_args or not all(isinstance(v, pl.DataFrame) for v in native_args):
        raise ValueError("concrete Polars panel inputs are required")
    if sum(v.height * v.width for v in native_args) > MAX_CERTIFICATION_CELLS:
        raise ValueError("parameter certification sample exceeds bounded cell limit")
    if not (0 <= rtol <= 1e-8 and 0 <= atol <= 1e-8):
        raise ValueError("parameter parity tolerances must remain finite and strict")
    if len(native_args) != len(reference_args):
        raise ValueError("native and reference input counts differ")
    for native_input, reference_input in zip(native_args, reference_args):
        if not hasattr(reference_input, "columns"):
            raise ValueError("concrete reference panels are required")
        columns = [str(c) for c in reference_input.columns]
        if any(c not in native_input.columns for c in columns):
            raise ValueError("native/reference input value columns differ")
        from factor_engine.backend.panel_polars import FE_TIME_COL
        if FE_TIME_COL in native_input.columns:
            import pandas as pd
            native_index = pd.Index(native_input[FE_TIME_COL].to_pandas())
            if not native_index.equals(reference_input.index):
                raise ValueError("native/reference input coordinates differ")
        if not np.allclose(native_input.select(columns).to_numpy(),
                           reference_input.to_numpy(dtype=float),
                           rtol=0, atol=0, equal_nan=True):
            raise ValueError("native/reference input values differ")
    point = _bound_scalar_parameters(native, list(native_args), dict(kwargs))
    if not point.complete:
        raise ValueError(f"incomplete parameter point: {point.missing}")
    before = _identity(canonical)
    scalar_point = {name: value for name, value in point.normalized.items()
                    if not isinstance(value, (pl.DataFrame, pl.Series))}

    variant = _execution_variant(native, "polars", canonical).to_key()
    dtype = _input_dtype(list(evaluated)).to_key()
    expected = reference.calculate(*reference_args, **kwargs)
    actual = native.calculate(*native_args, **kwargs)
    if not isinstance(actual, pl.DataFrame) or not hasattr(expected, "columns"):
        raise ValueError("panel result required for parameter parity evidence")
    columns = [str(c) for c in expected.columns]
    if any(c not in actual.columns for c in columns) or actual.height != len(expected):
        raise ValueError("native/reference output axes differ")
    for column in actual.columns:
        if column not in columns:
            if column not in native_args[0].columns or not actual[column].equals(native_args[0][column]):
                raise ValueError("native output changed a coordinate")
    left = actual.select(columns).to_numpy()
    right = expected.to_numpy(dtype=float)
    passed = bool(np.allclose(left, right, rtol=rtol, atol=atol, equal_nan=True))
    if before != _identity(canonical):
        raise RuntimeError("selected implementation changed during parameter test")
    source = _data_source_kind(context)
    key = CertificationKey.from_kwargs(
        canonical, scalar_point, backend="polars",
        semantic_version=_operator_semantic_version(canonical),
        execution_variant=variant, source_context=source, dtype=dtype,
        grain=str(getattr(context, "grain", "daily") or "daily"),
    )
    evidence = hashlib.sha256(json.dumps({
        "identity": before, "key": key.to_key(), "passed": passed,
        "native_output": actual.hash_rows(seed=0).to_list(),
        "input_hashes": [v.hash_rows(seed=0).to_list() for v in native_args],
        "reference_output": hashlib.sha256(right.tobytes()).hexdigest(),
    }, sort_keys=True, default=str).encode()).hexdigest()
    store.certify_point(key, passed, evidence_hash=evidence,
                        source="fresh_selected_polars_authority_panel_parity",
                        details={"rows": actual.height, "rtol": rtol, "atol": atol,
                                 "native_identity": before})
    if not passed:
        raise AssertionError(f"native parameter parity failed for {canonical}")
    return key


def certify_selected_point(canonical, native_args, reference_args, kwargs, *,
                           context, store, rtol=1e-10, atol=1e-12):
    return _certify_prepared_point(
        canonical, native_args, reference_args, kwargs, context=context,
        store=store, evaluated=native_args, rtol=rtol, atol=atol)


def certify_runtime_point(canonical, evaluated, kwargs, *, context, store,
                          rtol=1e-10, atol=1e-12):
    """Use the bridge's actual preparation and pre-preparation dtype identity.

    Reference conversion is validation-only, not a production kernel path.
    This bounded helper supports panel-preserving calls with scalar kwargs,
    not long-table or grain-transform operators.
    """
    import pandas as pd
    from factor_engine.backend.cleaned_bridge import _prepare_call_args
    from factor_engine.backend.panel_polars import FE_TIME_COL
    if not evaluated or not all(isinstance(v, (pd.DataFrame, pd.Series, pl.DataFrame))
                                for v in evaluated):
        raise ValueError("runtime certification requires concrete panel inputs")
    cells = sum(v.height * v.width if isinstance(v, pl.DataFrame)
                else v.size for v in evaluated)
    if cells > MAX_CERTIFICATION_CELLS:
        raise ValueError("parameter certification sample exceeds bounded cell limit")
    prepared, _, _ = _prepare_call_args(list(evaluated), context, backend="polars")
    reference_args = []
    for frame in prepared:
        if not isinstance(frame, pl.DataFrame):
            raise ValueError("runtime preparation did not produce a Polars panel")
        panel = frame.to_pandas()
        if FE_TIME_COL in panel.columns:
            panel = panel.set_index(FE_TIME_COL)
            panel.index.name = context.timestamp_col
        reference_args.append(panel)
    return _certify_prepared_point(
        canonical, prepared, reference_args, kwargs, context=context, store=store,
        evaluated=evaluated, rtol=rtol, atol=atol)
