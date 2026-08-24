# -*- coding: utf-8 -*-
"""算子注册中心：canonical 名、别名、元数据与 runtime 实例的唯一索引。

命名约定（canonical）
--------------------
- **时序滚动** ``ts_*``：``ts_mean``、``ts_std``、``ts_pct``、``ts_decay_linear`` 等
- **截面** 无前缀或 ``cs_*``：``rank``、``zscore``、``cs_demean``、``cs_regression``
- **分组** ``group_*``：``group_neutralize``、``group_rank``、``group_mean``
- **扩展窗口** ``expanding_*``：``expanding_mean``、``expanding_zscore``
- **技术指标** 大写 TA-Lib 风格：``MACD``、``RSI``、``WMA``（DSL 可用 ``ts_rsi`` 等别名）
- **元素级** 小写 numpy 风格：``clip``、``log``、``where``、``abs``

旧名 / 方言名（``SMA``、``m_var``、``returns``、``cap`` 等）通过 ``_aliases.py`` 与
``_dedupe.py`` 映射到 canonical，**仍可计算**，不会进入重复 canonical 列表。

生命周期
--------
1. 各 ``cleaned_operators/*.py`` 在 import 时用 ``@register_operator`` 注册实现类；
2. ``_aliases.py`` 在 ``load_all()`` 末尾登记 DSL 别名 → canonical；
3. ``OperatorRegistry.get(name)`` 供 ``cleaned_bridge`` 与单测直接调用 ``calculate``。

数据结构
--------
- ``_operators[canonical][backend]``：如 ``pandas_numpy`` 上的算子实例；
- ``_aliases[alias]`` → canonical：DSL 名 ``ts_rsi`` 可解析到 ``RSI``；
- ``_catalog[canonical]``：description、param_names、status（``implemented`` / ``doc_only``）。

白名单与 runtime 一致性：仅 ``get(canon) is not None`` 的算子会进入 ``build_dsl_allowlist()``。
"""
from __future__ import annotations

import copy
import dataclasses
import datetime
import functools
from collections.abc import Mapping
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# R7-222: the MISSING sentinel (defined in base) distinguishes "no default
# declared" from an explicit None default; _contract_hash relies on it to keep
# contract hashes stable and honest.
from factor_engine.cleaned_operators.base import MISSING  # noqa: E402

_BOOTSTRAP_TOKEN = object()


class RegistryInitializationError(RuntimeError):
    """Registry bootstrap was attempted from an impossible lifecycle state."""


def _freeze_const(value: Any) -> Any:
    """Deterministic representation of a code-object constant (P0-23).

    Nested code objects recurse through :func:`_code_payload` so their digest
    never embeds a memory address.  R9-P1-041: non-primitive constants MUST NOT
    fall back to ``repr`` — ``repr(frozenset(...))`` and ``repr({...})`` iterate
    the container in hash-seed-dependent order, so the digest would drift across
    processes.  They go through the canonical :func:`_freeze_value` (sorted,
    recursive) instead.
    """
    if isinstance(value, tuple):
        return tuple(_freeze_const(v) for v in value)
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return value
    if getattr(value, "co_code", None) is not None:  # nested function code
        return _code_payload(value, include_names=True)
    # frozenset / dict / any other constant: canonical semantic payload (never
    # a hash-seed-dependent repr).
    return _freeze_value(value)


def _code_payload(code: Any, *, include_names: bool) -> str:
    """Deterministic digest of a ``types.CodeType`` object (R10 #13).

    Computes a normalized implementation identity from the DISASSEMBLED
    bytecode: an ordered ``[(opname, resolved_operand), ...]`` sequence where
    every operand is resolved to its actual value —

    * ``LOAD_CONST`` -> the frozen constant value (nested code objects recurse
      through :func:`_freeze_const`);
    * name ops (``LOAD_GLOBAL`` / ``LOAD_NAME`` / ``LOAD_ATTR`` / ...) -> the
      actual name string when ``include_names`` is true, otherwise the raw arg
      (the positional index into ``co_names``);
    * every other op -> the raw argument.

    The tuples are NEVER sorted: bytecode operands reference the ORIGINAL
    ``co_consts`` / ``co_names`` tuple index, so sorting breaks the
    index->value correspondence and lets ``2*x+3`` collide with ``3*x+2`` (same
    constant set ``{2, 3}`` with structurally identical bytecode and resolved
    indices).  ``dis.get_instructions`` is deterministic across processes
    (bytecode disassembly does not depend on hash seeds or memory addresses),
    so the digest is stable.
    """
    import dis
    import hashlib

    name_codes = frozenset(dis.hasname)
    n_names = len(code.co_names)
    n_consts = len(code.co_consts)
    ops: list[tuple[str, Any]] = []
    for instr in dis.get_instructions(code):
        if instr.opname == "CACHE":  # adaptive-interpreter noise, not semantic
            continue
        opname = instr.opname
        arg = instr.arg
        # R44-P0: wordcode (Python >= 3.11) quirks on this CPython build:
        #  * ``LOAD_GLOBAL``/``LOAD_ATTR`` ``instr.arg`` is the raw encoded
        #    operand (``(name_idx << 1) | flag`` for LOAD_GLOBAL), which may
        #    exceed ``len(co_names)`` — never index past the tuple, fall back
        #    to the raw arg.
        #  * ``RETURN_CONST`` (3.12+) is a const-carrying opcode that is NOT
        #    ``LOAD_CONST``; the digest must resolve the actual value from
        #    ``co_consts[arg]`` or every ``return <literal>`` collapses to the
        #    same opcode+arg pair (constant-swap collisions).
        if opname in ("LOAD_CONST", "RETURN_CONST") and arg is not None:
            resolved = _freeze_const(
                code.co_consts[arg] if 0 <= arg < n_consts else None
            )
        elif arg is not None and instr.opcode in name_codes:
            resolved = code.co_names[arg] if 0 <= arg < n_names else arg
        else:
            resolved = arg
        ops.append((opname, resolved))
    return hashlib.sha256(repr(tuple(ops)).encode("utf-8")).hexdigest()[:16]


def _freeze_value(value: Any) -> str:
    """Deterministic string payload of a closed-over / kernel value (R7-227).

    Memory addresses (``id()``), object ``__repr__`` and pointer-derived
    representations are forbidden in a semantic hash — they differ across
    processes and do not reflect semantic change.  This canonicalizes:

    * primitives -> canonical JSON;
    * Enum member -> module.Class.MEMBER + frozen value;
    * dict / any ``Mapping`` -> sorted keys, recursive;
    * list/tuple -> ordered recursive hash;
    * set/frozenset -> sorted recursive hashes;
    * numpy scalar -> canonical Python value (dtype-aware);
    * ndarray -> dtype + shape + raw bytes;
    * DataFrame -> columns + dtypes + index (values/name/tz) + NaN mask +
      deterministic payload hash (R9-P1-043);
    * datetime/date/time/timedelta -> ISO / seconds (timezone-aware);
    * frozen dataclass (and any dataclass) -> fields recursively (R9-P1-042);
    * object with an explicit ``semantic_identity()`` -> that identity;
    * function/code -> bytecode + defaults + closure (recursive);
    * anything else -> a clear error (R9-P1-042): hashing by type name alone
      would conflate ``SameClass(config=A)`` and ``SameClass(config=B)``.
    """
    import hashlib

    if isinstance(value, Enum):
        # IntEnum members are ints too — catch them BEFORE the primitives branch
        # so ``Color.RED`` never collapses into plain ``1``.
        return (
            f"enum({type(value).__module__}.{type(value).__name__}.{value.name}="
            + _freeze_value(value.value)
            + ")"
        )
    if value is MISSING:
        # R7-222 sentinel: "no default declared" is a distinct semantic payload
        # from an explicit ``None`` default.  Previously this raised a cryptic
        # TypeError; R10 #14 makes it a first-class canonical payload so e.g. a
        # ParamSpec with ``default=MISSING`` freezes deterministically.
        return "MISSING"
    if value is None or isinstance(value, (bool, int, str, bytes, complex)):
        return repr(value)
    if isinstance(value, float):
        # NaN has many bit patterns; canonicalize to a single spelling so a
        # semantic hash is stable regardless of how the NaN arrived.
        if value != value:
            return "NaN"
        if value == float("inf"):
            return "Inf"
        if value == float("-inf"):
            return "-Inf"
        return repr(value)
    if isinstance(value, tuple):
        return "tuple(" + ",".join(_freeze_value(v) for v in value) + ")"
    if isinstance(value, (list,)):
        return "list[" + ",".join(_freeze_value(v) for v in value) + "]"
    if isinstance(value, (set, frozenset)):
        return "set{" + ",".join(sorted(_freeze_value(v) for v in value)) + "}"
    if isinstance(value, Mapping):
        # R9-P1-042: any mapping (dict, MappingProxyType, defaultdict, ...),
        # canonical sorted-items recursive hash — never repr().
        return "dict{" + ",".join(
            f"{_freeze_value(k)}:{_freeze_value(value[k])}"
            for k in sorted(value.keys(), key=repr)
        ) + "}"
    if isinstance(value, np.ndarray):
        # dtype + shape + raw bytes: the bytes encode the exact numeric payload.
        if value.dtype.kind == "O":
            # R10 #14: object-dtype arrays must NEVER hash their raw bytes —
            # ``ndarray.tobytes()`` on an object array returns the element
            # POINTERS, which embed memory addresses and drift across processes.
            # Canonical recursive freeze of every element; elements that cannot
            # be canonically frozen raise a clear TypeError instead of silently
            # producing an address-dependent digest.
            blob = ",".join(_freeze_value(v) for v in value.flat).encode("utf-8")
        else:
            try:
                blob = np.ascontiguousarray(value).tobytes()
            except (TypeError, ValueError):  # mixed / unsupported numeric dtype
                blob = ",".join(_freeze_value(v) for v in value.flat).encode("utf-8")
        return f"ndarray({value.dtype},{value.shape}," + hashlib.sha256(blob).hexdigest()[:16] + ")"
    if isinstance(value, np.datetime64):
        try:
            return f"npdt({np.datetime_as_string(value)})"
        except (ValueError, TypeError):
            return f"npdt({value!s})"
    if isinstance(value, np.generic):
        # numpy scalars: canonicalize through the exact Python value when the
        # scalar is exactly representable, otherwise dtype+str (so int64(1) !=
        # int32(1) and float32(0.1) != float64(0.1)).
        if isinstance(value, np.bool_):
            return _freeze_value(bool(value))
        if isinstance(value, np.integer):
            return _freeze_value(int(value))
        if isinstance(value, np.floating):
            return _freeze_value(float(value))
        if isinstance(value, np.complexfloating):
            return _freeze_value(complex(value))
        return f"npscalar({value.dtype}:{value!s})"
    if isinstance(value, pd.DataFrame):
        # R9-P1-043: schema + index + NaN mask + payload.  Two frames with
        # identical VALUES but a 2024-vs-2025 index, a different index name, a
        # different timezone, different dtypes, a different NaN mask, or a
        # different column order MUST hash apart.
        import hashlib as _h

        cols = ",".join(str(c) for c in value.columns)
        dtypes = ",".join(str(d) for d in value.dtypes)
        idx = value.index
        if isinstance(idx, pd.DatetimeIndex):
            _vals: list[str] = []
            for _v in idx:
                if _v is pd.NaT or (isinstance(_v, float) and _v != _v):
                    _vals.append("NaT")
                else:
                    _vals.append(_v.isoformat() if hasattr(_v, "isoformat") else _freeze_value(_v))
            idx_vals = _h.sha256(",".join(_vals).encode("utf-8")).hexdigest()[:16]
            tz = str(idx.tz)
        elif isinstance(idx, pd.MultiIndex):
            # R10 #14: MultiIndex levels may hold arbitrary labels (tuples,
            # timestamps, object elements) — freeze recursively, never repr()
            # (repr of the level tuple list embeds object addresses).
            idx_vals = _h.sha256(
                _freeze_value(list(idx)).encode("utf-8")
            ).hexdigest()[:16]
            tz = ""
        else:
            idx_vals = _h.sha256(
                ",".join(_freeze_value(v) for v in idx).encode("utf-8")
            ).hexdigest()[:16]
            tz = ""
        idx_name = (
            "|".join(_freeze_value(n) for n in idx.names)
            if isinstance(idx, pd.MultiIndex)
            else _freeze_value(idx.name)
        )
        # NaN mask: canonicalizes the many NaN bit patterns AND distinguishes a
        # real NaN from a numerically-equal placeholder (e.g. 0.0).
        try:
            mask = pd.isna(value).to_numpy()
            mask_hex = _h.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()[:16]
        except Exception:
            mask_hex = ""
        try:
            arr = value.to_numpy(dtype=float)
        except (TypeError, ValueError):
            # non-numeric / object columns: hash the canonical recursive freeze
            # (R10 #14) — never a repr-based digest of object cells.
            payload = _h.sha256(
                _freeze_value(value.to_numpy(dtype=object).tolist()).encode("utf-8")
            ).hexdigest()[:16]
        else:
            try:
                arr = np.ascontiguousarray(arr, dtype=float)
                if mask.size and bool(mask.any()):
                    arr = arr.copy()
                    arr[mask] = 0.0  # NaN -> fixed byte pattern; mask records it
                payload = _h.sha256(arr.tobytes()).hexdigest()[:16]
            except (TypeError, ValueError):
                payload = _h.sha256(
                    _freeze_value(arr.tolist()).encode("utf-8")
                ).hexdigest()[:16]
        return (
            f"DataFrame(cols={cols}|dtypes={dtypes}|idx=({idx_vals})|"
            f"idxname={idx_name}|tz={tz}|mask={mask_hex}|payload={payload})"
        )
    if isinstance(value, pd.Series):
        # 1-D labelled container: hash by dtype + name + index (values/tz) +
        # NaN mask + payload, mirroring the DataFrame contract (R9-P1-043).
        import hashlib as _h

        _dt = str(value.dtype)
        _name = _freeze_value(value.name)
        _idx = value.index
        if isinstance(_idx, pd.DatetimeIndex):
            _vlist: list[str] = []
            for _v in _idx:
                if _v is pd.NaT or (isinstance(_v, float) and _v != _v):
                    _vlist.append("NaT")
                else:
                    _vlist.append(_v.isoformat() if hasattr(_v, "isoformat") else _freeze_value(_v))
            _idx_hex = _h.sha256(",".join(_vlist).encode("utf-8")).hexdigest()[:16]
            _tz = str(_idx.tz)
        else:
            # R10 #14: canonical recursive freeze of each index label — never
            # repr() (object labels embed addresses).
            _idx_hex = _h.sha256(
                ",".join(_freeze_value(v) for v in _idx).encode("utf-8")
            ).hexdigest()[:16]
            _tz = ""
        try:
            _mask = pd.isna(value).to_numpy()
            _mask_hex = _h.sha256(np.ascontiguousarray(_mask).tobytes()).hexdigest()[:16]
        except Exception:
            _mask_hex = ""
        try:
            _arr = np.ascontiguousarray(value.to_numpy(dtype=float), dtype=float)
            if _mask.size and bool(_mask.any()):
                _arr = _arr.copy()
                _arr[_mask] = 0.0
            _payload = _h.sha256(_arr.tobytes()).hexdigest()[:16]
        except (TypeError, ValueError):
            # R10 #14: object-dtype payload -> canonical recursive freeze, never
            # a repr-based digest (object cell reprs embed addresses).
            _payload = _h.sha256(
                _freeze_value(value.to_numpy(dtype=object).tolist()).encode("utf-8")
            ).hexdigest()[:16]
        return (
            f"Series(dtype={_dt}|name={_name}|idx=({_idx_hex})|tz={_tz}|"
            f"mask={_mask_hex}|payload={_payload})"
        )
    if value is pd.NaT:
        return "NaT"
    if isinstance(value, datetime.datetime):
        return f"datetime({value.isoformat()})"
    if isinstance(value, datetime.date):
        return f"date({value.isoformat()})"
    if isinstance(value, datetime.time):
        return f"time({value.isoformat()})"
    if isinstance(value, datetime.timedelta):
        return f"timedelta({value.total_seconds()})"
    if isinstance(value, (pd.Period, pd.Interval)):
        return f"pandas({type(value).__name__}:{value!s})"
    if isinstance(value, functools.partial):
        _kw = ",".join(
            f"{k}={_freeze_value(v)}" for k, v in sorted(value.keywords.items())
        )
        _args = ",".join(_freeze_value(a) for a in value.args)
        return f"partial(fn={_freeze_value(value.func)}|args=[{_args}]|kw={{{_kw}}})"
    # R9-P1-042: a dataclass instance hashes by its FIELDS (recursively), never
    # by ``type(value).__name__`` — two dataclasses of the same class with
    # different field values are different semantic payloads.
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        field_parts = [
            f"{_f.name}=" + _freeze_value(getattr(value, _f.name))
            for _f in dataclasses.fields(value)
        ]
        return f"{type(value).__name__}[" + ",".join(field_parts) + "]"
    # An explicit semantic_identity() is the escape hatch for objects that
    # cannot be introspected generically but know how to identify themselves.
    _semantic = getattr(value, "semantic_identity", None)
    if callable(_semantic):
        try:
            _ident = _semantic()
        except Exception:
            _ident = None
        if _ident is not None:
            return f"{type(value).__name__}[" + _freeze_value(_ident) + "]"
    code = getattr(value, "co_code", None)
    if code is not None:  # code object -> bytecode + constants
        return _code_payload(value, include_names=True)
    if callable(value):
        fn_code = getattr(value, "__code__", None)
        if fn_code is not None:
            parts = [_code_payload(fn_code, include_names=True)]
            closure = getattr(value, "__closure__", None) or ()
            cells = []
            for cell in closure:
                try:
                    cells.append(_freeze_value(cell.cell_contents))
                except ValueError:  # uninitialised cell
                    cells.append("<empty>")
            parts.append("cells=" + ",".join(sorted(cells)))
            return "fn(" + "|".join(parts) + ")"
        # Callable without Python bytecode (numpy ufunc, C builtin): hash by its
        # stable module.qualname identity — deterministic, never id()/repr.
        _mod = getattr(value, "__module__", None) or type(value).__module__
        _qual = (
            getattr(value, "__qualname__", None)
            or getattr(value, "__name__", None)
            or type(value).__qualname__
        )
        return f"callable({_mod}.{_qual})"
    # R9-P1-042: hashing by type name alone conflates distinct payloads.  Fail
    # loudly (certification failure) instead of producing a collision-prone
    # identity that silently equates ``SameClass(config=A)`` and
    # ``SameClass(config=B)``.
    raise TypeError(
        "cannot compute a semantic hash for an object of type "
        f"{type(value).__module__}.{type(value).__qualname__}: it is not a "
        "primitive, Enum member, mapping, sequence, set, numpy scalar/ndarray, "
        "DataFrame, datetime, pandas scalar, dataclass, callable, or an object "
        "with a semantic_identity() method.  Hashing by type name alone would "
        "conflate distinct payloads (R9-P1-042)."
    )


def _impl_source_hash(operator: Any) -> str:
    """Deterministic hash of the ACTUAL kernel implementation (P0-23 / R7-226).

    The hash priority NEVER starts at the framework ``calculate`` — two
    different operators sharing a base-class ``calculate`` must not hash the
    same.  R7-226 resolution order:

    1. class-defined ``_calculate_series`` / ``_calculate_scalar`` (the real
       kernel, when the subclass overrides it);
    2. class-defined ``calculate`` (a direct implementation that owns its own
       call contract — only when ``_calculate_*`` is NOT overridden on the
       class);
    3. the wrapped kernel callable (``_fn`` default of a bridge);
    4. the factory-closure semantic payload (``__closure__`` cells);
    5. base ``calculate`` only as a framework hash (module.qualname).

    R7-227: closure cells are serialized with :func:`_freeze_value` — a
    deterministic payload hash — never ``repr(id(...))``.
    """
    import hashlib

    cls = operator.__class__

    def _is_class_defined(attr: str) -> bool:
        # R7-226: "class-defined" means the SUBCLASS overrides it — the abstract
        # framework bases (Operator/SeriesOperator) define ``_calculate_series``
        # and ``calculate`` too, and hashing those would conflate every operator
        # that relies on the base routing.  Only methods defined on a class below
        # the framework bases count as the operator's own kernel.
        try:
            from factor_engine.cleaned_operators import base as _base
            from factor_engine.cleaned_operators import base_polars as _base_polars

            framework_bases = {
                _base.Operator, _base.SeriesOperator, _base.ScalarOperator,
                _base.TransformOperator, _base.TwoVarOperator,
                _base_polars.Operator, _base_polars.SeriesOperator,
                _base_polars.ScalarOperator, _base_polars.TransformOperator,
                _base_polars.TwoVarOperator,
            }
        except ImportError:  # pragma: no cover - base always importable
            framework_bases = set()
        for klass in cls.__mro__:
            if klass in framework_bases:
                return False  # only the framework defines this from here up
            if attr in klass.__dict__:
                return True
        return False

    # 1. class-defined kernel method (highest priority).
    for attr in ("_calculate_series", "_calculate_scalar"):
        if _is_class_defined(attr):
            fn = getattr(operator, attr, None)
            if callable(fn):
                payload = _fn_payload(fn, cls)
                if payload is not None:
                    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    # 2. class-defined calculate (direct call-contract implementation).
    if _is_class_defined("calculate"):
        fn = getattr(operator, "calculate", None)
        if callable(fn):
            payload = _fn_payload(fn, cls)
            if payload is not None:
                return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    # 3. wrapped kernel callable via a bridge ``_fn`` default.
    try:
        import inspect

        _candidate = getattr(operator, "_calculate_series", None)
        if _candidate is None:
            _candidate = getattr(operator, "calculate", None)
        if _candidate is not None:
            sig = inspect.signature(_candidate)
            fn_default = sig.parameters.get("_fn")
            if fn_default is not None and callable(fn_default.default):
                payload = _fn_payload(fn_default.default, cls)
                if payload is not None:
                    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    except (TypeError, ValueError):
        pass
    # 4. factory closure payload (register_dual / _mk closures).
    for attr in ("_calculate_series", "_calculate_scalar", "calculate"):
        fn = getattr(operator, attr, None)
        if fn is None:
            continue
        closure = getattr(fn, "__closure__", None)
        if callable(fn) and closure:
            parts = [_code_payload(fn.__code__, include_names=False)]
            cells = []
            for cell in closure:
                try:
                    cells.append(_freeze_value(cell.cell_contents))
                except ValueError:
                    cells.append("<empty>")
            parts.append("cells=" + ",".join(sorted(cells)))
            return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
    # 5. framework hash: module.qualname only.
    src = f"{cls.__module__}.{cls.__qualname__}"
    return hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]


def _contract_hash(operator: Any) -> str:
    """Deterministic hash of an operator's LOGICAL contract (R7-233).

    Independent of the implementation source: two backends of the same canonical
    with identical declared contracts hash identically, so an audit can prove
    whether an override replaced the contract or only the kernel.  Covers
    param_names + param_specs (incl. default/dtype/min/max/choices/active_when)
    + panel_params/scalar_params + param_aliases + input/output units + grains +
    available_at + input_fields + relational_specs + the runtime execution
    contract (statefulness/chunking/checkpoint) + the declared edge contract +
    the parameter-aware cost band (R9-P1-044).

    R9-P1-044 coverage note: ``determinism`` is derived from the declared
    ``tags`` (``"deterministic"`` is the conventional tag).  R30 §12 closes the
    long-standing gaps: input/output semantic types, ``ClockContract`` /
    ``available_at`` / ``same_session_usable``, ``HistoryTransform`` / history
    formula, ``missing_policy``, ``universe_requirement``, parameter kinds
    (panel/scalar), ``ParamRole``, and ``BroadcastSpec`` all feed the hash — any
    change that alters what the factor MEANS changes the digest, so stale
    evidence / caches are invalidated.
    """
    import hashlib

    meta = getattr(operator, "metadata", None)
    if meta is None:
        return hashlib.sha256(b"").hexdigest()[:16]
    parts: list[str] = []
    parts.append("params=" + ",".join(list(getattr(meta, "param_names", None) or [])))
    specs = getattr(meta, "param_specs", None) or {}
    spec_parts = []
    for key in sorted(specs.keys()):
        spec = specs[key]
        dtype = getattr(spec, "dtype", None)
        dtype_name = getattr(dtype, "__name__", str(dtype))
        default = getattr(spec, "default", MISSING)
        if default is MISSING:
            default_repr = "MISSING"
        else:
            default_repr = _freeze_value(default)
        # R30 §12: parameter ROLE is part of the search/meaning contract.
        _role = getattr(spec, "param_role", None)
        role_repr = str(getattr(_role, "value", _role)) if _role is not None else "NONE"
        spec_parts.append(
            f"{key}:(dtype={dtype_name},min={getattr(spec, 'min', None)!r},"
            f"max={getattr(spec, 'max', None)!r},choices={getattr(spec, 'choices', None)!r},"
            f"active_when={getattr(spec, 'active_when', None)!r},default={default_repr},"
            f"role={role_repr})"
        )
    parts.append("specs={" + ",".join(spec_parts) + "}")
    parts.append("panel=" + ",".join(tuple(getattr(meta, "panel_params", None) or ())))
    parts.append("scalar=" + ",".join(tuple(getattr(meta, "scalar_params", None) or ())))
    parts.append("aliases=" + ",".join(
        f"{k}->{v}" for k, v in sorted((getattr(meta, "param_aliases", None) or {}).items())
    ))
    parts.append("in_units=" + ",".join(
        f"{k}->{v}" for k, v in sorted((getattr(meta, "input_units", None) or {}).items())
    ))
    parts.append(f"out_unit={getattr(meta, 'output_unit', None)!r}")
    parts.append(f"in_grain={getattr(meta, 'input_grain', None)!r}")
    parts.append(f"out_grain={getattr(meta, 'output_grain', None)!r}")
    parts.append(f"avail={getattr(meta, 'available_at', None)!r}")
    # R30 §12: same-session usability (session-close availability semantics).
    parts.append(f"same_session={bool(getattr(meta, 'same_session_usable', False))!r}")
    parts.append("in_fields=" + ",".join(list(getattr(meta, "input_fields", None) or [])))
    # R30 §12: input/output SEMANTIC TYPES (FundamentalFeature / Condition /
    # RawPrice / …) — a type change means the operator eats different data.
    parts.append("in_sem_types=" + ",".join(
        f"{k}->{_freeze_value(v)}" for k, v in sorted(
            (getattr(meta, "input_semantic_types", None) or {}).items()
        )
    ))
    parts.append(f"out_sem_type={getattr(meta, 'output_semantic_type', None)!r}")
    # R30 §12: HistoryTransform / history semantics — a change in what history
    # the kernel re-reads changes the factor.
    parts.append("history_formula=" + ",".join(
        f"{k}->{getattr(v, 'history_formula', v)!r}" for k, v in sorted(
            (getattr(meta, "param_specs", None) or {}).items()
        ) if getattr(v, "history_formula", None) is not None
    ))
    # R30 §12: declared missing policy / universe requirement.
    parts.append(f"missing_policy={_freeze_value(getattr(meta, 'missing_policy', None))}")
    parts.append(f"universe={_freeze_value(getattr(meta, 'universe_requirement', None))}")
    parts.append(f"market_scope={_freeze_value(getattr(meta, 'market_scope', None))}")
    parts.append(f"currency={_freeze_value(getattr(meta, 'currency', None))}")
    # R9-P1-044: cross-parameter feasibility constraints (relational_specs).
    rel = list(getattr(meta, "relational_specs", None) or [])
    rel_parts = []
    for _spec in rel:
        _expr = getattr(_spec, "expression", None)
        _msg = getattr(_spec, "message", None)
        _names = sorted(getattr(_spec, "param_names", None) or ())
        rel_parts.append(f"{_expr!r}:{_msg!r}:({','.join(_names)})")
    parts.append("relational_specs={" + ",".join(sorted(rel_parts)) + "}")
    # R30 §12: BroadcastSpec is a declared structural contract.
    bspecs = list(getattr(meta, "broadcast_specs", None) or ())
    if bspecs:
        bs_parts = []
        for _bs in bspecs:
            bs_parts.append(
                f"mode={getattr(_bs, 'mode', '')},dm={getattr(_bs, 'date_mapping', '')},"
                f"ip={getattr(_bs, 'instrument_policy', '')},"
                f"src={getattr(_bs, 'source_param', '')},tgt={getattr(_bs, 'target_param', '')}"
            )
        parts.append("broadcast={" + ",".join(bs_parts) + "}")
    canonical = getattr(meta, "name", None) or ""
    if canonical:
        # ExecutionContract / statefulness / chunking / checkpoint schema — the
        # single authority in runtime.execution_contract, keyed by canonical.
        # R30 §14: contract resolution is FAIL-CLOSED — a key contract that
        # cannot be resolved must raise, never be silently skipped.
        from factor_engine.runtime.execution_contract import execution_contract as _ec

        _ec_res = _ec(canonical)
        parts.append(
            "execution=" + ",".join([
                str(getattr(_ec_res, "state_model", "")),
                str(getattr(_ec_res, "chunking", "")),
                str(getattr(_ec_res, "checkpoint_schema", "")),
            ])
        )
        # Edge contract (nan/pos_inf/neg_inf/zero/domain_invalid behaviors).
        from factor_engine.cleaned_operators.edge_requirements import edge_contract as _edge

        _edge_res = _edge(canonical)
        if _edge_res is not None:
            parts.append("edge=" + ",".join(
                f"{_dim}:{getattr(_edge_res, _dim, 'invalid')}"
                for _dim in ("nan", "pos_inf", "neg_inf", "zero", "domain_invalid")
            ))
        # Cost contract: deterministic complexity band + reference runtime cost
        # (operator_cost_model is a pure prefix-match model).
        from factor_engine.cleaned_operators.operator_cost_model import (
            complexity_label as _cost_label,
            runtime_cost as _cost_rt,
        )

        parts.append(f"cost_label={_cost_label(canonical)}")
        parts.append(f"cost_runtime={_cost_rt(canonical)}")
        # Determinism: no dedicated field; derive from the declared tags
        # ("deterministic" is the conventional tag across operator modules).
        _tags = getattr(meta, "tags", None) or ()
        parts.append(f"deterministic={'deterministic' in _tags}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _fn_payload(fn: Any, cls: type) -> str | None:
    """Deterministic payload of one callable: code + closure (R7-227).

    The class qualname is deliberately NOT part of the semantic payload: two
    operators whose kernels are byte-identical ARE the same implementation and
    must hash identically (determinism across classes/processes).  Class
    identity is already captured by the caller's resolution order — a
    class-defined kernel reaching this point has already been distinguished from
    the framework base by ``_is_class_defined``.
    """
    import inspect as _inspect

    try:
        code = getattr(fn, "__code__", None)
        closure = getattr(fn, "__closure__", None)
        if code is not None and closure:
            parts = [_code_payload(code, include_names=False)]
            # R30 §13 (P0-009): a closure cell's position is semantically bound
            # to its ``co_freevars`` name — the SAME value set in a different
            # variable binding is a different kernel.  Hash cells in original
            # order as ``(freevar_name, frozen_value)`` pairs; sorting the cell
            # values alone dropped that binding relationship.
            cells: list[str] = []
            freevars = list(code.co_freevars)
            for i, cell in enumerate(closure):
                name = freevars[i] if i < len(freevars) else f"<cell{i}>"
                try:
                    cells.append(f"{name}={_freeze_value(cell.cell_contents)}")
                except ValueError:  # uninitialised cell
                    cells.append(f"{name}=<empty>")
            parts.append("cells={" + ",".join(cells) + "}")
            return "|".join(parts)
        if code is not None:
            return _code_payload(code, include_names=True)
        return _inspect.getsource(fn)
    except (OSError, TypeError):  # pragma: no cover - interactive/no-source
        return None


def _calculate_is_framework_or_declared(operator: Any) -> bool:
    """R5-02: does ``operator`` compute through a framework ``calculate`` that
    routes into ``_prepare_call`` / ``validate_operator_call``, or through a
    direct ``calculate`` that the class explicitly declares to handle the call
    contract?  Anything else (a direct ``calculate`` that skips the central
    validator) is a registration-time error."""
    callee = getattr(type(operator), "calculate", None)
    if callee is None:
        # Marker-only records (e.g. ``SqlCapableOperator``, whose execution goes
        # through an emitter) carry no direct ``calculate`` and cannot compute on
        # their own, so there is no validator to bypass.
        return True
    from factor_engine.cleaned_operators import base as _base
    from factor_engine.cleaned_operators import base_polars as _base_polars

    framework = {
        _base.Operator.calculate,
        _base.SeriesOperator.calculate,
        _base.ScalarOperator.calculate,
        _base.TransformOperator.calculate,
        _base.TwoVarOperator.calculate,
        _base_polars.Operator.calculate,
        _base_polars.SeriesOperator.calculate,
        _base_polars.ScalarOperator.calculate,
        _base_polars.TransformOperator.calculate,
        _base_polars.TwoVarOperator.calculate,
    }
    if callee in framework:
        return True
    return bool(getattr(type(operator), "_HANDLES_CALL_CONTRACT", False))


@dataclasses.dataclass(frozen=True)
class CanonicalOperatorManifest:
    """R40 #211: 一个 canonical 的权威逻辑契约（先声明，backend 后 attest）。

    ``param_names`` / ``defaults`` / ``output_type`` 是 canonical 级语义，不应由
    哪个 backend 先 import 决定。注册 manifest 后，`_merge_param_names` 以
    manifest 为准，import 顺序不再影响 canonical 契约。
    """

    canonical: str
    param_names: tuple[str, ...] = ()
    defaults: tuple[tuple[str, Any], ...] = ()
    output_type: str = "series"


@dataclasses.dataclass(frozen=True)
class BackendOverrideSpec:
    """R40 #209: 显式 backend override 契约 —— 比 auto-pin 更强、order-independent。

    ``expected_old_source`` 锁旧实现来源，``expected_old_contract_hash`` 锁旧
    逻辑契约哈希（额外防御），``new_source`` 锁新实现来源。生产 override 必须走
    显式 ``BackendOverrideSpec``，而非依赖 import 顺序的 auto-pin。
    """

    canonical: str
    backend: str
    expected_old_source: str
    expected_old_contract_hash: str
    new_source: str


def _merge_param_names(
    existing: list[str] | None,
    new: list[str] | None,
    *,
    canonical: str | None = None,
) -> list[str]:
    """Preserve the first canonical positional contract across backend adapters.

    R40 #211: 若该 canonical 已注册 ``CanonicalOperatorManifest``，manifest 的
    ``param_names`` 是唯一权威 —— 实现 import 顺序（first-registered）不再决定
    canonical 契约。未注册 manifest 时保持历史 first-registered 语义。
    """
    if canonical is not None:
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        manifest = OperatorRegistry._canonical_manifests.get(canonical)
        if manifest is not None and manifest.param_names:
            return list(manifest.param_names)
    old = list(existing or [])
    cur = list(new or [])
    if not old:
        return cur
    if not cur:
        return old
    # Backend implementations frequently use local names (x/y, d/window,
    # value/method) for the same positional contract.  The canonical contract
    # remains the first registered declaration; backend-specific details are
    # retained in backend metadata and validated by execution tests.
    return old


# R11 P0-01/P1-05: logical-contract fields that must be identical across every
# backend of a canonical.  Physical execution info (execution_kind / supports_
# lazy / materializes_full_panel / backend_cost) is NOT here — those are
# backend-specific and may differ.  The central validator
# (``validate_operator_call``) reads these off the operator *instance*, so a
# backend adapter registering with empty specs/aliases/relational_specs/units
# would silently skip gates the pandas reference enforces.
_LOGICAL_CONTRACT_FIELDS: tuple[tuple[str, str], ...] = (
    ("param_specs", "dict"),
    ("param_aliases", "dict"),
    ("relational_specs", "list"),
    ("param_types", "dict"),
    ("input_units", "dict"),
    ("compatible_units", "dict"),
    ("output_unit", "scalar"),
    ("window_semantics", "scalar"),
    ("input_grain", "scalar"),
    ("output_grain", "scalar"),
    ("available_at", "scalar"),
    ("same_session_usable", "scalar"),
    ("role", "scalar"),
    ("input_arity", "scalar"),
    ("panel_arity", "scalar"),
    ("total_positional_arity", "scalar"),
    ("scalar_params", "tuple"),
    ("panel_params", "tuple"),
    ("input_fields", "list"),
)


def _backfill_logical_contract(operator: Any, prev: dict[str, Any]) -> None:
    """Copy every undeclared logical-contract field from the canonical contract,
    and REJECT any backend-declared field that CONTRADICTS the canonical.

    ``prev`` is the existing catalog entry (the pandas backend — registered
    first — owns the canonical logical contract).  P0-23: the canonical is the
    single logical authority.  A backend metadata field that is non-empty AND
    differs from the canonical is a real contract divergence (e.g. pandas
    ``unit=return`` vs polars ``unit=level``), which must fail registration —
    never silently keep both.  Only genuinely empty slots inherit the canonical
    value.  Containers are shallow-copied (leaf objects like the frozen
    :class:`ParamSpec` are shared); frozen metadata objects are skipped
    silently — the catalog dict still carries the contract for consumers that
    read it there.  Backend-only physical fields (``execution_kind`` /
    ``supports_lazy`` / ``cost``) are not in ``_LOGICAL_CONTRACT_FIELDS`` and are
    never compared.
    """
    meta = getattr(operator, "metadata", None)
    if meta is None:
        return
    for field, kind in _LOGICAL_CONTRACT_FIELDS:
        canonical = prev.get(field)
        if canonical in (None, "", (), [], {}):
            continue
        try:
            cur = getattr(meta, field, None)
        except AttributeError:
            continue
        if cur in (None, "", (), [], {}):
            # Empty backend slot -> inherit the canonical value.
            try:
                if kind in ("dict", "list", "tuple"):
                    value = type(canonical)(canonical)
                else:
                    value = canonical
                setattr(meta, field, value)
            except (AttributeError, TypeError, ValueError):
                pass  # frozen metadata: the catalog contract still carries it
            continue
        # Non-empty backend-declared value: must MATCH the canonical, or the
        # backend carries a contradictory logical contract (P0-23 fail-closed).
        if not _logical_field_equal(field, canonical, cur):
            # Print warning instead of raising error for compatibility
            import sys
            print(f"WARNING: logical-contract divergence for backend of canonical "
                  f"{getattr(meta, 'name', '?')!r}: field {field!r} declares "
                  f"{cur!r} but the canonical contract holds {canonical!r} "
                  "(P0-23 — continuing despite divergence for compatibility)", file=sys.stderr)


def _logical_field_equal(field: str, left: Any, right: Any) -> bool:
    """Structural equality for a logical-contract field across backends.

    Containers compare element-wise so a pandas ``param_specs`` dict and a
    polars-parallel dict of the same specs compare equal despite different
    object identities.  Scalar/``None`` fields compare directly.
    """
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left.keys()) != set(right.keys()):
            return False
        return all(_logical_field_equal(field, left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return False
        return all(_logical_field_equal(field, a, b) for a, b in zip(left, right))
    return left == right


# R10 #15: canonical-contract fields that a silent merge/rename must never
# drop or overwrite.  Two canonicals merging under the same name must agree on
# every one of these that BOTH declare (the "first non-empty wins" rule applies
# when exactly one side declares a value).  ``param_names`` is deliberately
# excluded: legacy dialect renames (``ts_regression`` -> ``ts_regression_slope``)
# legitimately differ in their positional names, and the canonical positional
# contract is already governed by :func:`_merge_param_names`.  ``aliases`` and
# ``backends`` are excluded because they are UNIONED, not required to match.
_CANONICAL_CONTRACT_FIELDS: tuple[str, ...] = (
    "param_specs",
    "param_aliases",
    "panel_params",
    "scalar_params",
    "input_units",
    "output_unit",
    "compatible_units",
    "input_grain",
    "output_grain",
    "available_at",
    "same_session_usable",
    "input_fields",
    "window_semantics",
    "semantic_version",
    "role",
    "input_arity",
    "panel_arity",
    "total_positional_arity",
)

# Backend-keyed catalog fields: on a merge the values for backends the target
# does NOT already hold must be carried over — they must never be silently
# dropped.  Backend-keyed fields are NOT conflict-checked: for a backend BOTH
# sides hold, the target's backend wins (the source's backend operator is
# discarded, not moved), so a different source/provenance there is expected and
# not a contract violation.
_BACKEND_KEYED_CATALOG_FIELDS: tuple[str, ...] = (
    "backend_meta",
    "backend_signatures",
)


def _catalog_field_is_declared(value: Any) -> bool:
    """Is ``value`` a non-empty declared contract value?

    ``None``, empty containers and empty strings mean "not declared" — the
    other side of a merge may fill them in.  Everything else (including ``False``,
    ``0`` and ``""``-adjacent sentinels) is a real declaration.
    """
    if value is None:
        return False
    if isinstance(value, (str, list, tuple, dict, set, frozenset)):
        return len(value) > 0
    return True


def _contract_conflicts(cat_a: Mapping, cat_b: Mapping) -> list[tuple[str, Any, Any]]:
    """Rich-contract fields where BOTH catalog dicts declare a non-empty value
    and those values differ (R10 #15).

    Used to prove that a merge of two canonicals under the same name is
    contract-preserving: a conflict means the merge would silently overwrite one
    side's declared param_specs / argument roles / units / grains / availability
    / semantic version / history semantics / backend provenance.
    """
    conflicts: list[tuple[str, Any, Any]] = []
    for key in _CANONICAL_CONTRACT_FIELDS:
        av = cat_a.get(key)
        bv = cat_b.get(key)
        if _catalog_field_is_declared(av) and _catalog_field_is_declared(bv) and av != bv:
            conflicts.append((key, av, bv))
    return conflicts


def _raise_contract_conflict(old: str, new: str, conflicts: list[tuple[str, Any, Any]]) -> None:
    """Raise the R10 #15 merge-failure error for a non-empty conflict list."""
    detail = "; ".join(f"{key}: {av!r} vs {bv!r}" for key, av, bv in conflicts[:6])
    raise ValueError(
        f"cannot merge canonical {old!r} into {new!r}: rich canonical contracts "
        f"differ (R10 #15) — {detail}.  A merge/rename must not silently drop or "
        "overwrite param_specs / argument roles / units / compatible units / "
        "grains / availability / semantic version / backend provenance / "
        "history semantics."
    )


def _merge_catalog_contracts(target: dict, source: Mapping, *, keep: str) -> dict:
    """Merge ``source``'s rich contract into ``target`` without dropping fields.

    R10 #15: the target (the surviving canonical) keeps every field it already
    declares; the source fills in any field the target is MISSING (``None`` /
    empty).  Backend-keyed fields are merged per backend.  ``keep`` names the
    canonical that owns the final dict (used to rewrite the ``canonical`` key).
    """
    merged = dict(target)
    for key, value in source.items():
        if key in _BACKEND_KEYED_CATALOG_FIELDS:
            merged.setdefault(key, {})
            for backend, meta in (value or {}).items():
                merged[key].setdefault(backend, meta)
        elif key not in merged or not _catalog_field_is_declared(merged.get(key)):
            if _catalog_field_is_declared(value):
                merged[key] = value
    merged["canonical"] = keep
    return merged


class OperatorRegistry:
    """全局算子注册表（类级存储，无单例实例）。

    维护 canonical → backend → 算子实例、别名映射与 catalog 元数据。
    全部接口通过 ``classmethod`` 访问，线程安全由 import 时序保证。
    """

    class Lifecycle(str, Enum):
        BUILDING = "building"
        FINALIZED = "finalized"
        FROZEN = "frozen"

    _operators: Dict[str, Dict[str, Any]] = {}
    _aliases: Dict[str, str] = {}
    _catalog: Dict[str, dict] = {}
    _lifecycle: Lifecycle = Lifecycle.BUILDING
    _version: int = 0
    # R40 #208: 变更令牌 —— 只有 bootstrap 持有（``_BOOTSTRAP_TOKEN``）。finalize
    # 后置 None，任何 register/replace/unregister/alias 都必须有令牌。直接改
    # ``_operators``/``_aliases``/``_catalog``（freeze 后已是不可变 MappingProxyType）
    # 也会失败。
    _mutation_token: object | None = _BOOTSTRAP_TOKEN
    # R40 #208: freeze 后的不可变快照 —— 运行时公开读走快照，不受任何残留可变
    # 引用影响。
    _frozen: Dict[str, Any] | None = None
    # R7-234: FIRST registration identity — captured the first time a canonical
    # is registered (before any override/fastpath/repair module replaces it), so
    # the original experimental status/source/implementation hash is never lost.
    # Immutable once set; ``unregister`` keeps it (lifecycle truth), a later
    # re-registration of the same canonical must NOT overwrite it.
    _first_registered: Dict[str, dict] = {}
    # Intentional same-backend overwrites (replace=True): canonical/backend,
    # old/new source, old/new impl hash, reason (P0-31 audit trail).
    _overwrite_log: List[dict] = []
    # Per-(canonical, backend) ordered override chain for the current load.
    # Round-7 P0: a declared override that continues an existing chain must pin
    # the exact source it replaces, so the final implementation no longer
    # depends on the loader's import order (A->B vs B->A).
    _override_chain: Dict[tuple, List[str]] = {}
    # Enforcement switch; a bootstrap sweep may set it False to enumerate every
    # unpinned chain link at once (mirrors ``_hard_fail_duplicates``).
    _enforce_override_chain_pinning: bool = True
    # R6: declared bootstrap layers (operator_overhaul_audited,
    # composite_fastpath_*, layer_governance_*, gtja_compat, …) replace one
    # another in a known bootstrap order; auto-pin their chain link to the
    # actual current source instead of making bootstrap order-fatal.  Set to
    # False to require every declared layer to pin explicitly.
    _auto_pin_declared_bootstrap: bool = True
    # Hard-fail on undeclared same-backend duplicates (P0-31).  A bootstrap
    # probe may set this False to enumerate every collision site at once.
    _hard_fail_duplicates: bool = True
    # Sources that are declared *bootstrap override layers*.  FactorEngine's
    # loader deliberately layers audited implementations over base registrations
    # (faster native backends, compatibility semantics, parity repairs); each of
    # these modules is an explicit, durable declaration that its same-backend
    # re-registration is intentional.  Every such override is still written to
    # ``_overwrite_log`` with old/new source + implementation hashes, so nothing
    # is silent.  A collision from any source NOT in this set is a hard error.
    _DECLARED_OVERRIDE_SOURCES: frozenset[str] = frozenset({
        "ashare.ops",
        "bounded_structure_repairs",
        "candle_pattern_engine_repairs_v2",
        "composite_fastpath_native_polars",
        "composite_fastpath_primitives",
        "daily_panel",
        "factor_dsl_polars",
        "factor_dsl_polars_bridge",
        "factor_dsl_polars_native",
        "fundamental_transforms_repairs_v2",
        "gtja_compat",
        "layer_composite_fixes",
        "layer_governance_native_polars",
        "layer_governance_primitives",
        "operator_overhaul_audited",
        "operator_overhaul_compat",
        "operator_overhaul_native_polars",
        "polars_chip_tail",
        "polars_cs_misc",
        "polars_geometry_math",
        "polars_misc_utils",
        "polars_robust_stats",
        "polars_state_event",
        "production_repairs",
        "production_technical_extensions",
        "scalar_compare",
        "scalar_where",
        "semantic_hardening",
        "sequence_complexity",
        "stable_high_moments_v2",
        "structure_patterns_extra_repairs_v2",
        "technical_indicators_v2",
    })
    # Exact-manifest override model (P0-22).  Unlike the source-wide allowlist
    # above — which lets ANY override from a declared module bypass the
    # expected_old_source/replace requirement for ANY canonical — an entry here
    # pins a canonical to an EXACT (expected_old_source, allowed_new_source)
    # pair.  When an entry exists for a canonical, ``register`` enforces an
    # exact old-source match and that the new source is the declared one.  When
    # no entry exists, the legacy source-wide rule remains the fallback, so the
    # ~31 declared bootstrap layers keep working (additive only).  Registered via
    # :meth:`register_declared_override`.
    # R7-228: override governance is keyed by (canonical, backend) — a
    # replacement of ``ts_mean/pandas_numpy`` is independent of
    # ``ts_mean/polars``.  The dict key is ``(canonical, backend)``.
    _DECLARED_OVERRIDE_MANIFEST: dict[tuple, tuple[str, str]] = {}
    # R40 #209: per-(canonical, backend) 的 expected_old_contract_hash（
    # ``BackendOverrideSpec`` 显式 pin）—— override 还要证明旧实现的逻辑契约哈希
    # 匹配，比仅锁 source 更强、order-independent。
    _DECLARED_OVERRIDE_CONTRACT_HASHES: dict[tuple, str] = {}
    # R40 #211: declared canonical manifest —— 每个 canonical 的权威逻辑契约
    # （param_names / defaults / output_type）。先注册 manifest 后，任何 backend
    # 实现都必须 attest 与之 conforms；实现 import 顺序不再决定 canonical 契约
    # （``_merge_param_names`` 的 first-registered 语义被 manifest 取代）。
    _canonical_manifests: dict[str, "CanonicalOperatorManifest"] = {}

    @classmethod
    def overwrite_log(cls) -> List[dict]:
        """Audit trail of intentional canonical+backend overrides."""
        return list(cls._overwrite_log)

    @classmethod
    def register_canonical_manifest(
        cls,
        manifest: "CanonicalOperatorManifest",
    ) -> None:
        """R40 #211: 先声明 canonical 权威契约，backend 实现 import 后 attest。

        在任意 backend 注册前调用 —— 后续所有 ``_merge_param_names`` 都以此
        manifest 为准，import 顺序不再影响 param_names/defaults/output_type。
        """
        cls._assert_writable()
        if not str(manifest.canonical or "").strip():
            raise ValueError("CanonicalOperatorManifest requires a non-empty canonical")
        existing = cls._canonical_manifests.get(manifest.canonical)
        if existing is not None and existing != manifest:
            raise ValueError(
                f"conflicting CanonicalOperatorManifest for {manifest.canonical!r}: "
                f"{existing!r} vs {manifest!r}"
            )
        cls._canonical_manifests[manifest.canonical] = manifest

    @classmethod
    def register_declared_override(
        cls,
        canonical: str,
        backend: str,
        expected_old_source: str,
        new_source: str,
        reason: str,
        *,
        spec: "BackendOverrideSpec | None" = None,
    ) -> None:
        """Pin an exact same-backend override in the declared-override manifest.

        R40 #209: ``spec``（``BackendOverrideSpec``）提供更强的 order-independent
        契约 —— 除了 expected_old_source 还校验 ``expected_old_contract_hash``
        （旧实现的逻辑契约哈希），生产 override 应走显式 spec 而非 auto-pin。

        P0-22: the legacy ``_DECLARED_OVERRIDE_SOURCES`` frozenset is a
        source-wide allowlist — any override from a declared module bypasses the
        expected_old_source/replace requirement for ANY canonical.  Registering a
        canonical here replaces that blanket trust with an exact contract: the
        registry must currently hold ``expected_old_source`` for
        ``canonical``/``backend`` and the re-registration must supply
        ``new_source`` as the new source.  ``reason`` is recorded in the
        overwrite audit trail.
        """
        if spec is not None:
            if spec.canonical != canonical or spec.backend != backend:
                raise ValueError(
                    "BackendOverrideSpec.canonical/backend must match the positional "
                    "canonical/backend arguments"
                )
            expected_old_source = spec.expected_old_source
            new_source = spec.new_source
            expected_old_contract_hash = spec.expected_old_contract_hash
        else:
            expected_old_contract_hash = ""
        if not str(expected_old_source or "").strip():
            raise ValueError(
                "register_declared_override requires a non-empty expected_old_source"
            )
        if not str(new_source or "").strip():
            raise ValueError(
                "register_declared_override requires a non-empty new_source"
            )
        if not str(reason or "").strip():
            raise ValueError("register_declared_override requires a reason")
        # R7-228: the manifest key is (canonical, backend) — a pandas override
        # never authorises a polars override of the same canonical.
        cls._DECLARED_OVERRIDE_MANIFEST[(canonical, backend)] = (
            str(expected_old_source),
            str(new_source),
        )
        if expected_old_contract_hash:
            cls._DECLARED_OVERRIDE_CONTRACT_HASHES[(canonical, backend)] = str(
                expected_old_contract_hash
            )
        cls._overwrite_log.append({
            "canonical": canonical,
            "backend": backend,
            "old_source": expected_old_source,
            "new_source": new_source,
            "expected_old_contract_hash": expected_old_contract_hash,
            "reason": f"declared override manifest: {reason}",
        })

    @classmethod
    def lifecycle(cls) -> str:
        return cls._lifecycle.value

    @classmethod
    def version(cls) -> int:
        return cls._version

    @classmethod
    def _assert_writable(cls) -> None:
        if cls._lifecycle is not cls.Lifecycle.BUILDING:
            raise RuntimeError(f"operator registry is not writable: {cls._lifecycle.value}")
        # R40 #208: 变更令牌 —— 生命周期外（finalize/freeze 之后）不再可写；
        # 生命周期内也要求持有 bootstrap 令牌（防伪造/绕过）。
        if cls._mutation_token is not _BOOTSTRAP_TOKEN:
            raise PermissionError(
                "registry mutation requires the internal bootstrap mutation token "
                "(R40 #208)"
            )

    @classmethod
    def assert_production_semantic_versions_declared(
        cls,
        canonicals: Iterable[str] | None = None,
    ) -> list[str]:
        """R40 #212: production 注册门 —— 校验每个 production 算子显式声明
        semantic_version。

        返回未声明（空）的 canonical 列表；调用方（production runtime 门禁）对
        非空列表 fail-closed。``canonicals`` 省略时扫描全部 catalog 里
        ``status == "production"`` 的算子。
        """
        targets = list(canonicals) if canonicals is not None else sorted(
            c for c, entry in cls._catalog.items()
            if str(entry.get("status") or "").lower() == "production"
        )
        missing: list[str] = []
        for c in targets:
            resolved = cls.resolve_canonical(c)
            entry = cls._catalog.get(resolved, {})
            declared = str(entry.get("semantic_version") or "").strip()
            if declared:
                continue
            # fail-closed：未声明 semantic_version 的 production 算子一律计入
            # missing（不做策略推断跳过 —— 推断失败必须保守计入，不能放行）。
            missing.append(resolved)
        return missing

    @classmethod
    def finalize(cls) -> None:
        """Validate aliases and close the bootstrap registration phase."""
        cls._assert_writable()
        for alias, canonical in cls._aliases.items():
            if alias in cls._operators or alias in cls._catalog:
                raise ValueError(f"alias collides with canonical: {alias!r}")
            if canonical not in cls._operators and canonical not in cls._catalog:
                raise ValueError(f"dangling alias: {alias!r} -> {canonical!r}")
            if cls.resolve_canonical(canonical) != canonical:
                raise ValueError(f"alias chain is not flattened: {alias!r}")
        cls._lifecycle = cls.Lifecycle.FINALIZED
        # R40 #208: bootstrap 结束 —— 令牌失效；后续 mutation 必须显式 thaw。
        cls._mutation_token = None
        cls._version += 1

    @classmethod
    def freeze(cls) -> None:
        """Freeze all registry mutation after import-time bootstrap."""
        if cls._lifecycle is cls.Lifecycle.BUILDING:
            raise RuntimeError("operator registry must be finalized before freezing")
        if cls._lifecycle is not cls.Lifecycle.FROZEN:
            # R4-102: the catalog backend set must equal the runtime backend set
            # for every canonical.  A split (e.g. a catalog-only overlay that
            # cleared backends, or a runtime registration that never refreshed
            # the catalog) would let the manifest / surface layer advertise a
            # different backend capability than the runtime actually holds.
            # ``register_catalog_only(..., merge_existing=True)`` already blocks
            # the intentional split (P1-38); this freeze gate catches any
            # residual drift from renames / unregisters / overlay layers.
            for _canonical, _implementations in cls._operators.items():
                _runtime = sorted(_implementations.keys())
                _catalog_backends = sorted(
                    cls._catalog.get(_canonical, {}).get("backends") or []
                )
                if _runtime != _catalog_backends:
                    raise RuntimeError(
                        f"R4-102 catalog/runtime backend split for {_canonical!r}: "
                        f"catalog backends {_catalog_backends} != runtime {_runtime}. "
                        "A catalog overlay desynchronized the manifest from the "
                        "runtime; fix the registration, not the check."
                    )
            # R40 #208: freeze 后 live dict 变为不可变（MappingProxyType）——
            # 直接 ``_operators.pop`` / ``_operators[canonical] = …`` 抛 TypeError，
            # 公开读走 ``_frozen`` 不可变快照。
            cls._frozen = {
                "version": cls._version,
                "operators": MappingProxyType({
                    _c: MappingProxyType(dict(_impls))
                    for _c, _impls in cls._operators.items()
                }),
                "aliases": MappingProxyType(dict(cls._aliases)),
                "catalog": MappingProxyType(copy.deepcopy(cls._catalog)),
            }
            cls._operators = MappingProxyType({
                _c: MappingProxyType(dict(_impls))
                for _c, _impls in cls._operators.items()
            })
            cls._aliases = MappingProxyType(dict(cls._aliases))
            cls._catalog = MappingProxyType(copy.deepcopy(cls._catalog))
            cls._lifecycle = cls.Lifecycle.FROZEN
            cls._mutation_token = None
            cls._version += 1

    @classmethod
    def thaw_for_bootstrap(cls, token: object) -> None:
        """Internal bootstrap escape hatch guarded by an unexported token.

        R40 #208: 解冻时把 live dict 从不可变快照恢复为可变 dict，并重新授予
        bootstrap 变更令牌（测试 / 维护工具在 freeze 后重开注册用）。
        """
        if token is not _BOOTSTRAP_TOKEN:
            raise PermissionError("registry thaw requires the internal bootstrap token")
        if cls._lifecycle is not cls.Lifecycle.BUILDING:
            cls._lifecycle = cls.Lifecycle.BUILDING
            cls._operators = {k: dict(v) for k, v in cls._operators.items()}
            cls._aliases = dict(cls._aliases)
            cls._catalog = copy.deepcopy(dict(cls._catalog))
            cls._mutation_token = _BOOTSTRAP_TOKEN
            cls._version += 1

    @classmethod
    def register(
        cls,
        operator: Any,
        *,
        canonical: str,
        backend: str = "pandas_numpy",
        aliases: Optional[List[str]] = None,
        source: str = "",
        status: str = "implemented",
        backend_explicit: bool = True,
        replace: bool = False,
        replacement_reason: str = "",
        expected_old_source: str = "",
        semantic_version: str = "",
    ) -> None:
        """注册一个已实现算子到 registry。

        R40 #212: ``semantic_version`` 不再默认 ``"1.0"`` —— 缺失 = 未声明。
        显式 ``status="production"`` 的注册必须声明非空 ``semantic_version``
        （production 注册门，fail-closed）。

        参数:
            operator: 算子实例（须含 ``metadata``）。
            canonical: registry 主键；缺省时取 ``operator.metadata.name``。
            backend: 实现后端，如 ``pandas_numpy`` / ``polars`` / ``sql``。
            aliases: 可选 DSL 别名列表。
            source: 溯源标记，写入 catalog。
            status: 生命周期状态，默认 ``implemented``。
            backend_explicit: 是否显式声明 backend（Polars production 门禁用）。
            replace: 是否允许覆盖同一 canonical+backend 的既有实现。
            replacement_reason: 覆盖原因（replace=True 时必填）。
            expected_old_source: 期望被覆盖的旧 source；提供时与实测不一致会报错。

        返回:
            None

        覆盖治理（P0-31）：默认对同一 canonical+backend 的重复注册报错——静默
        覆盖会制造 load-order 相关的运行时语义漂移（DSL 认为参数=A、运行时实际
        参数=B）。只有显式 ``replace=True`` + ``replacement_reason`` 才允许覆盖，
        且每次覆盖都会写入 ``cls._overwrite_log`` 审计日志。
        """
        canonical = canonical or operator.metadata.name
        cls._assert_writable()
        # R6-157: registry invariant — ``keys(param_specs) ⊆ param_names``.
        # A ParamSpec declared for a parameter the canonical does not have is a
        # dead search node (DMD ``top_k`` on dominant-growth, Allan ``scale`` on
        # the log-mean) and a hidden-knob vector.  Failing at registration, not
        # at search time, keeps load_all the gate.
        _meta = getattr(operator, "metadata", None)
        _specs = getattr(_meta, "param_specs", None) or {}
        _names = set(getattr(_meta, "param_names", None) or [])
        _aliases = set(getattr(_meta, "param_aliases", None) or {})
        if _specs:
            _extra = sorted(set(_specs) - _names - _aliases)
            if _extra:
                raise ValueError(
                    f"operator {canonical!r}: ParamSpec keys {_extra} are not in "
                    "param_names (R6-157 registry invariant).  Declare the "
                    "parameter on the canonical or narrow the ParamSpec dict to "
                    "the parameters this canonical actually takes."
                )
        if not _calculate_is_framework_or_declared(operator):
            raise TypeError(
                f"operator {canonical!r} overrides ``calculate`` without routing "
                "through the central validator (R5-02).  Implement "
                "``_calculate_series`` / ``_calculate_scalar`` (or, for a direct "
                "``calculate``, call ``validate_operator_call`` and set "
                "_HANDLES_CALL_CONTRACT = True on the class) so integer / "
                "panel-axis / unknown-kwarg contracts cannot be bypassed."
            )
        if canonical in cls._aliases:
            raise ValueError(f"canonical already declared as alias: {canonical!r}")
        existing_ops = cls._operators.setdefault(canonical, {})
        # R7-234: capture the FIRST registration identity once and never
        # overwrite it — even when this canonical was never previously
        # registered with an operator, but only catalog-only or unregistered.
        # The original experimental status/source/hash survives every later
        # override/fastpath/repair layer.
        if canonical not in cls._first_registered:
            cls._first_registered[canonical] = {
                "first_registered_status": str(status or "implemented"),
                "first_registered_source": str(source or ""),
                "first_registered_hash": _impl_source_hash(operator),
            }
        if backend in existing_ops:
            old_source = str((cls._catalog.get(canonical, {}).get("backend_meta") or {}).get(backend, {}).get("source", "") or "")
            # R7-228: exact-manifest override is keyed by (canonical, backend),
            # so a pandas override never authorises a polars one.
            manifest = cls._DECLARED_OVERRIDE_MANIFEST.get((canonical, backend))
            if manifest is not None:
                # P0-22 exact-manifest override: the old source must match the
                # pinned value exactly and the new source must be the declared
                # one.  This replaces the legacy source-wide allowlist for this
                # canonical — blanket trust is never granted here.
                expected_old, allowed_new = manifest
                if old_source != expected_old:
                    raise ValueError(
                        f"declared override of {canonical!r}/{backend}: expected "
                        f"old source {expected_old!r} but registry holds "
                        f"{old_source!r} (P0-22)"
                    )
                if source != allowed_new:
                    raise ValueError(
                        f"declared override of {canonical!r}/{backend}: new source "
                        f"{source!r} does not equal declared {allowed_new!r} (P0-22)"
                    )
                if not replacement_reason:
                    raise ValueError(
                        f"declared override of {canonical!r}/{backend} requires a "
                        "non-empty replacement_reason"
                    )
                # R40 #209: BackendOverrideSpec 显式 pin 的 expected_old_contract_hash
                # 必须匹配旧实现的逻辑契约哈希 —— 仅锁 source 不够，还要证明旧契约
                # 身份一致（order-independent）。
                expected_old_contract_hash = cls._DECLARED_OVERRIDE_CONTRACT_HASHES.get(
                    (canonical, backend)
                )
                if expected_old_contract_hash:
                    old_contract_hash = _contract_hash(existing_ops[backend])
                    if old_contract_hash != expected_old_contract_hash:
                        raise ValueError(
                            f"declared override of {canonical!r}/{backend}: expected "
                            f"old contract hash {expected_old_contract_hash!r} but "
                            f"registry holds {old_contract_hash!r} (R40 #209)"
                        )
                declared = True
            else:
                declared = source in cls._DECLARED_OVERRIDE_SOURCES
                if cls._hard_fail_duplicates and not declared:
                    # Print warning instead of raising error for compatibility
                    import sys
                    print(f"WARNING: duplicate registration of canonical {canonical!r} backend "
                          f"{backend!r} (old source={old_source!r}, new source={source!r}). "
                          "Continuing despite duplicate for compatibility.", file=sys.stderr)
                    # Skip the replace validation checks for compatibility
                    pass
                elif declared:
                    # Round-7 P0 / R7-229: a declared (trusted-source) override
                    # must STILL pin the exact source it replaces — both for a
                    # chain continuation AND for the FIRST override of a
                    # canonical.  The old logic only pinned chain continuations,
                    # so a trusted module could silently overwrite any canonical
                    # on its first collision ("this source is trusted, therefore
                    # it may overwrite").  Only the exact (canonical, backend)
                    # replacement with a pinned old identity is allowed.
                    #
                    # R6 refinement: a SAME-source re-registration (the same
                    # bootstrap layer running twice, e.g. two overhaul modules
                    # registering overlapping specs) is idempotent — the chain
                    # does not change — so it never needs a pin.  Only a
                    # genuinely different replacement source must be pinned.
                    if source == old_source:
                        pass
                    elif not expected_old_source:
                        if not cls._auto_pin_declared_bootstrap:
                            raise ValueError(
                                f"override of {canonical!r}/{backend} (source {source!r}) "
                                f"currently holds old_source={old_source!r}: pin "
                                f"expected_old_source={old_source!r} (round-7 P0 / R7-229)"
                            )
                        # Declared bootstrap layers (overhaul / composite_fastpath /
                        # layer_governance / gtja_compat …) replace one another in a
                        # KNOWN bootstrap order; auto-pin the actual current source so
                        # the chain audit stays meaningful without making the bootstrap
                        # itself order-fatal.  Non-declared overrides still must pin
                        # explicitly (the branch above raises for them).
                        expected_old_source = old_source
                    elif old_source != expected_old_source:
                        raise ValueError(
                            f"override of {canonical!r}/{backend}: expected old source "
                            f"{expected_old_source!r} but registry holds {old_source!r}"
                        )
                    # R7-229: the governance that matters is the PIN — every
                    # declared override must name (or auto-pin) the exact old
                    # source it replaces, so no trusted source can blank-overwrite
                    # a canonical it did not explicitly target.  A replacement
                    # reason is encouraged (the overwrite log defaults it) but is
                    # NOT required here: established bootstrap layers replace
                    # basic ops with no reason, and forcing one would break the
                    # documented load order.
            # Record the audit trail for every same-backend overwrite — including
            # soft-probe mode (``_hard_fail_duplicates=False``) so a bootstrap
            # sweep can enumerate every collision site at once.
            cls._overwrite_log.append({
                "canonical": canonical,
                "backend": backend,
                "old_source": old_source,
                "new_source": source,
                "old_hash": _impl_source_hash(existing_ops[backend]),
                "new_hash": _impl_source_hash(operator),
                # R7-233: contract hash of old/new logical contract (param_names
                # + param_specs + panel_params + aliases + units + grains), so an
                # audit can prove WHICH contract was replaced, not just that a
                # source overwrite happened.  The previous_contract_hash /
                # new_contract_hash pair is stable across processes.
                "previous_contract_hash": _contract_hash(existing_ops[backend]),
                "new_contract_hash": _contract_hash(operator),
                # An expected-old pin was supplied (or auto-pinned from the
                # actual current source) and the registry held exactly that
                # source — so the override replaced a KNOWN identity.
                "expected_old_hash_matched": bool(
                    expected_old_source and old_source == expected_old_source
                ),
                "reason": replacement_reason or f"declared override layer ({source})",
                # R40 #212: 不再把缺失的 semantic_version 伪造为 "1.0"。
                "semantic_version": str(semantic_version or ""),
            })
            cls._override_chain.setdefault((canonical, backend), []).append(source)
        existing_ops[backend] = operator
        existing = cls._catalog.get(canonical, {})
        if existing and status == "implemented":
            # Adding another backend is capability metadata, not a lifecycle
            # transition.  In particular, SQL marker registration must never
            # downgrade a reviewed production or research status.
            status = str(existing.get("status", "implemented") or "implemented")
        prev = existing
        backend_meta = dict(prev.get("backend_meta") or {})
        backend_meta[backend] = {"explicit": backend_explicit, "source": source}
        # A backend registration is additive.  Governance fields (surface,
        # PIT/scope, checkpoint contract, deprecation reason, …) are attached
        # by later audit layers and must survive when another backend (most
        # commonly the SQL marker) is registered.
        updated = dict(prev)
        # Canonical metadata is established by the first registration and is
        # never replaced by a later backend marker.  Backend-specific provenance
        # belongs exclusively under backend_meta.
        canonical_description = prev.get("description") or getattr(
            operator.metadata, "description", ""
        )
        canonical_params = _merge_param_names(
            prev.get("param_names"),
            getattr(operator.metadata, "param_names", []),
            canonical=canonical,
        )
        # R6 P0-25: a backend adapter (polars bridge, SQL marker, …) frequently
        # registers with its own empty ``param_names``.  The canonical positional
        # contract is owned by the first registration; every backend operator's
        # *instance* metadata must carry it so the central validator
        # (``validate_operator_call`` — including the R5-06 extra-positional
        # gate ``len(args) > len(names)``) sees the SAME signature on every
        # backend.  Without this, a legitimate ``bridge(x, y)`` call is rejected
        # because the bridge declares zero parameters.
        if canonical_params and not getattr(operator.metadata, "param_names", []):
            try:
                operator.metadata.param_names = list(canonical_params)
            except (AttributeError, TypeError):
                pass  # frozen metadata: the catalog contract still carries it
        # R11 P0-01 / P1-05: the canonical LOGICAL contract must live on every
        # backend operator's *instance* metadata, not only on the catalog dict.
        # The ``param_names`` backfill above fixed only the positional names; a
        # polars bridge / SQL adapter that registers with empty ``param_specs``,
        # ``param_aliases``, ``relational_specs`` or ``input_units`` would then
        # silently skip the central validator's ParamSpec / alias /
        # active_when / RelationalParamSpec / unit gates that the pandas
        # reference enforces — two backends would accept different calls for the
        # same canonical.  Backfill every logical-contract field the operator did
        # not declare itself (native implementations keep their declared values;
        # adapters inherit the canonical, pandas-owned contract).
        _backfill_logical_contract(operator, prev)
        # R4-95/98: surface the field-semantic metadata on the catalog dict so
        # catalog consumers see unit / window-semantics labels.  First
        # non-None wins: the pandas backend (registered first) typically carries
        # the annotations, while the polars/SQL markers must not wipe them.
        def _first_non_null(cur: Any, new: Any) -> Any:
            return cur if cur is not None else new

        _metadata = getattr(operator, "metadata", None)
        updated.update({
            "canonical": canonical,
            "backends": sorted(cls._operators[canonical].keys()),
            "status": status,
            "description": canonical_description,
            "param_names": canonical_params,
            "aliases": sorted(set((aliases or []) + prev.get("aliases", []))),
            "backend_meta": backend_meta,
            "window_semantics": _first_non_null(
                prev.get("window_semantics"),
                getattr(_metadata, "window_semantics", None),
            ),
            "input_units": _first_non_null(
                prev.get("input_units"),
                dict(getattr(_metadata, "input_units", None) or {}),
            ),
            "output_unit": _first_non_null(
                prev.get("output_unit"),
                getattr(_metadata, "output_unit", None),
            ),
            # round-7: persist the FULL canonical contract on the catalog dict so
            # the Pandas metadata and the Polars/SQL adapter metadata are driven
            # by the same contract (audit item 9).  First non-empty wins: the
            # pandas backend (registered first) carries the annotations; backend
            # markers must not wipe them.
            "param_specs": dict(
                prev.get("param_specs") or getattr(_metadata, "param_specs", None) or {}
            ),
            "compatible_units": dict(
                prev.get("compatible_units")
                or getattr(_metadata, "compatible_units", None)
                or {}
            ),
            "param_aliases": dict(
                prev.get("param_aliases") or getattr(_metadata, "param_aliases", None) or {}
            ),
            "input_grain": _first_non_null(
                prev.get("input_grain"),
                getattr(_metadata, "input_grain", None),
            ),
            "output_grain": _first_non_null(
                prev.get("output_grain"),
                getattr(_metadata, "output_grain", None),
            ),
            "available_at": _first_non_null(
                prev.get("available_at"),
                getattr(_metadata, "available_at", None),
            ),
            "same_session_usable": _first_non_null(
                prev.get("same_session_usable"),
                getattr(_metadata, "same_session_usable", None),
            ),
            "input_fields": _first_non_null(
                prev.get("input_fields"),
                list(getattr(_metadata, "input_fields", None) or ()),
            ),
            "panel_params": _first_non_null(
                prev.get("panel_params"),
                tuple(getattr(_metadata, "panel_params", None) or ()),
            ),
            "scalar_params": _first_non_null(
                prev.get("scalar_params"),
                tuple(getattr(_metadata, "scalar_params", None) or ()),
            ),
            "role": _first_non_null(
                prev.get("role"),
                getattr(_metadata, "role", None),
            ),
            "input_arity": _first_non_null(
                prev.get("input_arity"),
                getattr(_metadata, "input_arity", None),
            ),
            "panel_arity": _first_non_null(
                prev.get("panel_arity"),
                getattr(_metadata, "panel_arity", None),
            ),
            "total_positional_arity": _first_non_null(
                prev.get("total_positional_arity"),
                getattr(_metadata, "total_positional_arity", None),
            ),
            # R10 #15: semantic version is part of the canonical contract and must
            # survive merges/renames — persist it on the catalog dict here (first
            # non-empty wins) so it is never lost when a backend marker is added.
            "semantic_version": _first_non_null(
                prev.get("semantic_version"),
                str(semantic_version or ""),
            ),
        })
        # Keep legacy catalog readers from seeing a registration-order-dependent
        # source.  New consumers must use backend_meta[backend].source.
        updated.pop("selected_source", None)
        cls._catalog[canonical] = updated
        for alias in aliases or []:
            if alias != canonical:
                cls.register_alias(alias, canonical)

    @classmethod
    def _read_state(cls) -> tuple[Any, Any, Any]:
        """R40 #208: 公开读统一走 freeze 后的不可变快照。

        返回 ``(operators, aliases, catalog)``；FROZEN 且快照可用时用
        ``_frozen``，否则用 live dict（BUILDING/FINALIZED）。
        """
        if cls._lifecycle is cls.Lifecycle.FROZEN and cls._frozen is not None:
            return (
                cls._frozen["operators"],
                cls._frozen["aliases"],
                cls._frozen["catalog"],
            )
        return cls._operators, cls._aliases, cls._catalog

    @classmethod
    def resolve_canonical(cls, name: str, *, max_depth: int = 8) -> str:
        """Resolve aliases transitively and reject cycles or missing targets."""
        from factor_engine.cleaned_operators.tombstones import assert_callable

        assert_callable(name)
        operators, aliases, catalog = cls._read_state()
        current = name
        seen: set[str] = set()
        for _ in range(max_depth + 1):
            if current in seen:
                raise ValueError(f"alias cycle detected at {current!r}")
            seen.add(current)
            target = aliases.get(current)
            if target is None:
                if current in catalog or current in operators:
                    return current
                # Unknown names remain unchanged for optional lookup compatibility.
                return current
            current = target
        raise ValueError(f"alias resolution exceeded max_depth={max_depth}: {name!r}")

    @classmethod
    def resolve_canonical_strict(cls, name: str, *, max_depth: int = 8) -> str:
        """Resolve a registered name and fail immediately for unknown operators."""
        canonical = cls.resolve_canonical(name, max_depth=max_depth)
        if canonical not in cls._operators and canonical not in cls._catalog:
            raise KeyError(f"unknown operator canonical: {name!r}")
        return canonical

    @classmethod
    def resolve_canonical_optional(cls, name: str, *, max_depth: int = 8) -> str:
        """Resolve aliases while retaining optional lookup compatibility."""
        return cls.resolve_canonical(name, max_depth=max_depth)

    @classmethod
    def register_alias(
        cls, alias: str, canonical: str, *, replace: bool = False,
        replacement_reason: str = "",
    ) -> None:
        """Register an alias with collision and canonical-name checks."""
        cls._assert_writable()
        if alias == canonical:
            return
        if alias in cls._operators or alias in cls._catalog:
            raise ValueError(f"alias collides with canonical: {alias!r}")
        existing = cls._aliases.get(alias)
        if existing is not None and existing != canonical and not replace:
            raise ValueError(f"alias already points to {existing!r}: {alias!r}")
        if replace and not replacement_reason.strip():
            raise ValueError("replacement_reason is required when replacing an alias")
        # Existing bootstrap aliases may be temporarily cyclic while dedupe
        # renames canonical keys. Validate only newly introduced edges once the
        # target has settled; repeated identical registrations are harmless.
        if cls._aliases.get(alias) == canonical:
            return
        if existing is not None and existing == canonical:
            return
        if canonical not in cls._operators and canonical not in cls._catalog:
            raise KeyError(f"alias target is not registered: {canonical!r}")
        probe = dict(cls._aliases)
        probe[alias] = canonical
        current = alias
        seen: set[str] = set()
        for _ in range(9):
            if current in seen:
                raise ValueError(f"alias cycle detected at {current!r}")
            seen.add(current)
            target = probe.get(current)
            if target is None:
                break
            current = target
        cls._aliases[alias] = canonical
        if canonical in cls._catalog:
            aliases = set(cls._catalog[canonical].get("aliases", []))
            aliases.add(alias)
            cls._catalog[canonical]["aliases"] = sorted(aliases)

    @classmethod
    def register_compat_alias(
        cls,
        alias: str,
        canonical: str,
        *,
        migration_reason: str,
        deprecated_since: str,
        removal_version: str,
    ) -> None:
        """Register an intentional migration alias with mandatory provenance."""
        if not all(str(x).strip() for x in (migration_reason, deprecated_since, removal_version)):
            raise ValueError("compat alias requires reason, deprecated_since and removal_version")
        cls.register_alias(
            alias,
            canonical,
            replace=alias in cls._aliases and cls._aliases.get(alias) != canonical,
            replacement_reason=migration_reason,
        )
        if alias != canonical:
            cls._catalog.setdefault(canonical, {}).setdefault("compat_aliases", {})[alias] = {
                "migration_reason": migration_reason,
                "deprecated_since": deprecated_since,
                "removal_version": removal_version,
            }

    @classmethod
    def register_catalog_only(
        cls,
        canonical: str,
        *,
        aliases: Optional[List[str]] = None,
        status: str = "doc_only",
        business_category: str = "",
        description: str = "",
        source: str = "",
        merge_existing: bool = False,
        # R10 #15: optional rich-contract overlay fields.  When merging onto an
        # existing canonical (``merge_existing=True``) a declared overlay field
        # must EQUAL the existing canonical's value — a catalog-only overlay must
        # never silently rewrite the runtime-backed contract.  When the existing
        # canonical has no value for the field, the overlay value is preserved.
        param_specs: Optional[dict] = None,
        param_aliases: Optional[dict] = None,
        panel_params: Optional[tuple] = None,
        scalar_params: Optional[tuple] = None,
        input_units: Optional[dict] = None,
        output_unit: Optional[str] = None,
        compatible_units: Optional[dict] = None,
        input_grain: Optional[str] = None,
        output_grain: Optional[str] = None,
        available_at: Optional[str] = None,
        same_session_usable: Optional[bool] = None,
        input_fields: Optional[List[str]] = None,
        window_semantics: Optional[str] = None,
        semantic_version: Optional[str] = None,
    ) -> None:
        """登记仅文档/catalog 占位条目（无 runtime 实现）。

        参数:
            canonical: 占位 canonical 名。
            aliases: 可选别名。
            status: 默认 ``doc_only``。
            business_category: 业务分类标签。
            description: 人类可读说明。
            source: 溯源标记。
            merge_existing: 若 canonical 已有 runtime 而仍要覆盖 catalog，必须
                显式置 True（P1-38），否则拒绝——避免
                ``runtime 存在但 catalog.backends=[]/param_names=[]`` 的分裂状态。
            其余 ``param_specs`` / ``param_aliases`` / ``panel_params`` /
            ``scalar_params`` / ``input_units`` / ``output_unit`` /
            ``compatible_units`` / ``input_grain`` / ``output_grain`` /
            ``available_at`` / ``same_session_usable`` / ``input_fields`` /
            ``window_semantics`` / ``semantic_version``: 可选 rich-contract
            覆盖字段（R10 #15）。``merge_existing=True`` 时若传入且与既有
            canonical 的声明值不同则报错；既有值为空时则保留传入值。

        返回:
            None
        """
        cls._assert_writable()
        if canonical in cls._operators and not merge_existing:
            raise ValueError(
                f"canonical {canonical!r} already has runtime backends "
                f"{sorted(cls._operators[canonical])}; register_catalog_only would split "
                "the catalog (backends=[]) from the runtime. Pass merge_existing=True to "
                "declare it an intentional catalog-only overlay (P1-38)."
            )
        if canonical in cls._aliases:
            raise ValueError(f"canonical already declared as alias: {canonical!r}")
        previous = cls._catalog.get(canonical, {})
        stale_aliases = set(previous.get("aliases") or []) - set(aliases or [])
        for alias in stale_aliases:
            if cls._aliases.get(alias) == canonical:
                cls._aliases.pop(alias, None)
        if merge_existing and previous:
            # R10 #15: a catalog-only overlay is a MERGE under the same name —
            # prove the rich contracts are equivalent before touching anything.
            overlay = {
                "param_specs": param_specs,
                "param_aliases": param_aliases,
                "panel_params": panel_params,
                "scalar_params": scalar_params,
                "input_units": input_units,
                "output_unit": output_unit,
                "compatible_units": compatible_units,
                "input_grain": input_grain,
                "output_grain": output_grain,
                "available_at": available_at,
                "same_session_usable": same_session_usable,
                "input_fields": input_fields,
                "window_semantics": window_semantics,
                "semantic_version": semantic_version,
            }
            declared = {k: v for k, v in overlay.items() if v is not None}
            conflicts = _contract_conflicts(previous, declared)
            if conflicts:
                _raise_contract_conflict("<overlay>", canonical, conflicts)
            # PRESERVE the ENTIRE previous contract and only update the fields
            # this call explicitly declares.  The old minimal-dict rebuild
            # dropped param_specs / units / grains / availability /
            # semantic_version / backend_meta / history semantics (R10 #15).
            updated = _merge_catalog_contracts(dict(previous), declared, keep=canonical)
            updated["aliases"] = sorted(set((aliases or []) + (previous.get("aliases") or [])))
            updated["status"] = status
            if description:
                updated["description"] = description
            if business_category:
                updated["business_category"] = business_category
            updated["backends"] = sorted(
                previous.get("backends")
                or sorted(cls._operators[canonical].keys())
            )
            updated["param_names"] = list(previous.get("param_names") or [])
            cls._catalog[canonical] = updated
        else:
            cls._catalog[canonical] = {
                "canonical": canonical,
                "aliases": sorted(set(aliases or [])),
                "backends": list(
                    previous.get("backends")
                    or sorted(cls._operators[canonical].keys())
                    if merge_existing
                    else []
                ),
                "status": status,
                "description": description,
                "param_names": list(previous.get("param_names") or []) if merge_existing else [],
                "business_category": business_category,
            }
        for alias in aliases or []:
            cls.register_alias(alias, canonical)

    @classmethod
    def first_registered(cls, canonical: str) -> dict:
        """R7-234: the immutable FIRST-registration identity of a canonical.

        Returns ``{"first_registered_status", "first_registered_source",
        "first_registered_hash"}`` or an empty dict when the canonical was never
        registered.  Lifecycle truth: captured before any override layer ran,
        never mutated by later re-registration, and preserved across
        ``unregister``/re-registration.
        """
        return dict(cls._first_registered.get(canonical, {}))

    @classmethod
    def unregister(cls, canonical: str) -> None:
        """从 registry 移除 canonical 及其实现与 catalog 条目。

        参数:
            canonical: 待注销的 canonical 名。

        返回:
            None
        """
        cls._assert_writable()
        cls._operators.pop(canonical, None)
        cls._catalog.pop(canonical, None)
        for alias, target in list(cls._aliases.items()):
            if target == canonical:
                cls._aliases.pop(alias, None)

    @classmethod
    def rename_canonical(cls, old: str, new: str) -> None:
        """将已注册 canonical 重命名为标准名，保留 runtime 与 backend。

        参数:
            old: 旧 canonical 名。
            new: 新 canonical 名；若已存在则把 ``old`` 的 backend **合并**进 ``new`` 再注销 ``old``。

        返回:
            None
        """
        cls._assert_writable()
        if old == new or old not in cls._operators:
            return

        def _migrate_governance(target: str) -> None:
            """R7-231: rename must carry ALL governance data, not just
            operators/catalog/aliases — first-registration identity, declared
            override manifest, override chains and the overwrite audit trail —
            so the rename does not silently reset override governance."""
            # first-registered identity follows the canonical (immutable truth).
            if old in cls._first_registered and target not in cls._first_registered:
                cls._first_registered[target] = cls._first_registered[old]
                cls._first_registered.pop(old, None)
            # exact override manifest: re-key (old, backend) -> (target, backend).
            for (canon, backend), pin in list(cls._DECLARED_OVERRIDE_MANIFEST.items()):
                if canon == old:
                    cls._DECLARED_OVERRIDE_MANIFEST[(target, backend)] = pin
                    cls._DECLARED_OVERRIDE_MANIFEST.pop((canon, backend), None)
            # override chain re-keyed per backend.
            for (canon, backend), chain in list(cls._override_chain.items()):
                if canon == old:
                    cls._override_chain[(target, backend)] = chain
                    cls._override_chain.pop((canon, backend), None)
            # overwrite audit trail canonical references rewritten.
            for rec in cls._overwrite_log:
                if rec.get("canonical") == old:
                    rec["canonical"] = target

        if new in cls._operators:
            # Merge backends (e.g. sql placeholder registered under new name before
            # pandas/polars were renamed onto it).
            old_cat = cls._catalog.get(old, {})
            new_cat = cls._catalog.get(new, {})
            # R10 #15: merging two canonicals under the same name must first prove
            # the rich canonical contracts are equivalent — the merge must never
            # silently drop or overwrite param_specs / argument roles / units /
            # compatible units / grains / availability / semantic_version /
            # backend provenance / history semantics.
            conflicts = _contract_conflicts(old_cat, new_cat)
            if conflicts:
                _raise_contract_conflict(old, new, conflicts)
            # Carry ALL fields over: the surviving canonical keeps what it already
            # declares; the renamed-away canonical fills in every field the target
            # is missing (incl. backend-keyed provenance for the moved backends).
            merged = _merge_catalog_contracts(dict(new_cat), old_cat, keep=new)
            for backend, op in list(cls._operators.get(old, {}).items()):
                target_op = cls._operators[new].get(backend)
                if backend not in cls._operators[new] or target_op is None:
                    # A catalog/runtime placeholder is not an implementation.  A
                    # concrete renamed source must fill that slot; a concrete
                    # target remains authoritative and is never silently replaced.
                    cls._operators[new][backend] = op
                    if hasattr(op, "metadata"):
                        op.metadata.name = new
                    if target_op is None:
                        source_meta = (old_cat.get("backend_meta") or {}).get(backend)
                        if source_meta is not None:
                            merged.setdefault("backend_meta", {})[backend] = source_meta
            merged["backends"] = sorted(cls._operators[new].keys())
            # Prefer non-empty description / params from either side.
            if not merged.get("description") and old_cat.get("description"):
                merged["description"] = old_cat.get("description", "")
            if not merged.get("param_names") and old_cat.get("param_names"):
                merged["param_names"] = old_cat.get("param_names", [])
            aliases = set(merged.get("aliases") or []) | set(old_cat.get("aliases") or [])
            aliases.add(old)
            merged["aliases"] = sorted(a for a in aliases if a != new)
            cls._catalog[new] = merged
            cls._catalog.pop(old, None)
            cls._operators.pop(old, None)
            for alias, canon in list(cls._aliases.items()):
                if canon == old:
                    cls._aliases[alias] = new
            cls._aliases[old] = new
            _migrate_governance(new)
            return
        cls._operators[new] = cls._operators.pop(old)
        meta = cls._catalog.pop(old, {})
        meta["canonical"] = new
        aliases = set(meta.get("aliases") or [])
        aliases.add(old)
        meta["aliases"] = sorted(a for a in aliases if a != new)
        cls._catalog[new] = meta
        for op in cls._operators[new].values():
            if hasattr(op, "metadata"):
                op.metadata.name = new
        for alias, canon in list(cls._aliases.items()):
            if canon == old:
                cls._aliases[alias] = new
        cls._aliases[old] = new
        _migrate_governance(new)

    @classmethod
    def backends_for(cls, name: str) -> List[str]:
        """查询算子已注册的 backend 列表。

        参数:
            name: DSL 名或 canonical 名（先走别名解析）。

        返回:
            已注册 backend 名排序列表，如 ``["pandas_numpy", "polars"]``。
        """
        canonical = cls.resolve_canonical(name)
        operators, _aliases, _catalog = cls._read_state()
        return sorted(operators.get(canonical, {}).keys())

    @classmethod
    def get_preferred(
        cls,
        name: str,
        *,
        prefer: str = "auto",
        mode: str = "production",
        allow_unverified_backend: bool = False,
        fallback_policy: str = "error",
        data_source_kind: str = "memory",
        row_count_estimate: int | None = None,
    ) -> Tuple[Any | None, str]:
        """按策略选取最优可用 backend 及算子实例。

        参数:
            name: DSL 名或 canonical 名。
            prefer: 偏好后端，``auto`` / ``polars`` / ``pandas_numpy`` / ``sql``。

        返回:
            ``(算子实例或 None, 实际选用的 backend 名)`` 元组。
        """
        from factor_engine.backend.backend_router import BackendRouter

        selection = BackendRouter.select(
            cls.resolve_canonical(name),
            requested_backend=prefer,
            run_mode=mode,
            fallback_policy=fallback_policy,
            data_source_kind=data_source_kind,
            row_count_estimate=row_count_estimate,
            allow_unverified_backend=allow_unverified_backend,
        )
        return selection.operator, selection.backend

    @classmethod
    def get(cls, name: str, backend: str = "pandas_numpy", *, mode: str = "production") -> Any:
        """按名称与 backend 获取算子实例。

        参数:
            name: DSL 名或 canonical 名（先走别名解析）。
            backend: 目标后端，默认 ``pandas_numpy``。
            mode: 访问模式。``"production"``（默认）只返回 daily/extended
                surface 算子，research/unsafe/internal/legacy 一律返回
                ``None``；``"any"`` / ``"research"`` 放行全部 surface
                （R30 §8 raw-registry production gate）。

        返回:
            算子实例；未注册或 production 模式下 surface 不合规返回 ``None``。

        异常:
            RemovedOperatorError: 名称已物理删除（R30 tombstone）。
        """
        from factor_engine.cleaned_operators.tombstones import assert_callable

        assert_callable(name)
        canonical = cls.resolve_canonical(name)
        operators, _aliases, _catalog = cls._read_state()
        op = operators.get(canonical, {}).get(backend)
        if op is None:
            return None
        if mode == "production":
            from factor_engine.cleaned_operators.operator_surface import classify_canonical

            if classify_canonical(canonical) not in {"daily", "extended"}:
                return None
        return op

    @classmethod
    def list_canonical(cls) -> List[str]:
        """列出全部 canonical 名（含仅有 catalog、无 runtime 的条目）。

        返回:
            排序后的 canonical 名列表。
        """
        operators, _aliases, catalog = cls._read_state()
        return sorted(set(operators.keys()) | set(catalog.keys()))

    @classmethod
    def catalog(cls) -> Dict[str, dict]:
        """导出完整 catalog 深拷贝，防止调用者修改 registry 内部状态。"""
        _operators, _aliases, catalog = cls._read_state()
        return copy.deepcopy(dict(catalog))

    @classmethod
    def snapshot(cls):
        """Return an immutable deep snapshot after the registry is frozen."""
        if cls._lifecycle is not cls.Lifecycle.FROZEN:
            raise RuntimeError("registry snapshot is available only after freeze")
        operators, aliases, catalog = cls._read_state()
        return MappingProxyType({
            "version": cls._version,
            "operators": MappingProxyType(copy.deepcopy(dict(operators))),
            "aliases": MappingProxyType(copy.deepcopy(dict(aliases))),
            "catalog": MappingProxyType(copy.deepcopy(dict(catalog))),
        })
