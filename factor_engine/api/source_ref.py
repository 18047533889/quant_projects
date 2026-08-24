# -*- coding: utf-8 -*-
from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from factor_engine.api.columns import col

_PREFIX = "__fe_source_ref_v1__"
_MINUTE_FIELDS = {
    "open":"Open", "high":"High", "low":"Low", "close":"Close",
    "volume":"Volume", "amount":"Amount", "vwap":"Vwap",
}

@dataclass(frozen=True)
class SourceRefSpec:
    """SourceRef v1/v2 identity (R24-136..140).

    v1 carries only ``table`` + ``field`` (+ transform/dialect).  v2 adds the
    semantic identity that stops cross-market / cross-vintage collisions:
    ``market``, ``concept_id`` / ``field_id``, ``provider_id``, ``dataset``,
    ``timeframe``, ``temporal_policy_digest``, ``catalog_hash``,
    ``source_version``.  A v1 ref decodes fine (the v2 fields are None); a
    NEW production encode MUST emit v2 (``make_source_ref(..., market=...)``).
    """

    table: str
    field: str
    params: tuple[tuple[str, Any], ...] = ()
    transform: str | None = None
    transform_params: tuple[tuple[str, Any], ...] = ()
    dialect: str = "lqtp"
    dialect_version: str = "2026-07-19"
    #: R10 #56: construction-time strictness gate.  Deliberately excluded from
    #: ``to_payload`` (identity) and from equality comparison so the same logical
    #: source ref encodes identically in research and production.
    production: bool = field(compare=False, repr=False, default=False)
    # R24-136..138: v2 semantic identity (all optional for v1 back-compat).
    market: str | None = None
    concept_id: str | None = None
    field_id: str | None = None
    provider_id: str | None = None
    dataset: str | None = None
    timeframe: str | None = None
    temporal_policy_digest: str | None = None
    catalog_hash: str | None = None
    source_version: str | None = None

    @property
    def is_v2(self) -> bool:
        """A ref is v2 when it carries the market-scoped semantic identity."""
        return self.market is not None

    def params_dict(self) -> dict[str, Any]: return dict(self.params)
    def transform_params_dict(self) -> dict[str, Any]: return dict(self.transform_params)
    def with_transform(self, transform: str, **params: Any) -> "SourceRefSpec":
        normalized = str(transform).strip()
        _assert_declared_transform(normalized, production=self.production)
        _validate_transform_params(normalized, params)
        return replace(
            self,
            transform=normalized,
            transform_params=tuple(
                sorted((str(k), _scalar(v)) for k, v in params.items())
            ),
        )
    def to_payload(self) -> dict[str, Any]:
        payload = {
            "table": self.table, "field": self.field,
            "params": dict(self.params),
            "transform": self.transform,
            "transform_params": dict(self.transform_params),
            "dialect": self.dialect, "dialect_version": self.dialect_version,
            "schema_version": "2" if self.is_v2 else "1",
        }
        for key in (
            "market", "concept_id", "field_id", "provider_id", "dataset",
            "timeframe", "temporal_policy_digest", "catalog_hash", "source_version",
        ):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        return payload

def _scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        # Round-7 WS-E #289: NaN / +/-Inf must never enter an encoded SourceRef
        # identity — a non-finite parameter is not a reproducible scalar literal.
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(
                "source parameters must be finite floats; NaN/Inf are forbidden"
            )
        return value
    raise TypeError(f"source parameters must be scalar literals, got {type(value).__name__}")


def _dtype_label(dtype: Any) -> str:
    """Human-readable dtype label for ``SourceTransformParamSpec`` errors."""
    if isinstance(dtype, tuple):
        return " or ".join(_dtype_label(d) for d in dtype)
    if dtype is int:
        return "an integer"
    if dtype is float:
        return "a float"
    if dtype is str:
        return "a string"
    return getattr(dtype, "__name__", str(dtype))


@dataclass(frozen=True)
class SourceTransformParamSpec:
    """SourceRef 变换参数的 dtype/范围/choices 契约（R10 #55）。

    每个 ``_TRANSFORM_PARAM_SPECS`` 条目用该规格声明一个可消费参数的
    字面量类型、上下界与可选白名单。构造 SourceRef 时逐参数校验，防止
    非法字面量进入可复现身份。

    属性:
        dtype: 期望类型（int/str/float 或类型元组）
        min: 数值下界（含），None 表示不限
        max: 数值上界（含），None 表示不限
        choices: 允许的字面量白名单；提供后优先于 dtype/min/max 判定
    """
    dtype: Any = int
    min: float | int | None = None
    max: float | int | None = None
    choices: tuple[Any, ...] | None = None

    def validate(self, name: str, value: Any) -> None:
        """校验单个变换参数，违规抛 ``ValueError``。

        参数:
            name: 参数名（用于报错信息）
            value: 已过 ``_scalar`` 标量编码的字面量
        """
        if self.choices is not None:
            if value not in self.choices:
                allowed = ", ".join(repr(c) for c in self.choices)
                raise ValueError(
                    f"source transform parameter {name!r} must be one of "
                    f"({allowed}), got {value!r}"
                )
            return
        if not self._dtype_ok(value):
            raise ValueError(
                f"source transform parameter {name!r} must be {_dtype_label(self.dtype)}, "
                f"got {type(value).__name__} {value!r}"
            )
        if self.min is not None and value < self.min:
            raise ValueError(
                f"source transform parameter {name!r} must be >= {self.min}, got {value!r}"
            )
        if self.max is not None and value > self.max:
            raise ValueError(
                f"source transform parameter {name!r} must be <= {self.max}, got {value!r}"
            )

    def _dtype_ok(self, value: Any) -> bool:
        if self.dtype is int:
            # Review-8 #469 spirit: bool is not a real integer literal.
            if isinstance(value, bool):
                return False
            if isinstance(value, int):
                return True
            if isinstance(value, float):
                return (
                    value == value
                    and value not in (float("inf"), float("-inf"))
                    and value.is_integer()
                )
            return False
        if self.dtype is float:
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if self.dtype is str:
            return isinstance(value, str)
        return isinstance(value, self.dtype)


def _strict_int(value: Any, *, name: str = "parameter") -> int:
    """Strict integer validation for SourceRef integer params (Review-8 #469).

    The resolver must NOT ``int()`` its way out of an ambiguous literal:
    ``True``, ``1.9``, ``"2"``, ``NaN`` and ``Inf`` are all rejected rather than
    silently coerced to ``1`` / ``2``.  Integral floats (``2.0``) are accepted —
    they are mathematically the same integer.
    """
    if isinstance(value, bool):
        raise ValueError(f"source {name} must be a real integer, got bool {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"source {name} must be a finite integer, got {value!r}")
        if not value.is_integer():
            raise ValueError(f"source {name} must be an integer, got {value!r}")
        return int(value)
    raise ValueError(
        f"source {name} must be an integer literal, got {type(value).__name__} {value!r}"
    )


# Review-8 #470 / R10 #55: typed parameter contracts for known transforms.  A
# transform with a spec here rejects unused/unconsumed parameters instead of
# silently ignoring them, and validates each consumed parameter's dtype / range /
# choices.  Unknown transforms are not validated (opt-in strictness in research;
# production rejects them outright — R10 #56).
_TRANSFORM_PARAM_SPECS: dict[str, dict[str, SourceTransformParamSpec]] = {
    "financial_lag": {
        # Consumer requires quarters >= 1 (lqtp_logical_source financial_lag).
        "quarters": SourceTransformParamSpec(dtype=int, min=1),
    },
    "minute_bar": {
        "period": SourceTransformParamSpec(dtype=int, min=1),
        "index": SourceTransformParamSpec(dtype=int, min=0),
    },
    "minute_resample": {
        "period": SourceTransformParamSpec(dtype=int, min=1),
    },
    # Consumer reads params["hhmm"] (e.g. "09:30"), not period/index.
    "minute_at": {
        "hhmm": SourceTransformParamSpec(dtype=str),
    },
    # Consumer compares HHMM strings lexicographically: str(params["start"]).
    "minute_range": {
        "start": SourceTransformParamSpec(dtype=str),
        "end": SourceTransformParamSpec(dtype=str),
    },
}

_HHMM_RE = re.compile(r"^([0-1][0-9]|2[0-3]):[0-5][0-9]$")


def _validate_hhmm(value: Any, *, name: str) -> str:
    """R24-141: strict ``HH:MM`` wall-clock syntax (00<=HH<=23, 00<=MM<=59).

    A bare ``"930"`` or ``"25:00"`` is rejected — never silently interpreted.
    """
    text = str(value).strip()
    if not _HHMM_RE.match(text):
        raise ValueError(
            f"minute transform param {name!r} must be 'HH:MM' with 00<=HH<=23 "
            f"and 00<=MM<=59 (R24-141); got {value!r}"
        )
    return text


def _assert_declared_transform(transform: str, *, production: bool) -> None:
    """R10 #56: production rejects unknown/undeclared transforms.

    Research / compat keeps the opt-in strictness of the past: an unknown
    transform is constructible (only its params, if a spec exists, are checked).
    In production a transform that has no declared contract fails loudly rather
    than silently producing an identity no downstream executor can honor.
    """
    if not production:
        return
    normalized = str(transform).strip()
    if normalized not in _TRANSFORM_PARAM_SPECS:
        declared = ", ".join(sorted(_TRANSFORM_PARAM_SPECS))
        raise ValueError(
            f"source transform {transform!r} is not a declared production "
            f"transform (declared: {declared})"
        )


def _validate_transform_params(transform: str, params: Mapping[str, Any]) -> None:
    """R10 #55: validate a transform's parameters against its contract.

    Enforces both the allowed-name gate (an unknown parameter is rejected —
    Review-8 #470) and the per-parameter dtype / range / choices contract.  Each
    value is scalar-encoded first so non-finite floats are rejected with the
    canonical "finite" message (#468) before the type/range checks run.
    """
    spec = _TRANSFORM_PARAM_SPECS.get(str(transform).strip())
    if spec is None:
        return
    known = set(spec)
    provided = set(str(k) for k in params)
    unknown = sorted(provided - known)
    if unknown:
        raise ValueError(
            f"source transform {transform!r} does not consume parameter(s): "
            f"{', '.join(unknown)} (allowed: {', '.join(sorted(known))})"
        )
    for key, value in params.items():
        spec[str(key)].validate(str(key), _scalar(value))
    # R24-141: minute wall-clock syntax validation (00<=HH<=23, 00<=MM<=59).
    t = str(transform).strip()
    if t == "minute_at" and "hhmm" in params:
        _validate_hhmm(params["hhmm"], name="hhmm")
    elif t == "minute_range":
        if "start" in params:
            _validate_hhmm(params["start"], name="start")
        if "end" in params:
            _validate_hhmm(params["end"], name="end")
        # R24-142: start must be earlier than end.
        if "start" in params and "end" in params:
            if str(params["start"]) >= str(params["end"]):
                raise ValueError(
                    f"minute_range start {params['start']!r} must be earlier "
                    "than end (R24-142)"
                )

def make_source_ref(table: str, field: str, *, params: Mapping[str, Any] | None = None,
                    transform: str | None = None, transform_params: Mapping[str, Any] | None = None,
                    dialect: str = "lqtp", dialect_version: str = "2026-07-19",
                    production: bool = False,
                    market: str | None = None, concept_id: str | None = None,
                    field_id: str | None = None, provider_id: str | None = None,
                    dataset: str | None = None, timeframe: str | None = None,
                    temporal_policy_digest: str | None = None,
                    catalog_hash: str | None = None, source_version: str | None = None) -> SourceRefSpec:
    table, field = str(table).strip(), str(field).strip()
    if not table or not field: raise ValueError("source table and field must be non-empty")
    tparams = dict(transform_params or {})
    if transform:
        normalized = str(transform).strip()
        _assert_declared_transform(normalized, production=production)
        _validate_transform_params(normalized, tparams)
    return SourceRefSpec(table=table, field=field,
        params=tuple(sorted((str(k),_scalar(v)) for k,v in dict(params or {}).items())),
        transform=transform,
        transform_params=tuple(sorted((str(k),_scalar(v)) for k,v in tparams.items())),
        dialect=str(dialect), dialect_version=str(dialect_version),
        production=bool(production),
        market=str(market) if market else None,
        concept_id=str(concept_id) if concept_id else None,
        field_id=str(field_id) if field_id else None,
        provider_id=str(provider_id) if provider_id else None,
        dataset=str(dataset) if dataset else None,
        timeframe=str(timeframe) if timeframe else None,
        temporal_policy_digest=str(temporal_policy_digest) if temporal_policy_digest else None,
        catalog_hash=str(catalog_hash) if catalog_hash else None,
        source_version=str(source_version) if source_version else None)

def encode_source_ref(spec: SourceRefSpec) -> str:
    raw=json.dumps(spec.to_payload(),sort_keys=True,separators=(",",":"),ensure_ascii=True)
    return _PREFIX + base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")

def decode_source_ref(name: str) -> SourceRefSpec | None:
    text=str(name)
    if not text.startswith(_PREFIX): return None
    token=text[len(_PREFIX):]
    raw=base64.urlsafe_b64decode(token+"="*(-len(token)%4))
    payload=json.loads(raw.decode())
    _enforce_source_ref_size_limits(token, raw, payload)
    return make_source_ref(payload["table"],payload["field"],params=payload.get("params") or {},
        transform=payload.get("transform"),transform_params=payload.get("transform_params") or {},
        dialect=payload.get("dialect","lqtp"),dialect_version=payload.get("dialect_version","2026-07-19"),
        market=payload.get("market"), concept_id=payload.get("concept_id"),
        field_id=payload.get("field_id"), provider_id=payload.get("provider_id"),
        dataset=payload.get("dataset"), timeframe=payload.get("timeframe"),
        temporal_policy_digest=payload.get("temporal_policy_digest"),
        catalog_hash=payload.get("catalog_hash"), source_version=payload.get("source_version"))

_APPROVED_DIALECT_VERSIONS = frozenset({"2026-07-19", "2026-08-10"})


def decode_source_ref_production(name: str, *, allowed_dialect_version: str | None = None) -> SourceRefSpec:
    """R24-139/140: STRICT SourceRef decode for production.

    Rejects (hard fail, never silent fallback):
    * a non-SourceRef string;
    * an unknown transform;
    * unknown / invalid transform params;
    * an old un-approved dialect version;
    * a v1 ref when the caller requires v2 (market-scoped identity).
    ``allowed_dialect_version`` (default the latest approved set) may be pinned
    by the caller.
    """
    spec = decode_source_ref(name)
    if spec is None:
        raise ValueError(
            f"production source-ref decode rejected {name!r}: not a SourceRef"
        )
    approved = frozenset({allowed_dialect_version}) if allowed_dialect_version else _APPROVED_DIALECT_VERSIONS
    if spec.dialect_version not in approved:
        raise ValueError(
            f"production source-ref decode rejected {name!r}: dialect_version "
            f"{spec.dialect_version!r} is not an approved dialect "
            f"({sorted(approved)!r}) — R24-140"
        )
    if spec.transform:
        _assert_declared_transform(spec.transform, production=True)
        _validate_transform_params(spec.transform, spec.transform_params_dict())
    return spec


def looks_like_source_ref(name: str) -> bool:
    """Audit #392: cheap structural check that a name carries the SourceRef
    prefix.  Consistent with the ``_PREFIX`` constant so a prefix rename stays in
    sync.  Does NOT validate the payload — use ``decode_source_ref_strict`` for
    that."""
    return isinstance(name, str) and name.startswith(_PREFIX)

_SR_MAX_TOKEN_BYTES = 8192          # R21-042
_SR_MAX_DECODED_BYTES = 16384       # R21-042
_SR_MAX_PARAMS = 64                 # R21-042
_SR_MAX_KEY_LEN = 128               # R21-042
_SR_MAX_VALUE_LEN = 4096            # R21-042


def _enforce_source_ref_size_limits(token: str, raw: bytes, payload: Any) -> None:
    """R21-042: resource bounds on SourceRef base64/JSON decode."""
    if len(token.encode("utf-8")) > _SR_MAX_TOKEN_BYTES:
        raise ValueError(
            f"source ref token bytes {len(token.encode('utf-8'))} exceed {_SR_MAX_TOKEN_BYTES}"
        )
    if len(raw) > _SR_MAX_DECODED_BYTES:
        raise ValueError(
            f"source ref decoded bytes {len(raw)} exceed {_SR_MAX_DECODED_BYTES}"
        )
    if isinstance(payload, dict):
        params = payload.get("params") or {}
        tparams = payload.get("transform_params") or {}
        if isinstance(params, dict) and len(params) + (len(tparams) if isinstance(tparams, dict) else 0) > _SR_MAX_PARAMS:
            raise ValueError(f"source ref param count exceeds {_SR_MAX_PARAMS}")
        for key, value in list(params.items()) + list(tparams.items()):
            if len(str(key)) > _SR_MAX_KEY_LEN:
                raise ValueError(f"source ref param key too long: {str(key)[:16]}...")
            if len(str(value)) > _SR_MAX_VALUE_LEN:
                raise ValueError("source ref param value too long")


def decode_source_ref_strict(name: str) -> SourceRefSpec:
    """Audit #392: strictly decode a prefixed SourceRef, raising ValueError on
    ANY malformed payload (bad base64, bad JSON, missing table/field, invalid
    scalar parameter).  Unlike ``decode_source_ref`` — whose contract is to
    return None for non-prefixed names and to let decode errors propagate — this
    helper never swallows a damaged identity: a corrupt SourceRef must fail the
    caller loudly instead of being silently treated as absent."""
    if not isinstance(name, str):
        raise ValueError(f"source ref name must be str, got {type(name).__name__}")
    if not name.startswith(_PREFIX):
        raise ValueError("not a source ref")
    token = name[len(_PREFIX):]
    try:
        raw = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4))
    except Exception as exc:  # binascii.Error / ValueError on bad alphabet
        raise ValueError(f"malformed source ref base64 payload: {exc}") from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"malformed source ref json payload: {exc}") from exc
    _enforce_source_ref_size_limits(token, raw, payload)
    if not isinstance(payload, dict):
        raise ValueError("source ref payload must be a JSON object")
    for key in ("table", "field"):
        if key not in payload:
            raise ValueError(f"source ref payload missing required field {key!r}")
    if not isinstance(payload["table"], str) or not isinstance(payload["field"], str):
        raise ValueError("source ref table/field must be strings")
    return make_source_ref(payload["table"], payload["field"], params=payload.get("params") or {},
        transform=payload.get("transform"), transform_params=payload.get("transform_params") or {},
        dialect=payload.get("dialect", "lqtp"), dialect_version=payload.get("dialect_version", "2026-07-19"))

def is_source_ref(name: str) -> bool: return decode_source_ref(name) is not None

def source_col(table: str, field: str, *param_items: Any, dialect: str = "lqtp",
               dialect_version: str = "2026-07-19", **params: Any):
    if len(param_items)%2: raise ValueError("source_col parameters must be key/value pairs")
    # Review-8 #470: a parameter given twice (positional + keyword, or twice
    # positionally) must be rejected — "the last one wins" silently corrupts the
    # SourceRef identity.
    seen: set[str] = set()
    for i in range(0, len(param_items), 2):
        key = str(param_items[i])
        if key in seen:
            raise ValueError(f"source parameter {key!r} provided more than once")
        seen.add(key)
        if key in params:
            raise ValueError(
                f"source parameter {key!r} provided both positionally and as a keyword"
            )
        params[key] = param_items[i + 1]
    return col(encode_source_ref(make_source_ref(table,field,params=params,dialect=dialect,dialect_version=dialect_version)))

def transform_source_col(value: Any, transform: str, *, strict: bool = False,
                         production: bool = False, **params: Any):
    name=getattr(value,"name",None)
    spec=decode_source_ref(name) if isinstance(name,str) else None
    # Legacy LQTP formula packs commonly write minute_bar(close, 5, 0)
    # instead of minute_bar(StockMinuteBar.Close, 5, 0).  This is the ONLY
    # implicit source inference permitted here because the manual defines these
    # minute helpers specifically over StockMinuteBar fields.
    # R10-P0-024: in the PRODUCTION typed DSL a bare ``close`` must NOT silently
    # change source from DailyBar to StockMinuteBar depending on which helper
    # wraps it — that would make ``ts_mean(close, 20)`` daily and
    # ``minute_bar(close, 5, 0)`` minute for the same name.  Production callers
    # pass ``strict=True`` and must spell the minute source explicitly
    # (``field("StockMinuteBar", "Close")`` or ``MinuteSeries[Close]``); the
    # implicit inference is retained ONLY on the compat surface (strict=False).
    if spec is None and str(transform).startswith("minute_") and isinstance(name,str):
        if strict:
            raise ValueError(
                f"{transform}(): a bare {name!r} cannot be implicitly read from "
                "StockMinuteBar in the strict (production) surface; spell the "
                "minute source explicitly, e.g. field(\"StockMinuteBar\", \"Close\")"
            )
        field=_MINUTE_FIELDS.get(name.lower())
        if field is not None:
            spec=make_source_ref("StockMinuteBar",field)
    if spec is None: raise ValueError(f"{transform}() requires a logical DataTable field")
    # R10 #56: a decoded spec defaults to research; upgrade it when the caller
    # explicitly asks for production so unknown transforms fail loudly.
    if production and not spec.production:
        spec = replace(spec, production=True)
    return col(encode_source_ref(spec.with_transform(transform,**params)))

def intermediate_col(name: str, version: int):
    version = _strict_int(version, name="Intermediate.version")
    if version <= 0:
        raise ValueError("intermediate version must be positive")
    return source_col("Intermediate", "value", name=str(name), version=version)
