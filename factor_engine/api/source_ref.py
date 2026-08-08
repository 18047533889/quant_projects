# -*- coding: utf-8 -*-
from __future__ import annotations

import base64
import json
from dataclasses import dataclass, replace
from typing import Any, Mapping

from api.columns import col

_PREFIX = "__fe_source_ref_v1__"
_MINUTE_FIELDS = {
    "open":"Open", "high":"High", "low":"Low", "close":"Close",
    "volume":"Volume", "amount":"Amount", "vwap":"Vwap",
}

@dataclass(frozen=True)
class SourceRefSpec:
    table: str
    field: str
    params: tuple[tuple[str, Any], ...] = ()
    transform: str | None = None
    transform_params: tuple[tuple[str, Any], ...] = ()
    dialect: str = "lqtp"
    dialect_version: str = "2026-07-19"

    def params_dict(self) -> dict[str, Any]: return dict(self.params)
    def transform_params_dict(self) -> dict[str, Any]: return dict(self.transform_params)
    def with_transform(self, transform: str, **params: Any) -> "SourceRefSpec":
        return replace(self, transform=str(transform), transform_params=tuple(sorted(params.items())))
    def to_payload(self) -> dict[str, Any]:
        return {"table":self.table,"field":self.field,"params":dict(self.params),
                "transform":self.transform,"transform_params":dict(self.transform_params),
                "dialect":self.dialect,"dialect_version":self.dialect_version}

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

def make_source_ref(table: str, field: str, *, params: Mapping[str, Any] | None = None,
                    transform: str | None = None, transform_params: Mapping[str, Any] | None = None,
                    dialect: str = "lqtp", dialect_version: str = "2026-07-19") -> SourceRefSpec:
    table, field = str(table).strip(), str(field).strip()
    if not table or not field: raise ValueError("source table and field must be non-empty")
    return SourceRefSpec(table=table, field=field,
        params=tuple(sorted((str(k),_scalar(v)) for k,v in dict(params or {}).items())),
        transform=transform,
        transform_params=tuple(sorted((str(k),_scalar(v)) for k,v in dict(transform_params or {}).items())),
        dialect=str(dialect), dialect_version=str(dialect_version))

def encode_source_ref(spec: SourceRefSpec) -> str:
    raw=json.dumps(spec.to_payload(),sort_keys=True,separators=(",",":"),ensure_ascii=True)
    return _PREFIX + base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")

def decode_source_ref(name: str) -> SourceRefSpec | None:
    text=str(name)
    if not text.startswith(_PREFIX): return None
    token=text[len(_PREFIX):]
    payload=json.loads(base64.urlsafe_b64decode(token+"="*(-len(token)%4)).decode())
    return make_source_ref(payload["table"],payload["field"],params=payload.get("params") or {},
        transform=payload.get("transform"),transform_params=payload.get("transform_params") or {},
        dialect=payload.get("dialect","lqtp"),dialect_version=payload.get("dialect_version","2026-07-19"))

def looks_like_source_ref(name: str) -> bool:
    """Audit #392: cheap structural check that a name carries the SourceRef
    prefix.  Consistent with the ``_PREFIX`` constant so a prefix rename stays in
    sync.  Does NOT validate the payload — use ``decode_source_ref_strict`` for
    that."""
    return isinstance(name, str) and name.startswith(_PREFIX)

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
    for i in range(0,len(param_items),2): params[str(param_items[i])] = param_items[i+1]
    return col(encode_source_ref(make_source_ref(table,field,params=params,dialect=dialect,dialect_version=dialect_version)))

def transform_source_col(value: Any, transform: str, **params: Any):
    name=getattr(value,"name",None)
    spec=decode_source_ref(name) if isinstance(name,str) else None
    # Legacy LQTP formula packs commonly write minute_bar(close, 5, 0)
    # instead of minute_bar(StockMinuteBar.Close, 5, 0).  This is the only
    # implicit source inference permitted here because the manual defines these
    # minute helpers specifically over StockMinuteBar fields.
    if spec is None and str(transform).startswith("minute_") and isinstance(name,str):
        field=_MINUTE_FIELDS.get(name.lower())
        if field is not None:
            spec=make_source_ref("StockMinuteBar",field)
    if spec is None: raise ValueError(f"{transform}() requires a logical DataTable field")
    return col(encode_source_ref(spec.with_transform(transform,**params)))

def intermediate_col(name: str, version: int):
    version=int(version)
    if version<=0: raise ValueError("intermediate version must be positive")
    return source_col("Intermediate","value",name=str(name),version=version)
