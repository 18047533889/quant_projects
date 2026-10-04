"""Immutable, payload-free research feature receipt metadata."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from itertools import islice
from numbers import Real
from typing import Any

import numpy as np

from ._canonical import canonical_digest

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be nonempty trimmed text")
    return value


def _digest(name: str, value: Any, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


@dataclass(frozen=True)
class ResearchSourceBinding:
    """Caller-declared source references and expected component identities."""

    factor_id: str
    manifest_uri: str
    manifest_sha256: str
    source_uri: str
    source_sha256: str
    time_axis_hash: str
    asset_axis_hash: str
    values_hash: str
    validity_hash: str | None = None

    def __post_init__(self) -> None:
        for name in ("factor_id", "manifest_uri", "source_uri"):
            object.__setattr__(self, name, _text(name, getattr(self, name)))
        for name in ("manifest_sha256", "source_sha256", "time_axis_hash",
                     "asset_axis_hash", "values_hash"):
            object.__setattr__(self, name, _digest(name, getattr(self, name)))
        object.__setattr__(self, "validity_hash",
                           _digest("validity_hash", self.validity_hash, optional=True))


@dataclass(frozen=True)
class ResearchFeatureReceipt:
    """Metadata-only receipt for a single-factor T×N×1 feature."""

    source_binding: ResearchSourceBinding
    values_dtype: str
    validity_dtype: str | None
    shape: tuple[int, int, int]
    time_axis_name: str
    time_axis_dtype: str
    asset_axis_name: str
    asset_axis_dtype: str
    embedding: tuple[float, ...]
    embedding_spec: str
    embedding_model_version: str
    scope: str = "research_only"
    schema_version: int = 1
    source_binding_status: str = "caller_supplied"
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.source_binding, ResearchSourceBinding):
            raise TypeError("source_binding must be ResearchSourceBinding")
        for name in ("values_dtype", "time_axis_name", "time_axis_dtype",
                     "asset_axis_name", "asset_axis_dtype", "embedding_spec", "embedding_model_version"):
            object.__setattr__(self, name, _text(name, getattr(self, name)))
        if self.validity_dtype is not None:
            object.__setattr__(self, "validity_dtype", _text("validity_dtype", self.validity_dtype))
        if (self.validity_dtype is None) != (self.source_binding.validity_hash is None):
            raise ValueError("validity hash and dtype presence must agree")
        if self.scope != "research_only":
            raise ValueError("scope must be research_only")
        if self.source_binding_status != "caller_supplied":
            raise ValueError("source_binding_status must be caller_supplied")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("schema_version must be 1")

        shape = tuple(islice(iter(self.shape), 4))
        if (len(shape) != 3 or any(type(dim) is not int or dim <= 0 for dim in shape)
                or shape[-1] != 1):
            raise ValueError("shape must be positive integer T×N×1")
        object.__setattr__(self, "shape", shape)

        try:
            raw_embedding = tuple(islice(iter(self.embedding), 4097))
        except TypeError as exc:
            raise ValueError("embedding must be a finite numeric sequence") from exc
        if not raw_embedding or len(raw_embedding) > 4096:
            raise ValueError("embedding length must be between 1 and 4096")
        normalized: list[float] = []
        for value in raw_embedding:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (Real, np.floating)):
                raise ValueError("embedding values must be real numbers")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("embedding values must be finite")
            if isinstance(value, np.floating) and np.asarray(value).dtype.itemsize > 8:
                if np.longdouble(number) != value:
                    raise ValueError("embedding value loses precision as float64")
            normalized.append(number)
        embedding = tuple(normalized)
        object.__setattr__(self, "embedding", embedding)

        binding = self.source_binding
        object.__setattr__(self, "content_hash", canonical_digest(
            binding.factor_id, binding.manifest_uri, binding.manifest_sha256,
            binding.source_uri, binding.source_sha256, binding.time_axis_hash,
            binding.asset_axis_hash, binding.values_hash, binding.validity_hash,
            self.source_binding_status, self.values_dtype, self.validity_dtype,
            shape, self.time_axis_name, self.time_axis_dtype,
            self.asset_axis_name, self.asset_axis_dtype, embedding,
            self.embedding_spec, self.embedding_model_version, self.scope, self.schema_version,
        ))
