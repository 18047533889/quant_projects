"""Provider-neutral factor-pack contracts for AutoFactorEvaluation."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol, runtime_checkable


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FactorDefinition:
    name: str
    formula: str
    source_formula: str = ""
    description: str = ""
    formula_hash: str = ""
    metadata: Mapping[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        metadata = {} if self.metadata is None else dict(self.metadata)
        object.__setattr__(self, "metadata", metadata)
        expected = sha256_text(self.formula)
        if not self.name.strip():
            raise ValueError("factor name must be non-empty")
        if not self.formula.strip():
            raise ValueError(f"factor {self.name} formula must be non-empty")
        if self.formula_hash and self.formula_hash != expected:
            raise ValueError(f"factor {self.name} formula hash mismatch")
        object.__setattr__(self, "formula_hash", expected)
        if not self.description:
            object.__setattr__(self, "description", self.name)


@dataclass(frozen=True)
class FactorPack:
    name: str
    version: str
    source_hash: str
    pack_hash: str
    factors: tuple[FactorDefinition, ...]
    metadata: Mapping[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", {} if self.metadata is None else dict(self.metadata))
        self.validate()

    @property
    def source_catalog_hash(self) -> str:
        """Compatibility alias used by historical reports."""
        return self.source_hash

    def names(self) -> list[str]:
        return [factor.name for factor in self.factors]

    def by_name(self) -> dict[str, FactorDefinition]:
        return {factor.name: factor for factor in self.factors}

    def select(
        self,
        names: Iterable[str] | None = None,
        *,
        limit: int | None = None,
    ) -> tuple[FactorDefinition, ...]:
        selected = list(self.factors)
        if names is not None:
            requested = list(dict.fromkeys(str(name) for name in names))
            mapping = self.by_name()
            missing = [name for name in requested if name not in mapping]
            if missing:
                raise KeyError(f"unknown factors in pack {self.name}: {missing[:10]}")
            selected = [mapping[name] for name in requested]
        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be positive")
            selected = selected[:limit]
        return tuple(selected)

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("factor-pack name must be non-empty")
        if not self.version.strip():
            raise ValueError(f"factor pack {self.name} version must be non-empty")
        if not self.factors:
            raise ValueError(f"factor pack {self.name} is empty")
        names = self.names()
        if len(set(names)) != len(names):
            raise ValueError(f"factor pack {self.name} contains duplicate factor names")
        hashes = [factor.formula_hash for factor in self.factors]
        if len(set(hashes)) != len(hashes):
            raise ValueError(f"factor pack {self.name} contains duplicate formulas")


@runtime_checkable
class FactorPackProvider(Protocol):
    def __call__(self) -> FactorPack: ...
