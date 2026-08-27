"""Package-local hashable frozen-mapping helper for factor_assets contracts.

FA canonical artifacts (``ClusterArtifact``, ``SimilarityArtifact`` and the
governance artifacts) snapshot caller-supplied mappings into an immutable,
DEEP-FROZEN, **hashable** form.  The standard :class:`types.MappingProxyType`
wraps only the *top level* of a dict: it mutation-guards the surface but — as
observed by the DLIB-XPKG cross-package gate — a raw proxy is not
``copy.deepcopy``-safe and a ``(frozen=True)`` dataclass whose field holds a
plain ``dict``-shaped proxy is not hashable (``TypeError: unhashable type:
'dict'``).

This module fixes that by recursively freezing nested dict / list / tuple /
set values into hashable immutables (matching the QE ``FrozenMapping``
convention referenced by the XPKG gate), and by giving the returned mapping a
stable content-derived ``__hash__``, so FA artifacts become deepcopy-safe and
hashable while remaining mutation-guarded and content-hash stable.

Design notes:

- **Recursive freeze** — dict / Mapping -> :class:`FrozenMapping`; list /
  tuple -> tuple; set / frozenset -> frozenset; scalar immutables pass
  through.  Any other *mutable* object type raises ``TypeError`` fail-closed
  so the frozen artifact is provably un-mutable.
- **Stable hash** — hashing content rather than addresses: equal content
  hashes identically across processes (independent of ``PYTHONHASHSEED``),
  exactly like QE's ``FrozenMapping.__hash__``.
- **Deepcopy-safe** — :meth:`__getstate__` / :meth:`__setstate__` make the
  immutable wrapper picklable, which also un-blocks :mod:`copy.deepcopy`.
- It is **not** a ``collections.abc.Set`` of items and is deliberately small;
  it lives here so no giant ``common/utils`` module is introduced.
"""

from __future__ import annotations

import hashlib
import sys
from types import MappingProxyType
from typing import Mapping

__all__ = ["FrozenMapping", "freeze"]


class FrozenMapping(Mapping):
    """A recursively-immutable, hashable, deepcopy-safe read-only mapping.

    Wraps a plain dict in a :class:`types.MappingProxyType` so item assignment
    and deletion raise ``TypeError``.  Nested dicts / lists / sets are
    recursively frozen to nested :class:`FrozenMapping` / tuples / frozensets
    so no reachable value can be mutated through the artifact, and the mapping
    is hashable and deepcopy-safe.
    """

    __slots__ = ("_data",)

    def __init__(self, data: Mapping):
        if not isinstance(data, Mapping):
            raise TypeError(
                f"FrozenMapping requires a Mapping, got {type(data).__name__}"
            )
        # Freeze the VALUES recursively (nested dict -> FrozenMapping, list ->
        # tuple, ...).  The mapping itself is NOT passed back through
        # _freeze_value (that would re-wrap it in another FrozenMapping and
        # recurse forever); the top layer is the MappingProxyType below.
        frozen = {k: _freeze_value(v) for k, v in data.items()}
        object.__setattr__(self, "_data", MappingProxyType(frozen))

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __repr__(self) -> str:
        return f"FrozenMapping({dict(self._data)!r})"

    def __contains__(self, key):
        return key in self._data

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FrozenMapping):
            return dict(self._data) == dict(other._data)
        if isinstance(other, Mapping):
            return dict(self._data) == dict(other)
        return NotImplemented

    def __hash__(self) -> int:
        # Stable across processes (unlike builtin hash(), which is salted by
        # PYTHONHASHSEED).  Follows the QE FrozenMapping convention: derive
        # the int from a canonical content digest so equal content hashes
        # identically.
        digest = hashlib.sha256(
            canonical_bytes(dict(self._data))
        ).hexdigest()
        return int(digest, 16) % _HASH_MODULUS

    def __getstate__(self) -> dict:
        # Pickle as a plain dict; __setstate__ restores the read-only wrapper.
        # This also makes copy.deepcopy work (deepcopy pickles the instance).
        return dict(self._data)

    def __setstate__(self, state: dict) -> None:
        object.__setattr__(self, "_data", MappingProxyType(_freeze_value(state)))


_HASH_MODULUS = getattr(sys.hash_info, "modulus", 2**63 - 1)


def freeze(value: Mapping) -> FrozenMapping:
    """Deeply freeze any mapping into a :class:`FrozenMapping`.

    ``freeze`` is the entry point for artifacts that snapshot a caller
    mapping: it copies-and-freezes so mutating the caller's original dict
    after construction can never change the artifact.
    """
    if not isinstance(value, Mapping):
        raise TypeError(
            f"freeze() requires a Mapping, got {type(value).__name__}"
        )
    return FrozenMapping(value)


def _freeze_value(value: object) -> object:
    """Recursively freeze an arbitrary nested value into an immutable form.

    dict / Mapping  -> FrozenMapping
    list / tuple    -> tuple of frozen elements
    set / frozenset -> frozenset of frozen elements
    immutable scalars / None / bytes pass through unchanged
    anything else (a mutable object) raises ``TypeError`` fail-closed.
    """
    if isinstance(value, FrozenMapping):
        # Already-frozen wrapper: return as-is so re-freezing an artifact's
        # own frozen fields is idempotent (and cannot recurse).  Checked FIRST
        # because FrozenMapping is a Mapping subclass.
        return value
    if isinstance(value, Mapping):
        return FrozenMapping({k: _freeze_value(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_value(v) for v in value)
    if value is None or isinstance(
        value,
        (
            str,
            bytes,
            int,
            float,
            bool,
            complex,
            type(...),
        ),
    ):
        return value
    raise TypeError(
        f"_freeze_value: unsupported mutable value of type "
        f"{type(value).__name__} (fail-closed)"
    )


def canonical_bytes(value: object) -> bytes:
    """Canonical, order-independent, type-distinguishing byte encoding.

    Mirrors :mod:`factor_assets.contracts._canonical`.  Written locally so
    :class:`FrozenMapping` stays self-contained (no import cycle / layering
    surprises).
    """
    if isinstance(value, Mapping):
        children = sorted(
            canonical_bytes(k) + canonical_bytes(v) for k, v in value.items()
        )
        body = b"".join(children)
        return b"M" + str(len(body)).encode("ascii") + b":" + body
    if isinstance(value, FrozenMapping):
        return canonical_bytes(dict(value))
    if isinstance(value, (list, tuple)):
        body = b"".join(canonical_bytes(v) for v in value)
        tag = b"L" if isinstance(value, list) else b"T"
        return tag + str(len(body)).encode("ascii") + b":" + body
    if isinstance(value, (set, frozenset)):
        body = b"".join(sorted(canonical_bytes(v) for v in value))
        return b"E" + str(len(body)).encode("ascii") + b":" + body
    if value is None:
        return b"N1:\x00"
    if isinstance(value, bool):
        return b"B1:" + (b"1" if value else b"0")
    if isinstance(value, str):
        body = value.encode("utf-8")
        return b"S" + str(len(body)).encode("ascii") + b":" + body
    if isinstance(value, (int, float, complex)):
        body = repr(value).encode("utf-8")
        return b"N" + str(len(body)).encode("ascii") + b":" + body
    if isinstance(value, bytes):
        body = value
        return b"C" + str(len(body)).encode("ascii") + b":" + body
    raise TypeError(
        f"canonical_bytes: unsupported type {type(value).__name__} (fail-closed)"
    )