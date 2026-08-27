"""Small package-local canonical structural encoding helper.

A deterministic, type-distinguishing, order-independent byte encoding used to
hash nested artifact fields.  It is deliberately *not* ``str(value)``: two
structurally equal nested objects encode identically regardless of mapping
insertion order or ``str(dict)`` repr quirks, and equal-looking values of
different types (``1`` vs ``1.0`` vs ``"1"``; dict vs list vs tuple vs set)
encode differently.

This is the single canonical encoder for FA artifacts that need a typed
structural hash (e.g. :class:`~factor_assets.clustering.families.ClusterArtifact`
and the cluster-governance artifacts).  It lives here so no giant
``common/utils`` module is introduced.
"""

from __future__ import annotations

from typing import Mapping

__all__ = ["canonical_bytes", "canonical_digest"]


def canonical_bytes(value: object) -> bytes:
    """Canonical structural byte encoding of a nested object.

    Guarantees:
      - Structurally equal nested objects encode identically regardless of
        mapping insertion order or ``str(dict)`` repr quirks.
      - Types are distinguished (``1`` vs ``1.0`` vs ``"1"``; dict vs list vs
        tuple vs set), so equal-looking values of different types differ.
      - Every atomic and aggregate is length-prefixed, so delimiters cannot
        collide and no two distinct structures share a prefix/suffix encoding.
      - Containers are recursively canonicalized; no depth guard is needed for
        the bounded, validated artifact fields this module hashes.
    """
    if value is None:
        return b"N1:\x00"
    if isinstance(value, bool):
        return b"B1:" + (b"1" if value else b"0")
    if isinstance(value, str):
        body = value.encode("utf-8")
        return b"S" + str(len(body)).encode("ascii") + b":" + body
    if isinstance(value, int):
        body = str(value).encode("ascii")
        return b"I" + str(len(body)).encode("ascii") + b":" + body
    if isinstance(value, float):
        body = repr(value).encode("ascii")
        return b"F" + str(len(body)).encode("ascii") + b":" + body
    if isinstance(value, Mapping):
        children = sorted(canonical_bytes(k) + canonical_bytes(v) for k, v in value.items())
        body = b"".join(children)
        return b"M" + str(len(body)).encode("ascii") + b":" + body
    if isinstance(value, (list, tuple)):
        body = b"".join(canonical_bytes(v) for v in value)
        tag = b"L" if isinstance(value, list) else b"T"
        return tag + str(len(body)).encode("ascii") + b":" + body
    if isinstance(value, (set, frozenset)):
        body = b"".join(sorted(canonical_bytes(v) for v in value))
        return b"E" + str(len(body)).encode("ascii") + b":" + body
    raise TypeError(
        f"cannot canonicalize value of unsupported type {type(value).__name__}"
    )


def canonical_digest(*fields: object) -> str:
    """sha256 over the canonical encodings of the given fields.

    Each field is length-prefixed so two adjacent fields cannot collide with a
    single concatenated field.  Returns a hex digest string.
    """
    import hashlib

    digest = hashlib.sha256()
    for field in fields:
        body = canonical_bytes(field)
        digest.update(str(len(body)).encode("ascii"))
        digest.update(b":")
        digest.update(body)
    return digest.hexdigest()
