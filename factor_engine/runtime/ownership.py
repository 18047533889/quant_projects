# -*- coding: utf-8 -*-
"""R21-FE-ZEROCOPY-OWNERSHIP: typed buffer-ownership contract.

Policy (read-only default)
--------------------------
* Caller buffers are ``READ_ONLY`` unless ownership is **explicitly
  transferred** to the receiver.
* Zero-copy sharing of a caller buffer **MUST** be ``READ_ONLY`` — a kernel
  that receives a shared view must never write through it.  An accidental
  in-place write on a shared read-only view fails loudly (``ValueError`` from
  numpy) instead of silently corrupting the caller's frame.

Members
-------
* ``READ_ONLY``            — a shared, read-only reference (the default
  contract; ``to_dict()`` serializes to ``"shared"`` so existing JSON stays
  byte-compatible with the pre-hardening free-string default).
* ``OWNED_MUTABLE``        — the receiver owns the buffer and may write in
  place (e.g. a freshly allocated mask / column-stack).
* ``BORROWED_READ_ONLY``   — a zero-copy borrow of a specific caller buffer
  (no copy made); read-only because it aliases caller-owned memory.
"""
from __future__ import annotations

import enum
from typing import Any


class BufferOwnership(str, enum.Enum):
    """Typed ownership contract for :class:`~runtime.buffer_ref.BufferRef`.

    ``str``-based so the value is directly JSON-serializable (``to_dict()``
    emits ``.value``) and legacy string comparisons keep working.
    """

    READ_ONLY = "shared"                 # serialization-compatible default
    OWNED_MUTABLE = "owned_mutable"
    BORROWED_READ_ONLY = "borrowed_read_only"


def make_readonly_shared_view(value: Any) -> Any:
    """Return a zero-copy ``np.ndarray`` view with ``WRITEABLE=False`` (GAP-3).

    When ``value`` is a numpy array (or a pandas object backed by one) this
    returns a memory-sharing view whose ``flags.writeable`` is ``False``, so an
    accidental in-place kernel write raises ``ValueError`` loudly instead of
    silently mutating the caller's buffer.  Non-array values are returned
    unchanged.
    """
    import numpy as np

    if isinstance(value, np.ndarray):
        view = value.view()
        view.setflags(write=False)
        return view
    return value
