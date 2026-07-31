"""Factor DSL allowlists with an explicit compatibility shell.

Canonical operator semantics remain in ``cleaned_operators``.  External LQTP /
JQ spellings are added only as factories that resolve to those canonicals, so
aliases never create duplicate numerical kernels.
"""
from __future__ import annotations

from typing import Any, Callable

from api.columns import col
from backend.cleaned_bridge import build_cleaned_dsl_allowlist

STUB_IR_OPS: frozenset[str] = frozenset()


def build_dsl_allowlist(*, surface: str = "daily") -> dict[str, Callable[..., Any]]:
    """Return a surface-scoped DSL mapping.

    ``compat`` combines daily + extended authoring.  In addition, the LQTP
    compatibility shell exposes reviewed historical spellings and research
    factor calls for parsing.  This affects parsing only; production admission
    is still enforced on the canonical plan by ``runtime.production_policy``.
    """
    allow: dict[str, Callable[..., Any]] = {"col": col}
    surfaces = ("daily", "extended") if surface == "compat" else (surface,)
    for selected in surfaces:
        allow.update(build_cleaned_dsl_allowlist(set(), surface=selected))

    from api.lqtp_compat import augment_dsl_allowlist

    return augment_dsl_allowlist(allow, surface=surface)
