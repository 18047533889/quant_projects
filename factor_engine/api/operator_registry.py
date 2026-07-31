"""Factor DSL allowlists with an explicit compatibility shell.

Canonical operator semantics remain in ``cleaned_operators``. External LQTP/JQ
spellings are factories that resolve to those canonicals, never duplicate
numerical kernels.
"""
from __future__ import annotations

from typing import Any, Callable

from api.columns import col
from backend.cleaned_bridge import build_cleaned_dsl_allowlist

STUB_IR_OPS: frozenset[str] = frozenset()


def build_dsl_allowlist(*, surface: str = "daily") -> dict[str, Callable[..., Any]]:
    """Return a surface-scoped DSL mapping.

    ``compat`` combines daily + extended. ``lqtp`` starts from daily + extended
    + factor-shaped research visibility and then applies versioned compatibility
    spellings. Production admission remains a separate canonical-plan gate.
    """
    allow: dict[str, Callable[..., Any]] = {"col": col}
    if surface == "compat":
        surfaces = ("daily", "extended")
    elif surface == "lqtp":
        surfaces = ("daily", "extended", "research")
    else:
        surfaces = (surface,)
    for selected in surfaces:
        allow.update(build_cleaned_dsl_allowlist(set(), surface=selected))

    from api.lqtp_compat import augment_dsl_allowlist
    return augment_dsl_allowlist(allow, surface=surface)
