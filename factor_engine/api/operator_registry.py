"""Factor DSL allowlists with orthogonal authoring surfaces and dialects."""
from __future__ import annotations

from typing import Any, Callable

from factor_engine.api.columns import col, field
from factor_engine.backend.cleaned_bridge import build_cleaned_dsl_allowlist

STUB_IR_OPS: frozenset[str] = frozenset()


def build_dsl_allowlist(
    *,
    surface: str = "daily",
    dialect: str = "native",
    dialect_version: str | None = None,
) -> dict[str, Callable[..., Any]]:
    """Return a canonical surface mapping, optionally decorated by a dialect.

    This is the **authoring** surface: operators may be experimental because
    authoring writes formulas (which later fail closed at mining/production
    admission).  Mining tasks must call the dedicated entry points below.
    """
    raw_surface = str(surface or "daily")
    raw_dialect = str(dialect or "native").lower()
    if raw_surface == "lqtp":
        raw_surface = "compat_research"
        raw_dialect = "lqtp"

    allow: dict[str, Callable[..., Any]] = {"col": col, "field": field}
    if raw_surface == "extended":
        # Extended authoring is a strict superset of the daily primitive surface.
        # Composite technical macros lower into these daily primitives before IR.
        surfaces = ("daily", "extended")
    elif raw_surface == "compat":
        surfaces = ("daily", "extended")
    elif raw_surface == "compat_research":
        surfaces = ("daily", "extended", "research")
    elif raw_surface == "all":
        surfaces = ("all",)
    else:
        surfaces = (raw_surface,)
    for selected in surfaces:
        allow.update(build_cleaned_dsl_allowlist(set(), surface=selected))

    if raw_surface in {"extended", "compat", "compat_research", "all"}:
        from factor_engine.api.intraday_daily import INTRADAY_DAILY_DSL_FUNCTIONS
        from factor_engine.api.technical_macros_v2 import augment_technical_macros
        from factor_engine.storage.sources.intraday_clock_install import install_intraday_clock_runtime

        install_intraday_clock_runtime()
        allow.update(INTRADAY_DAILY_DSL_FUNCTIONS)
        # Public composite names lower to primitive Expr DAGs before Analyzer/IR.
        # The registry implementations remain semantic references only.
        allow = augment_technical_macros(allow)

    if raw_dialect in {"native", ""}:
        return allow
    if raw_dialect != "lqtp":
        raise ValueError(f"unsupported factor dialect: {dialect!r}")

    from factor_engine.api.lqtp_compat import DEFAULT_LQTP_DIALECT_VERSION, augment_dsl_allowlist

    version = str(dialect_version or DEFAULT_LQTP_DIALECT_VERSION)
    if version != DEFAULT_LQTP_DIALECT_VERSION:
        raise ValueError(
            f"unsupported LQTP dialect_version={version!r}; supported={DEFAULT_LQTP_DIALECT_VERSION!r}"
        )
    out = augment_dsl_allowlist(allow, surface=raw_surface)
    from factor_engine.api.lqtp_neutralization import augment_neutralization

    out = augment_neutralization(out)
    from factor_engine.api.lqtp_market import augment_market

    out = augment_market(out)
    from factor_engine.api.lqtp_functions_loader import augment_from_functions_yaml

    return augment_from_functions_yaml(out)


def build_authoring_allowlist(
    *,
    surface: str = "daily",
    dialect: str = "native",
    dialect_version: str | None = None,
) -> dict[str, Callable[..., Any]]:
    """Authoring surface: operators may be experimental (review §2.6).

    Used when writing formulas / recipes.  Admission to production mining is a
    separate, stricter gate (``build_production_mining_allowlist``).
    """
    return build_dsl_allowlist(
        surface=surface, dialect=dialect, dialect_version=dialect_version
    )


def build_research_mining_allowlist() -> dict[str, Callable[..., Any]]:
    """Research mining surface: any operator with a runtime is usable.

    Research experiments may reference experimental / research-registered
    operators; the results are explicitly risk-labelled and never admitted to
    production mining without six-gate certification (review §2.6).
    """
    from factor_engine.backend.cleaned_bridge import build_cleaned_dsl_allowlist

    return build_cleaned_dsl_allowlist(set(), surface="all")


def build_production_mining_allowlist() -> dict[str, Callable[..., Any]]:
    """Production mining surface: only six-gate certified operators.

    Every name passes ``build_operator_spec().allow_in_production``, which now
    requires daily/extended classification, ``production_certified is True``
    (six-gate), at least one evidence-backed production backend, and no
    compatibility/diagnostic/benchmark-only flag (review §2.6).  AlphaProbe /
    AlphaMiner / CogAlpha production tasks must call this entry point.
    """
    from factor_engine.backend.cleaned_bridge import build_production_dsl_allowlist

    return build_production_dsl_allowlist()
