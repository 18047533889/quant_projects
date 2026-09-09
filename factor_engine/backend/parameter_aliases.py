"""Canonical parameter aliases normalized before IR construction.

R19-006 (parameter alias SINGLE authority):

* The ONLY authoritative source of parameter aliases is each operator's
  ``OperatorMetadata.param_aliases``.
* The planning-time global alias map (``PARAMETER_ALIASES``) is the legacy
  COMPAT FALLBACK: it is used only when an operator's metadata either is not
  registered yet (bootstrap/planning before ``load_all``) or declares NO
  ``param_aliases``.  Every compat entry that is not backed by metadata must be
  listed in ``_COMPAT_ONLY_ALIAS_CANONICALS`` (the explicit marker) — a diff the
  ``assert_global_alias_map_consistent()`` CI gate tolerates ONLY for those
  marked canonicals.
* ``derive_alias_map_from_metadata()`` builds the authoritative map by walking
  the loaded registry, and ``global_alias_map()`` (``GLOBAL_ALIAS_MAP``) is the
  production-facing effective map = metadata-derived + explicitly-marked compat
  fallback.  ``assert_global_alias_map_consistent()`` enforces
  ``GLOBAL_ALIAS_MAP == DERIVED_METADATA_ALIAS_MAP`` modulo the compat marker.
"""
from __future__ import annotations

from typing import Any, Mapping


class ParameterAliasError(ValueError):
    """A legacy and canonical parameter conflict or reaches execution."""


_WINDOW_CANONICALS = frozenset({
    "ts_autocorr", "ts_beta", "ts_corr", "ts_cov", "ts_max", "ts_mean",
    "ts_median", "ts_min", "ts_rank", "ts_sharpe", "ts_std", "ts_sum",
    "ts_var", "ts_zscore",
})

# R19-006: planning-time COMPAT FALLBACK global alias map.  It must never be the
# production authority — ``OperatorMetadata.param_aliases`` is.  Entries whose
# canonical does NOT declare the alias in metadata are compat-only and MUST stay
# listed in ``_COMPAT_ONLY_ALIAS_CANONICALS``.
PARAMETER_ALIASES: dict[str, dict[str, str]] = {
    **{name: {"d": "window"} for name in _WINDOW_CANONICALS},
    "ts_delay": {"d": "n", "window": "n", "lag": "n", "periods": "n"},
    "ts_delta": {"d": "n", "window": "n", "lag": "n", "periods": "n"},
    "ts_log_return": {"periods": "d", "window": "d", "lag": "d"},
    "ts_pct": {"periods": "d", "window": "d", "lag": "d", "n": "d"},
    "lerp": {"f": "fraction"},
    "round": {"d": "decimals", "k": "decimals"},
    "truncate": {"d": "decimals", "k": "decimals"},
}

# R19-006: explicit marker for planning-layer COMPAT-ONLY alias entries whose
# canonical does NOT (yet) declare ``param_aliases`` in metadata.  The CI gate
# re-derives this set from the loaded registry and FAILS on drift (an entry that
# gains metadata backing must be removed here, or the assertion breaks).
#
# ``ts_beta``, ``ts_median``, and ``ts_zscore`` gained metadata
# ``param_aliases={"d": "window"}``, so they are no longer compat-only; their
# PARAMETER_ALIASES entries stay as planning-before-load_all fallbacks, but the
# compat marker excludes them.  ``ts_rank`` remains compat-only (its polars native
# contract explicitly rejects ``d`` as a runtime alias — see
# test_rolling_parameter_contracts).
_COMPAT_ONLY_ALIAS_CANONICALS: frozenset[str] = (
    frozenset(PARAMETER_ALIASES.keys())
    - frozenset({"ts_beta", "ts_median", "ts_zscore"})
)


def _metadata_aliases_for(canonical: str) -> dict[str, str] | None:
    """Return the operator's ``metadata.param_aliases``, or ``None`` when the
    registry is not loaded (bootstrap/planning-before-load_all) so the caller
    falls back to the compat map.

    Returns ``{}`` when the operator is registered but its metadata declares NO
    ``param_aliases`` (metadata authority: no aliases -> compat fallback still
    applies for legacy support).
    """
    try:
        from factor_engine.cleaned_operators.registry import OperatorRegistry
    except Exception:  # pragma: no cover - import fallback
        return None
    try:
        operators = OperatorRegistry._operators
        if not operators:
            return None  # registry not loaded -> compat fallback
        implementations = operators.get(canonical)
        if not implementations:
            return None  # not registered -> compat fallback
        merged: dict[str, str] = {}
        for op in implementations.values():
            meta = getattr(op, "metadata", None)
            aliases = getattr(meta, "param_aliases", None)
            if aliases:
                merged.update(aliases)
        return merged
    except Exception:  # pragma: no cover - defensive
        return None


def derive_alias_map_from_metadata() -> dict[str, dict[str, str]]:
    """R19-006: build the authoritative ``{canonical: aliases}`` map by walking
    every registered operator's ``OperatorMetadata.param_aliases``.

    Returns ``{}`` when the registry is not loaded (bootstrap/compiler tooling
    before ``load_all``).  Only canonicals with a NON-EMPTY ``param_aliases``
    appear — an operator that declares no aliases is not an alias authority.
    """
    try:
        from factor_engine.cleaned_operators.registry import OperatorRegistry
    except Exception:  # pragma: no cover - import fallback
        return {}
    try:
        operators = OperatorRegistry._operators
    except Exception:  # pragma: no cover - defensive
        return {}
    out: dict[str, dict[str, str]] = {}
    for canonical, implementations in (operators or {}).items():
        merged: dict[str, str] = {}
        for op in implementations.values():
            meta = getattr(op, "metadata", None)
            aliases = getattr(meta, "param_aliases", None)
            if aliases:
                merged.update(aliases)
        if merged:
            out[canonical] = merged
    return out


def _effective_alias_map(canonical: str) -> dict[str, str]:
    """R19-006: the aliases to apply for ``canonical`` — metadata is the single
    authority when it declares aliases; otherwise the explicitly-marked compat
    fallback applies."""
    derived = _metadata_aliases_for(canonical)
    if derived:
        return derived  # metadata authority
    return PARAMETER_ALIASES.get(canonical, {})  # compat fallback


def global_alias_map() -> dict[str, dict[str, str]]:
    """R19-006: the production-facing GLOBAL alias map.

    Metadata-derived aliases (the authority) merged with the explicitly-marked
    compat-only fallback entries for operators whose metadata does not declare
    ``param_aliases``.
    """
    derived = derive_alias_map_from_metadata()
    out: dict[str, dict[str, str]] = {k: dict(v) for k, v in derived.items()}
    for canonical, aliases in PARAMETER_ALIASES.items():
        if canonical in out and out[canonical]:
            continue  # metadata authority
        out.setdefault(canonical, dict(aliases))
    return out


def assert_global_alias_map_consistent() -> None:
    """R19-006 CI gate: ``GLOBAL_ALIAS_MAP == DERIVED_METADATA_ALIAS_MAP`` modulo
    the explicitly-marked compat-only fallback.

    Asserts three invariants:
      1. every metadata-declared alias is the single authority — the hand-written
         compat map must not conflict with it;
      2. every compat-map entry not backed by metadata is EXPLICITLY marked in
         ``_COMPAT_ONLY_ALIAS_CANONICALS`` (and the marker set is exactly the
         actual metadata-backed gap);
      3. the effective global map exposes every metadata-derived alias.
    """
    derived = derive_alias_map_from_metadata()
    # 1. Metadata is the authority: no conflict with the compat map.
    for canonical, meta_aliases in derived.items():
        compat = PARAMETER_ALIASES.get(canonical)
        if compat is not None and compat != meta_aliases:
            raise AssertionError(
                f"R19-006: {canonical}: PARAMETER_ALIASES {compat} conflicts with "
                f"metadata.param_aliases {meta_aliases}; metadata is the single "
                "alias authority"
            )
    # 2. Every compat entry must be explicitly marked (and the marker exactly
    #    matches the metadata-backed gap).
    actual_compat = frozenset(
        c for c in PARAMETER_ALIASES if not derived.get(c)
    )
    if actual_compat != _COMPAT_ONLY_ALIAS_CANONICALS:
        raise AssertionError(
            f"R19-006: compat-only alias marker drift: declared "
            f"{sorted(_COMPAT_ONLY_ALIAS_CANONICALS)} but metadata-backed gap is "
            f"{sorted(actual_compat)}; update _COMPAT_ONLY_ALIAS_CANONICALS"
        )
    # 3. GLOBAL_ALIAS_MAP == DERIVED_METADATA_ALIAS_MAP for every metadata alias.
    global_map = global_alias_map()
    for canonical, aliases in derived.items():
        if global_map.get(canonical) != aliases:
            raise AssertionError(
                f"R19-006: {canonical}: global alias map {global_map.get(canonical)} "
                f"!= metadata-derived {aliases}"
            )


def normalize_parameter_aliases(
    canonical: str,
    attrs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return canonical attrs and reject conflicting duplicate spellings.

    R19-006: the alias map is resolved from ``OperatorMetadata.param_aliases``
    (single authority) with an explicit compat fallback to ``PARAMETER_ALIASES``
    only when metadata declares nothing / is not loaded.
    """
    normalized = dict(attrs)
    for alias, target in _effective_alias_map(canonical).items():
        if alias not in normalized:
            continue
        value = normalized.pop(alias)
        if target in normalized and normalized[target] != value:
            raise ParameterAliasError(
                f"{canonical}: conflicting parameters {alias}={value!r} and "
                f"{target}={normalized[target]!r}"
            )
        normalized[target] = value
    return normalized


def reject_runtime_parameter_aliases(canonical: str, attrs: Mapping[str, Any]) -> None:
    """Execution layers accept canonical parameters only; Analyzer owns aliases."""
    aliases = sorted(set(attrs).intersection(_effective_alias_map(canonical)))
    if aliases:
        raise ParameterAliasError(
            f"{canonical}: non-canonical runtime parameters {aliases!r}; normalize in Analyzer"
        )


# R19-006: the two sides of the CI equality assertion.  They are functions
# because they require the registry to be loaded; call them after ``load_all``.
DERIVED_METADATA_ALIAS_MAP = derive_alias_map_from_metadata
GLOBAL_ALIAS_MAP = global_alias_map
