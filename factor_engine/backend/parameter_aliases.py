"""Canonical parameter aliases normalized before IR construction."""
from __future__ import annotations

from typing import Any, Mapping


class ParameterAliasError(ValueError):
    """A legacy and canonical parameter conflict or reaches execution."""


_WINDOW_CANONICALS = frozenset({
    "ts_autocorr", "ts_beta", "ts_corr", "ts_cov", "ts_max", "ts_mean",
    "ts_median", "ts_min", "ts_rank", "ts_sharpe", "ts_std", "ts_sum",
    "ts_var", "ts_zscore",
})

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


def normalize_parameter_aliases(
    canonical: str,
    attrs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return canonical attrs and reject conflicting duplicate spellings."""
    normalized = dict(attrs)
    for alias, target in PARAMETER_ALIASES.get(canonical, {}).items():
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
    aliases = sorted(set(attrs).intersection(PARAMETER_ALIASES.get(canonical, {})))
    if aliases:
        raise ParameterAliasError(
            f"{canonical}: non-canonical runtime parameters {aliases!r}; normalize in Analyzer"
        )
