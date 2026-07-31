# -*- coding: utf-8 -*-
"""LQTP/JQ formula compatibility shell.

External spellings resolve to canonical ``CleanedCall`` nodes; no compatibility
name owns a second numerical kernel. Rules that are not proven equivalent are
kept explicit instead of being guessed.
"""
from __future__ import annotations

import io
import token
import tokenize
from typing import Any, Callable


class LQTPCompatibilityError(ValueError):
    """A compatibility rule cannot be normalized without changing semantics."""


class LQTPDataDependencyError(LQTPCompatibilityError):
    """Formula requires a logical source/derived field, not a daily-bar column."""


def _factory(canonical: str) -> Callable[..., Any]:
    from api.cleaned_ops import make_cleaned_call_factory
    return make_cleaned_call_factory(canonical)


def _sma_dispatch(*args: Any, **kwargs: Any):
    """Dispatch simple two-argument SMA and Chinese/JQ recursive three-argument SMA."""
    if kwargs:
        if set(kwargs) <= {"window"} and len(args) == 1:
            return _factory("ts_mean")(*args, **kwargs)
        if set(kwargs) <= {"n", "m"} and len(args) == 1 and {"n", "m"} <= set(kwargs):
            return _factory("ts_sma_cn")(*args, **kwargs)
        raise LQTPCompatibilityError(
            "sma accepts sma(x, window) or Chinese/JQ sma(x, n, m)"
        )
    if len(args) == 2:
        return _factory("ts_mean")(*args)
    if len(args) == 3:
        return _factory("ts_sma_cn")(*args)
    raise LQTPCompatibilityError(
        "sma accepts exactly 2 arguments for simple mean or 3 for recursive SMA"
    )


def _safe_log_dispatch(x: Any):
    """LQTP safe_log macro using existing canonical primitives only."""
    positive = _factory("gt")(x, 0.0)
    logged = _factory("log")(x)
    zero = _factory("subtract")(x, x)
    null_value = _factory("safe_div_null")(zero, zero)
    return _factory("where")(positive, logged, null_value)


def _momentum_dispatch(x: Any, window: Any):
    """Expand the repository's canonical momentum recipe to ``ts_delta``."""
    return _factory("ts_delta")(x, window)


def _real_turnover_rate_dispatch(*args: Any):
    """Normalize the LQTP zero-argument turnover helper to explicit inputs.

    LQTP defines ``real_turnover_rate()`` as DailyBar.Volume divided by the
    configured TurnoverBaseDaily value.  FactorEngine's canonical operator is a
    pure two-input kernel, so the compatibility shell injects logical columns
    instead of giving the kernel hidden access to a data source.
    """
    if len(args) == 0:
        from api.columns import col
        return _factory("real_turnover_rate")(col("volume"), col("turnover_base"))
    if len(args) == 2:
        return _factory("real_turnover_rate")(*args)
    raise LQTPCompatibilityError(
        "real_turnover_rate accepts no arguments (LQTP source-aware form) or "
        "explicit (volume, turnover_base) inputs"
    )


def _minute_bar_dispatch(*args: Any, **kwargs: Any):
    """Fail early until a minute-source/resampling plan is configured.

    The supplied LQTP contract defines minute_bar(field, period, index) over a
    minute sequence. A daily panel cannot represent that nested frequency
    correctly, so silently mapping it to a daily rolling operator would be a
    semantic error.
    """
    raise LQTPDataDependencyError(
        "minute_bar(field, period, index) requires a configured minute-bar source "
        "and minute resampling plan; it cannot run against ashare_stock_daily"
    )


# Only source-supported equivalences belong here.
_EXACT_COMPAT: dict[str, str] = {
    "decay_linear": "ts_decay_linear",
    "ts_rank_pct": "ts_rank",
    "ts_ewm_mean": "ts_ema",
    "ts_expanding_rank": "expanding_rank",
    "ts_hump_decay": "hump_decay",
    "fp_beta": "rolling_beta_to_market",
}

# The supplied LQTP material names these functions but does not define their
# exact semantics. Rejecting them explicitly is safer than silently mapping to
# a superficially similar operator and materializing wrong factor values.
_AMBIGUOUS_EXTERNAL_NAMES: frozenset[str] = frozenset({
    "ts_regression_slope_sequence",
    "ts_sumac",
})


def _ambiguous_dispatch(name: str) -> Callable[..., Any]:
    def _raise(*args: Any, **kwargs: Any):
        raise LQTPCompatibilityError(
            f"{name} is present in the LQTP corpus but its exact semantic definition "
            "is not supplied; add a versioned dialect rule before execution"
        )
    _raise.__name__ = name
    return _raise


def augment_dsl_allowlist(
    allow: dict[str, Callable[..., Any]],
    *,
    surface: str,
) -> dict[str, Callable[..., Any]]:
    """Add compatibility names without changing canonical production policy."""
    if surface not in {"daily", "compat", "research", "all", "lqtp"}:
        return allow

    from cleaned_operators.production_tiers import LQTP_COMPAT_PARSE_CANONICALS
    from cleaned_operators.registry import OperatorRegistry

    out = dict(allow)
    out.setdefault("sma", _sma_dispatch)
    out.setdefault("safe_log", _safe_log_dispatch)
    out.setdefault("momentum", _momentum_dispatch)
    out["real_turnover_rate"] = _real_turnover_rate_dispatch
    out.setdefault("minute_bar", _minute_bar_dispatch)

    for external, canonical in _EXACT_COMPAT.items():
        if OperatorRegistry.get(canonical) is not None:
            out.setdefault(external, _factory(canonical))
    for name in _AMBIGUOUS_EXTERNAL_NAMES:
        out.setdefault(name, _ambiguous_dispatch(name))

    for canonical in sorted(LQTP_COMPAT_PARSE_CANONICALS):
        if OperatorRegistry.get(canonical) is None:
            continue
        out.setdefault(canonical, _factory(canonical))
        for alias, target in OperatorRegistry._aliases.items():
            if target == canonical:
                out.setdefault(alias, _factory(canonical))
    return out


def normalize_lqtp_formula(text: str) -> str:
    """Normalize only source-proven legacy templates before restricted AST parsing."""
    source = str(text or "")
    if not source.strip():
        return source

    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    pairs: list[tuple[int, str]] = []
    for i, current in enumerate(tokens):
        if current.type == token.NAME and current.string == "true_range":
            j = i + 1
            while j < len(tokens) and tokens[j].type in {
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
            }:
                j += 1
            is_call = j < len(tokens) and tokens[j].string == "("
            if not is_call:
                pairs.extend([
                    (token.NAME, "true_range"),
                    (token.OP, "("),
                    (token.NAME, "high"),
                    (token.OP, ","),
                    (token.NAME, "low"),
                    (token.OP, ","),
                    (token.NAME, "close"),
                    (token.OP, ")"),
                ])
                continue
        if current.type != tokenize.ENDMARKER:
            pairs.append((current.type, current.string))
    try:
        return tokenize.untokenize(pairs)
    except Exception as exc:  # pragma: no cover
        raise LQTPCompatibilityError(f"failed to normalize LQTP formula: {source}") from exc
