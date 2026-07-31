# -*- coding: utf-8 -*-
"""LQTP/JQ formula compatibility shell.

Compatibility is intentionally implemented above the canonical operator layer.
Different external spellings resolve to one canonical ``CleanedCall`` and
therefore one IR/DAG/runtime implementation.  A compatibility name never
creates a duplicate numerical kernel.

The module also performs one narrowly-scoped legacy template rewrite: a bare
``true_range`` token means ``true_range(high, low, close)``.  Calls already
written as ``true_range(...)`` are left untouched.
"""
from __future__ import annotations

import io
import token
import tokenize
from typing import Any, Callable


class LQTPCompatibilityError(ValueError):
    """Base error for an LQTP compatibility rule that cannot be normalized."""


class LQTPDataDependencyError(LQTPCompatibilityError):
    """Formula requires a logical source/derived field not available as a bar column."""


def _factory(canonical: str) -> Callable[..., Any]:
    from api.cleaned_ops import make_cleaned_call_factory

    return make_cleaned_call_factory(canonical)


def _sma_dispatch(*args: Any, **kwargs: Any):
    """Dispatch LQTP/JQ ``sma`` by arity without conflating two semantics."""
    if kwargs:
        # Named forms are kept explicit to avoid silently guessing whether n/m
        # were intended as a simple or recursive average.
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


# Exact spelling/semantic compatibility.  Every value is a canonical runtime;
# no duplicated operator implementation is created here.
_EXACT_COMPAT: dict[str, str] = {
    "decay_linear": "ts_decay_linear",
    "safe_log": "safe_log_null",
    "ts_rank_pct": "ts_rank",
    "ts_ewm_mean": "ts_ema",
    "ts_regression_slope_sequence": "ts_time_slope",
    "ts_expanding_rank": "expanding_rank",
    "ts_hump_decay": "hump_decay",
    "fp_beta": "rolling_beta_to_market",
}


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

    for external, canonical in _EXACT_COMPAT.items():
        if OperatorRegistry.get(canonical) is not None:
            out.setdefault(external, _factory(canonical))

    # Make the canonical itself and all known aliases visible for the LQTP
    # corpus even when the operator lifecycle is extended/research.  This is a
    # parse capability only; production mode independently rejects unapproved
    # operators at the plan gate.
    for canonical in sorted(LQTP_COMPAT_PARSE_CANONICALS):
        if OperatorRegistry.get(canonical) is None:
            continue
        out.setdefault(canonical, _factory(canonical))
        for alias, target in OperatorRegistry._aliases.items():
            if target == canonical:
                out.setdefault(alias, _factory(canonical))
    return out


def normalize_lqtp_formula(text: str) -> str:
    """Normalize legacy LQTP formula tokens before the restricted AST parser."""
    source = str(text or "")
    if not source.strip():
        return source

    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    rewritten: list[tokenize.TokenInfo] = []
    i = 0
    while i < len(tokens):
        current = tokens[i]
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
                replacement = tokenize.generate_tokens(
                    io.StringIO("true_range(high, low, close)").readline
                )
                repl = [t for t in replacement if t.type not in {tokenize.ENDMARKER}]
                rewritten.extend(repl)
                i += 1
                continue
        rewritten.append(current)
        i += 1
    try:
        return tokenize.untokenize(rewritten)
    except Exception as exc:  # pragma: no cover - tokenizer defensive path
        raise LQTPCompatibilityError(f"failed to normalize LQTP formula: {source}") from exc
