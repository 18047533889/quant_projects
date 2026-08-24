# -*- coding: utf-8 -*-
"""Safe external LQTP functions.yaml alias/template loader.

This is deliberately not Python eval. Templates are restricted expression ASTs
whose calls must already exist in the current LQTP allowlist.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any, Callable

import yaml


class LQTPFunctionsConfigError(ValueError):
    pass


def _config_path() -> Path | None:
    raw = os.environ.get("FACTOR_ENGINE_LQTP_FUNCTIONS_YAML", "").strip()
    return Path(raw).expanduser().resolve() if raw else None


def _load_payload(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise LQTPFunctionsConfigError(f"functions.yaml must be a mapping: {path}")
    return payload


def _factory(canonical: str):
    from factor_engine.api.cleaned_ops import make_cleaned_call_factory
    return make_cleaned_call_factory(canonical)


def _template_callable(
    name: str,
    params: list[str],
    expression: str,
    base_allow: dict[str, Callable[..., Any]],
) -> Callable[..., Any]:
    try:
        tree = ast.parse(str(expression), mode="eval")
    except SyntaxError as exc:
        raise LQTPFunctionsConfigError(f"invalid template {name!r}: {expression}") from exc

    def evaluate(node: ast.AST, bound: dict[str, Any]) -> Any:
        if isinstance(node, ast.Name):
            if node.id in bound:
                return bound[node.id]
            raise LQTPFunctionsConfigError(
                f"template {name!r} contains unbound bare name {node.id!r}; "
                "all data inputs must be declared parameters"
            )
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (str, int, float, bool)) or node.value is None:
                return node.value
            raise LQTPFunctionsConfigError(f"template {name!r} uses unsupported literal")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn_name = node.func.id
            fn = base_allow.get(fn_name)
            if fn is None:
                raise LQTPFunctionsConfigError(
                    f"template {name!r} calls unregistered function {fn_name!r}"
                )
            args = [evaluate(arg, bound) for arg in node.args]
            kwargs = {
                kw.arg: evaluate(kw.value, bound)
                for kw in node.keywords
                if kw.arg is not None
            }
            if any(kw.arg is None for kw in node.keywords):
                raise LQTPFunctionsConfigError("template **kwargs are forbidden")
            return fn(*args, **kwargs)
        if isinstance(node, ast.BinOp):
            left, right = evaluate(node.left, bound), evaluate(node.right, bound)
            if isinstance(node.op, ast.Add): return left + right
            if isinstance(node.op, ast.Sub): return left - right
            if isinstance(node.op, ast.Mult): return left * right
            if isinstance(node.op, ast.Div): return left / right
            if isinstance(node.op, ast.Pow): return _factory("power")(left, right)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -evaluate(node.operand, bound)
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1:
            left, right = evaluate(node.left, bound), evaluate(node.comparators[0], bound)
            mapping = {
                ast.Lt: "lt", ast.LtE: "le", ast.Eq: "eq",
                ast.Gt: "gt", ast.GtE: "ge", ast.NotEq: "ne",
            }
            for kind, canonical in mapping.items():
                if isinstance(node.ops[0], kind):
                    return _factory(canonical)(left, right)
        raise LQTPFunctionsConfigError(
            f"template {name!r} contains forbidden syntax {type(node).__name__}"
        )

    def call(*args: Any, **kwargs: Any):
        if len(args) > len(params):
            raise ValueError(f"{name} expected at most {len(params)} positional arguments")
        bound = dict(zip(params, args))
        for key, value in kwargs.items():
            if key not in params:
                raise ValueError(f"{name} got unknown parameter {key!r}")
            if key in bound:
                raise ValueError(f"{name} got duplicate parameter {key!r}")
            bound[key] = value
        missing = [param for param in params if param not in bound]
        if missing:
            raise ValueError(f"{name} missing parameters {missing}")
        return evaluate(tree.body, bound)

    call.__name__ = name
    return call


def _entries(payload: dict[str, Any]) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    aliases: dict[str, str] = {}
    templates: dict[str, dict[str, Any]] = {}

    raw_aliases = payload.get("aliases", {})
    if raw_aliases:
        if not isinstance(raw_aliases, dict):
            raise LQTPFunctionsConfigError("aliases must be a mapping")
        aliases.update({str(k): str(v) for k, v in raw_aliases.items()})

    raw_templates = payload.get("templates", {})
    if raw_templates:
        if not isinstance(raw_templates, dict):
            raise LQTPFunctionsConfigError("templates must be a mapping")
        for name, spec in raw_templates.items():
            if not isinstance(spec, dict):
                raise LQTPFunctionsConfigError(f"template {name!r} must be a mapping")
            templates[str(name)] = dict(spec)

    raw_functions = payload.get("functions", {})
    if raw_functions:
        if not isinstance(raw_functions, dict):
            raise LQTPFunctionsConfigError("functions must be a mapping")
        for name, spec in raw_functions.items():
            if isinstance(spec, str):
                aliases[str(name)] = spec
            elif isinstance(spec, dict):
                if "alias" in spec or "canonical" in spec:
                    aliases[str(name)] = str(spec.get("alias") or spec.get("canonical"))
                elif "expression" in spec:
                    templates[str(name)] = dict(spec)
                else:
                    raise LQTPFunctionsConfigError(
                        f"function {name!r} requires alias/canonical or expression"
                    )
            else:
                raise LQTPFunctionsConfigError(f"unsupported function entry {name!r}")
    return aliases, templates


def augment_from_functions_yaml(
    allow: dict[str, Callable[..., Any]],
    *,
    path: str | Path | None = None,
) -> dict[str, Callable[..., Any]]:
    resolved = Path(path).expanduser().resolve() if path is not None else _config_path()
    if resolved is None:
        return allow
    if not resolved.is_file():
        raise LQTPFunctionsConfigError(f"functions.yaml not found: {resolved}")
    payload = _load_payload(resolved)
    aliases, templates = _entries(payload)
    out = dict(allow)

    from factor_engine.cleaned_operators.registry import OperatorRegistry
    for external, canonical_or_alias in aliases.items():
        try:
            canonical = OperatorRegistry.resolve_canonical_strict(canonical_or_alias)
        except Exception as exc:
            raise LQTPFunctionsConfigError(
                f"external alias {external!r} targets unknown canonical {canonical_or_alias!r}"
            ) from exc
        out[external] = _factory(canonical)

    base_allow = dict(out)
    for name, spec in templates.items():
        params = spec.get("params", spec.get("parameters", []))
        expression = spec.get("expression")
        if not isinstance(params, list) or not all(isinstance(x, str) for x in params):
            raise LQTPFunctionsConfigError(f"template {name!r} params must be a string list")
        if not isinstance(expression, str) or not expression.strip():
            raise LQTPFunctionsConfigError(f"template {name!r} expression must be non-empty")
        out[name] = _template_callable(name, list(params), expression, base_allow)
    return out
