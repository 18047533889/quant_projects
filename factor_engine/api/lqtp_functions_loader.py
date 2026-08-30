# -*- coding: utf-8 -*-
"""Safe external LQTP functions.yaml alias/template loader.

This is deliberately not Python eval. Templates are restricted expression ASTs
whose calls must already exist in the current LQTP allowlist.
"""
from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Any, Callable

import yaml


class LQTPFunctionsConfigError(ValueError):
    pass


def _config_path() -> Path | None:
    raw = os.environ.get("FACTOR_ENGINE_LQTP_FUNCTIONS_YAML", "").strip()
    return Path(raw).expanduser().resolve() if raw else None


def _looks_like_operator_name(value: str) -> bool:
    """Platform aliases that are FE operator names resolve to a canonical.

    Everything else (a DSL expression like ``rank(daily_return)``, a
    ``field ${...}`` template, a ``DataTable.Field`` path) is platform
    authoring, not an executable operator — skip at load so an unchanged
    platform functions.yaml stays loadable.
    """
    s = str(value).strip()
    if not s:
        return False
    if any(ch in s for ch in "().${}[]=,+-*/<>!&|"):
        return False
    return all(ch.isalnum() or ch == "_" for ch in s)


def _load_payload(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise LQTPFunctionsConfigError(f"functions.yaml must be a mapping: {path}")
    return payload


def _factory(canonical: str):
    from factor_engine.api.cleaned_ops import make_cleaned_call_factory
    return make_cleaned_call_factory(canonical)


def _call_order(expr: str) -> list[str]:
    """Ordered bare-name (Call/Name) appearances in a template body.

    Walks the AST in source order and collects ``ast.Name`` identifiers that
    are called (``ts_mean(...)``) or referenced (``high``) so the inferred
    positional parameter list matches the template body's data-input order.
    """
    import ast as _ast

    tree = _ast.parse(str(expr), mode="eval")
    order: list[str] = []

    def walk(node: _ast.AST) -> None:
        if isinstance(node, _ast.Name):
            if node.id not in order:
                order.append(node.id)
            return
        for child in _ast.iter_child_nodes(node):
            walk(child)

    walk(tree.body)
    return order


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
                sql_expr = spec.get("sql_expr")
                if sql_expr:
                    # ``sql_expr`` entries (safe_div / nullif_zero / safe_log /
                    # clean) are SQL rendered at platform query time, not
                    # executable FE kernels — none of their call sites currently
                    # thread the platform SQL through FactorEngine.  Skip them
                    # (never eval) so an unchanged platform functions.yaml loads;
                    # unsupported use fails later at parse/allowlist with the
                    # same visibility as any unsupported name.
                    continue
                if "alias" in spec or "canonical" in spec:
                    aliases[str(name)] = str(spec.get("alias") or spec.get("canonical"))
                elif "expression" in spec:
                    templates[str(name)] = dict(spec)
                elif "parameterized_source" in spec:
                    # Platform source-entry (e.g. benchmark_index with a
                    # parameterized_source + params) is a data-source binding,
                    # not an FE operator alias/template — skip so an unchanged
                    # platform functions.yaml stays loadable.
                    continue
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
        # ``${...}``-parameterized template bodies (ma / std_n / delta /
        # atr / volume_ratio / ... = ``ts_mean(${field}, ${window})``) are
        # parameterized templates, not executable alias strings.  They are
        # registered as templates below (reuse of the ``templates`` machinery
        # keyed by the parameter list derived from the body).
        if "${" in canonical_or_alias:
            if _config_path() is None and path is None:
                raise LQTPFunctionsConfigError(
                    f"external alias {external!r} targets unknown canonical "
                    f"{canonical_or_alias!r}"
                )
            continue
        # ``sql_expr``-style field aliases (open/high/low/close/... + Field)
        # define the platform's LQTP *adjustment basis*, which FactorEngine
        # already applies at the StockDailyBar source layer — the canonical
        # string is NOT an FE operator name and must not be registry-resolved.
        # Both are platform-authoring conveniences, not runnable FE operator
        # names; skip so the unchanged platform functions.yaml loads, and let a
        # formula that actually uses the name fail at parse/allowlist as any
        # unsupported name would.
        if " " in canonical_or_alias or "." in canonical_or_alias:
            if _config_path() is None and path is None:
                raise LQTPFunctionsConfigError(
                    f"external alias {external!r} targets unknown canonical "
                    f"{canonical_or_alias!r}"
                )
            continue
        if not _looks_like_operator_name(canonical_or_alias):
            if _config_path() is None and path is None:
                raise LQTPFunctionsConfigError(
                    f"external alias {external!r} targets unknown canonical "
                    f"{canonical_or_alias!r}"
                )
            continue
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

    # Parameterized function aliases (``atr: "ts_mean(true_range, ${window})"``)
    # reuse the template machinery: derive the parameter list from the
    # ``${param}`` placeholders in the body so the platform functions.yaml
    # loads unchanged and its names become runnable LQTP DSL templates.
    for name, body in aliases.items():
        if "${" not in body or name in out:
            continue
        # Substitute ${param} with param FIRST so the body is valid Python AST
        # (e.g. ``ts_mean(true_range, ${window})`` -> ``ts_mean(true_range, window)``).
        expr = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", lambda m: m.group(1), body)
        explicit_names = [m.group(1) for m in re.finditer(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", body)]
        # Macro references: bare names that reference OTHER platform aliases /
        # templates (e.g. ``true_range`` inside ``atr``).  Any name that is a
        # platform alias whose body is a plain (non-${}) expression OR a
        # parameterized template is a macro.  Expand them so the final
        # template only references the caller's series + explicit scalar
        # params.  ``true_range`` itself is a platform alias whose body is a
        # full ``where(...)`` expression.
        def _expand_macros(text: str, depth: int = 0) -> str:
            if depth > 4:
                return text
            for n in _call_order(text):
                if n in explicit_names:
                    continue
                mbody = aliases.get(n)
                if mbody is None or "${" in mbody:
                    continue
                # Only expand when the alias body is itself a runnable operator
                # expression with a Call/Compare node (e.g. ``true_range:
                # "where(...)"``).  Data-field aliases (``high:
                # "StockDailyBar.High * ..."``) and bare DataTable paths are the
                # platform adjustment-basis fields, already applied at the
                # source layer — leave the bare name as the series input so the
                # template signature uses ``high`` directly.
                try:
                    _mtree = ast.parse(str(mbody), mode="eval")
                    _has_call = any(
                        isinstance(node, (ast.Call, ast.Compare))
                        for node in ast.walk(_mtree)
                    )
                except Exception:
                    _has_call = False
                if not _has_call:
                    continue
                text = text.replace(n, f"({mbody})")
                return _expand_macros(text, depth + 1)
            return text

        expr = _expand_macros(expr)
        try:
            all_names = _call_order(expr)
        except Exception:
            continue
        # Bare series inputs: names that are not operator calls and not the
        # explicit ${...} scalar params.  ``_call_order`` returns ALL bare
        # names (including operator calls like ts_mean) — filter those against
        # the allowlist so only data-series inputs remain.
        try:
            bare = [
                n
                for n in all_names
                if n not in base_allow
                and n not in explicit_names
            ]
        except Exception:
            continue
        # Explicit ${...} placeholders become trailing scalar params.
        explicit = []
        for m in re.finditer(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", body):
            if m.group(1) not in explicit:
                explicit.append(m.group(1))
        params = bare + explicit
        if not params:
            continue
        try:
            out[name] = _template_callable(name, params, expr, base_allow)
        except Exception:
            pass
    return out
