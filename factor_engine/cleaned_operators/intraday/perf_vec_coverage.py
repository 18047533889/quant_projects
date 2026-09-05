# -*- coding: utf-8 -*-
"""PERF-2 vector coverage generator (GO_PROMPT §6.1 / §6.3).

Builds a declarative JSON snapshot of which ``daily_agg{,_two,_three}``-routed
intraday operators run through the vectorized (day, bar, inst) fast path vs.
which fall back to the per-(inst, day) Python scalar loop.

The scalar loop is the *only* reason an operator is NOT vectorized:

* ``perf_vec_kernels.bind_whitelist()`` attaches ``__vec__`` only to kernels
  whose equivalence the PERF-2 harness proved (rtol/atol 1e-12) over random
  fixtures incl. NaN gaps and all-NaN days.
* Every other routed operator passes a module-level ``_fn``/``lambda`` closure
  to ``daily_agg*``.  The dispatch hook (``_core._vec_daily_agg*``) can only
  read ``fn.__vec__``; a freshly-built ``lambda`` has no such attribute, so it
  always falls back to the scalar path.

A lambda or module-level ``_fn`` CANNOT expose ``__vec__`` — that requires the
raw named kernel itself to be passed (``true_gap_batch3._session_mean_reversion_kernel``,
``_price_delay_kernel``, ``_volume_imbalance_kernel``, ``smart_money._stock_graph_features``,
``_common_trading_intensity``, ``vwap_path._time_above_vwap``).  That raw-kernel
call convention is exactly the PERF-2 pattern the vectorized whitelist targets.
"""
from __future__ import annotations

import ast
import os
import subprocess
from dataclasses import dataclass
from typing import Any

from factor_engine.cleaned_operators.intraday import perf_vec_kernels as pvk
from factor_engine.cleaned_operators.intraday import _core

_INTRA_OPERATORS_MODULES = (
    "higher_moments",
    "realized_beta",
    "time_structure",
    "vwap_path",
    "overnight",
    "topology_manifold",
    "state_space",
    "pattern_recognition",
    "intra_state_space",
    "smart_money",
    "true_gap_batch3",
    "sufficient_stats_ops",
)

# Dispatch points we route through daily_agg{,_two,_three}.
_DAILY_AGG_CALLS = ("daily_agg", "daily_agg_two", "daily_agg_three")


@dataclass
class _Route:
    canonical: str
    module: str
    aggf: str
    arg_expr: str
    route_kind: str  # "raw-kernel-fn" | "module-level-fn" | "lambda" | "unknown"
    line: int


# Factory-registered canonical -> (module, arg_expr, route_kind) fallback for
# operators whose registered name is a dynamic (variable) argument — an
# ``ast.Name``, invisible to the AST route scan (e.g. the intraday
# sufficient-statistics family).  Records point at the module-level raw
# kernel so the report still shows them being dispatched to daily_agg*.
_FACTORY_CANONICALS: dict[str, tuple[str, str, str]] = {
    "intra_ts_sum": ("sufficient_stats_ops", "_ts_sum", "module-level-fn"),
    "intra_ts_mean": ("sufficient_stats_ops", "_ts_mean", "module-level-fn"),
    "intra_ts_variance": ("sufficient_stats_ops", "_ts_variance", "module-level-fn"),
    "intra_ts_std": ("sufficient_stats_ops", "_ts_std", "module-level-fn"),
    "intra_ts_min": ("sufficient_stats_ops", "_ts_min", "module-level-fn"),
    "intra_ts_max": ("sufficient_stats_ops", "_ts_max", "module-level-fn"),
    "intra_ts_last": ("sufficient_stats_ops", "_ts_last", "module-level-fn"),
    "intra_ts_first": ("sufficient_stats_ops", "_ts_first", "module-level-fn"),
    "intra_ts_last_value": ("sufficient_stats_ops", "_ts_last_value", "module-level-fn"),
    "intra_ts_argmax": ("sufficient_stats_ops", "_ts_argmax", "module-level-fn"),
    "intra_ts_argmin": ("sufficient_stats_ops", "_ts_argmin", "module-level-fn"),
    "intra_ts_realized_variance": ("sufficient_stats_ops", "_ts_realized_variance", "module-level-fn"),
    "intra_ts_vwap": ("sufficient_stats_ops", "_ts_vwap", "module-level-fn"),
    "intra_ts_volume_weighted_return": ("sufficient_stats_ops", "_ts_volume_weighted_return", "module-level-fn"),
    "intra_ts_realized_covariance": ("sufficient_stats_ops", "_ts_realized_covariance", "module-level-fn"),
    "intra_ts_amount_weighted_mean": ("sufficient_stats_ops", "_ts_amount_weighted_mean", "module-level-fn"),
}


def _collect_module_routes(module: str) -> list[_Route]:
    """Static (AST) scan of one intraday module for ``daily_agg*`` routes.

    Walks statements in *source order* so each ``daily_agg*`` call is paired
    with the most recently declared ``@register_operator(name=...)`` rather than
    an arbitrary file-level name.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{module}.py")
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    tree = ast.parse(src)

    # Resolve module-level names (raw kernels / module-level _fn) to python values.
    mod_scope: dict[str, object] = {}
    try:
        mod_scope.update(vars(__import__(
            f"factor_engine.cleaned_operators.intraday.{module}", fromlist=["*"]
        )))
    except Exception:  # pragma: no cover - import side effects may fail under bare python
        pass

    routes: list[_Route] = []
    # fn positional index: daily_agg=v[1], daily_agg_two=v[2], daily_agg_three=v[3].
    _FN_INDEX = {"daily_agg": 1, "daily_agg_two": 2, "daily_agg_three": 3}

    def handle_call(call: ast.Call, canonical: str) -> None:
        fn_args = call.args
        fn_i = _FN_INDEX.get(call.func.id, -1)
        if fn_i < 0 or fn_i >= len(fn_args):
            # fall back to an explicit `fn=` keyword.
            fn_i = -1
            for kw in (call.keywords or []):
                if kw.arg == "fn":
                    fn_arg = kw.value
                    break
        else:
            fn_arg = fn_args[fn_i]
        if fn_arg is None:
            return
        arg_src = ast.get_source_segment(src, fn_arg) or "<?>"
        routes.append(_Route(
            canonical=canonical,
            module=module,
            aggf=call.func.id,
            arg_expr=arg_src[:60],
            route_kind=_classify_arg(fn_arg, arg_src, mod_scope),
            line=getattr(call, "lineno", 0),
        ))

    def find_daily_calls(class_node: ast.ClassDef, canonical: str) -> None:
        """Collect every ``daily_agg*`` Call inside one operator class body."""
        for child in ast.walk(class_node):
            if child is class_node:
                continue
            if isinstance(child, ast.Call) and getattr(child.func, "id", "") in _DAILY_AGG_CALLS:
                handle_call(child, canonical)

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        # Canonical comes from the @register_operator(name="...") decorator.  A
        # factory class whose @register_operator uses a dynamic name= argument
        # (e.g. higher_moments._TailOp) emits MULTIPLE operators from one shared
        # body, so it must not be double-counted under a fabricated canonical —
        # skip it here (its callers are still captured when the concrete
        # registered operators are collected via the canonical-name path).
        canonical = ""
        for dec in node.decorator_list:
            if getattr(dec.func, "id", "") == "register_operator":
                for kw in (dec.keywords or []):
                    if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                        canonical = str(kw.value.value)
        if not canonical:
            continue
        find_daily_calls(node, canonical)
    return routes


def _classify_arg(fn_arg: ast.AST, arg_src: str, mod_scope: dict) -> str:
    if isinstance(fn_arg, ast.Lambda):
        return "lambda"
    if arg_src.startswith("lambda") or ("lambda v" in arg_src) or ("lambda c" in arg_src):
        return "lambda"
    if arg_src.endswith("_kernel"):
        return "raw-kernel-fn"
    if mod_scope.get(arg_src) is not None and callable(mod_scope[arg_src]):
        return "module-level-fn"
def _parse_bound_id(bid: str) -> tuple[str, str]:
    """Parse a whitelist id like ``module:operator[min_finite]`` -> (module, operator)."""
    op = bid
    module = ""
    if ":" in bid:
        module, op = bid.split(":", 1)
    if "[" in op:
        op = op.split("[", 1)[0]  # strip the [min_finite] param suffix
    return module, op


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip() or "unknown"
    except Exception:  # pragma: no cover - not a git env
        return "unknown"


def build_vector_coverage() -> dict:
    """Render the full intraday vector-coverage snapshot as a dict.

    - Enumerates ``bind_whitelist()`` ids (the vectorized set).
    - Enumerates ALL statically-routed ``daily_agg*`` operators.
    - Fails closed (asserts ``count_bound() == len(bind_whitelist())``).
    """
    # Activate the whitelist (import triggers it, but be explicit + idempotent).
    bound_ids = list(pvk.bind_whitelist())
    bound_count = int(pvk.count_bound())
    # Fail-closed: a bind whose vector impl is missing is an error, not a report.
    assert bound_count == len(bound_ids), (
        f"PERF-2 count_bound()={bound_count} != len(bind_whitelist())={len(bound_ids)}; "
        "a whitelisted bind lost its live __vec__ — refusing to emit an inconsistent report."
    )

    # Map whitelist id operator -> (module, vec impl name, proven).
    # ``bind_whitelist()`` returns ids in the SAME order as it populates
    # ``_BOUND_FNS``, so we zip them to pair each id's canonical (the
    # ``module:operator[min_finite]`` operator part) with its bound vector impl.
    bound_fns = list(getattr(pvk, "_BOUND_FNS", []))
    proven: dict[str, tuple[str, str, str]] = {}
    for bid, fn in zip(bound_ids, bound_fns):
        _bmod, op_canonical = _parse_bound_id(bid)
        vec = getattr(fn, "__vec__", None)
        if vec is None:
            continue
        module = fn.__module__.rsplit(".", 1)[-1]
        proven[op_canonical] = (module, vec.__name__, "harness-proven (rtol/atol 1e-12)")

    # Merge proven + unproven routed routes into one operator table.
    seen: dict[str, list] = {}
    for module in _INTRA_OPERATORS_MODULES:
        for r in _collect_module_routes(module):
            entry = seen.setdefault(r.canonical, [])
            # De-duplicate same canonical+aggf+arg (dual `_calculate_series` variants).
            if any(
                old["aggf"] == r.aggf and old["arg_expr"] == r.arg_expr
                for old in entry
            ):
                continue
            entry.append({
                "canonical": r.canonical,
                "module": r.module,
                "aggf": r.aggf,
                "arg_expr": r.arg_expr,
                "route_kind": r.route_kind,
                "line": r.line,
            })

    # Factory-registered operators (``@register_operator(name=<var>, ...)``)
    # are invisible to the AST route scan (the name is an ``ast.Name``, not a
    # ``Constant``).  Surface them via the declarative fallback table so a
    # bound kernel is never dropped from the report.
    for _canonical, (_pmod, _kname, _rk) in _FACTORY_CANONICALS.items():
        if _canonical not in seen:
            seen[_canonical] = [{
                "canonical": _canonical,
                "module": _pmod,
                "aggf": "daily_agg" if _rk == "module-level-fn" and len(_canonical.split("_")) <= 3 else "daily_agg_two",
                "arg_expr": _kname,
                "route_kind": _rk,
                "line": 0,
            }]

    operators: list[dict] = []
    for canonical, routes in seen.items():
        if canonical in proven:
            _pmod, _vimpl, _equidoc = proven[canonical]
            operators.append({
                "canonical": canonical,
                "scalar_impl": f"{_pmod}:proven-scalar-kernel",
                "vector_impl": _vimpl,
                "vector_bound": True,
                "vector_equivalence": _equidoc,
                "fallback_reason": None,
                "routes": routes,
            })
            continue

        # Unproven: choose the most representative single route.
        primary = routes[0]
        rk = primary["route_kind"]
        if rk == "lambda":
            reason = (
                "kernel not vectorized: per-(inst,day) Python loop — fresh lambda closure "
                "carries no __vec__"
            )
        elif rk == "module-level-fn":
            reason = (
                "kernel not vectorized: per-(inst,day) Python loop — module-level _fn carries "
                "no __vec__; PERF-2 requires the raw named kernel be passed"
            )
        elif rk == "raw-kernel-fn":
            reason = (
                "kernel not vectorized: raw kernel passed but not PERF-2-whitelisted "
                "(needs harness equivalence before binding)"
            )
        else:
            reason = "kernel not vectorized: per-(inst,day) Python loop"

        scalar_impl = f"{primary['module']}:{primary['arg_expr']}"
        operators.append({
            "canonical": canonical,
            "scalar_impl": scalar_impl,
            "vector_impl": None,
            "vector_bound": False,
            "vector_equivalence": "unproven",
            "fallback_reason": reason,
            "routes": routes,
        })

    operators.sort(key=lambda o: (not o["vector_bound"], o["canonical"]))
    total = len(operators)
    vectorized = sum(1 for o in operators if o["vector_bound"])
    return {
        "schema": "intraday_vector_coverage/v1",
        "version": "1.0.0",
        "generated_at": _git_head(),
        "bound_count": bound_count,
        "operators": operators,
        "summary": {
            "total": total,
            "vectorized": vectorized,
            "scalar_only": total - vectorized,
        },
    }


def render_intraday_markdown(cov: dict) -> str:
    """Render the coverage snapshot as the INTRADAY_VECTOR_COVERAGE.md table."""
    lines = [
        "# Intraday Vector Coverage",
        "",
        "Vectorized `daily_agg` kernel routing snapshot (GO_PROMPT §6.1 / §6.3).",
        "Generated head SHA: `{sha}`".format(sha=cov["generated_at"]),
        "Bound WHITELIST ids: `{}` (count_bound == len(bind_whitelist()) == {})".format(
            cov["bound_count"], cov["bound_count"]
        ),
        "",
        "Regenerate: `python3 scripts/generate_intraday_vector_coverage.py`",
        "",
        "| canonical | vector_bound | scalar_impl | vector_impl | fallback_reason |",
        "|---|---|---|---|---|",
    ]
    for o in cov["operators"]:
        lines.append(
            "| {c} | {b} | {si} | {vi} | {fr} |".format(
                c=o["canonical"],
                b=str(o["vector_bound"]),
                si=_md(o["scalar_impl"]),
                vi=_md(o["vector_impl"]),
                fr=_md(o["fallback_reason"]),
            )
        )
    s = cov["summary"]
    lines.append("")
    lines.append(
        "## Summary — total={total} · vectorized={vectorized} · scalar_only={scalar_only}".format(
            total=s["total"], vectorized=s["vectorized"], scalar_only=s["scalar_only"]
        )
    )
    lines.append("")
    lines.append(
        "> Scalar-only operators are unvectorized because a fresh `lambda` / module-level `_fn` "
        "closure carries no `__vec__`; only the PERF-2 harness-proven raw kernel call convention "
        "routes to the vectorized fast path."
    )
    return "\n".join(lines)


def _md(v: Any) -> str:
    if v is None:
        return "—"
    return str(v).replace("|", "/").replace("\n", " ")
