# -*- coding: utf-8 -*-
"""Audit every R57 row for positional-arity / type clashes against the registered
operator contract.

`engine.compile()` does not validate positional arity against `metadata.param_names`,
so a row can compile green and then explode at execution -- e.g. QQE is declared as
`x, length, smooth, factor, output` but the catalog calls
`QQE(open, high, low, close, volume)`, binding `length` to a DataFrame.

Two corrections over the first pass, both needed to avoid false positives:
  * calls are resolved through the alias table before the registry lookup, otherwise
    every `multiply`/`subtract`/`add` alias reads as "operator not registered";
  * operators whose evaluation signature is variadic (`*args`, e.g. row_sum_skipna
    and the cs_* feature-matrix families) are exempt from the positional-arity check,
    because taking N series is their contract.

Read-only. Shardable. Writes only --outdir.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import inspect
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_PARSER = None
_SPECS: dict[str, dict] = {}
_CANON: dict[str, str] = {}


def _build_specs() -> tuple[dict[str, dict], dict[str, str]]:
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    load_all()
    specs: dict[str, dict] = {}
    canon: dict[str, str] = {}
    for canonical in sorted(OperatorRegistry._catalog):
        canon[canonical] = canonical
        op = None
        for backend in ("pandas_numpy", "polars", "pandas"):
            try:
                op = OperatorRegistry.get(canonical, backend, mode="any")
            except Exception:  # noqa: BLE001
                op = None
            if op is not None:
                break
        if op is None:
            continue
        md = getattr(op, "metadata", None)
        names = list(getattr(md, "param_names", None) or [])
        pspecs = set((getattr(md, "param_specs", None) or {}).keys())
        # NOTE: the public ``calculate`` wrapper is always ``(*args, **kwargs)``,
        # so it tells us nothing. The real binding contract is ``_calculate_series``
        # (and, for the polars natives, ``_calculate_series`` too).
        variadic = not names or "..." in names
        if not variadic:
            for attr in ("_calculate_series", "apply", "__call__"):
                fn = getattr(op, attr, None)
                if fn is None:
                    continue
                try:
                    params = inspect.signature(fn).parameters
                except (TypeError, ValueError):
                    break
                variadic = any(
                    p.kind is inspect.Parameter.VAR_POSITIONAL for p in params.values()
                )
                break
        specs[canonical] = {
            "param_names": names,
            "scalar_params": pspecs,
            "variadic": variadic,
        }
    # aliases -> canonical
    try:
        for alias, target in dict(OperatorRegistry._aliases).items():
            canon[str(alias)] = str(target)
    except Exception:  # noqa: BLE001
        pass
    return specs, canon


def _init() -> None:
    global _PARSER, _SPECS, _CANON
    from factor_engine.api.dsl_parser import DSLParser

    _SPECS, _CANON = _build_specs()
    _PARSER = DSLParser(surface="compat_research", dialect="native")


def _walk(node):
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        kid = list(n.children() or [])
        for _, value in getattr(n, "kwargs", ()) or ():
            if hasattr(value, "children"):
                kid.append(value)
        stack.extend(kid)


def _audit_formula(formula: str) -> list[dict]:
    from factor_engine.expr.cleaned_call import CleanedCall
    from factor_engine.expr.literal import Literal

    issues: list[dict] = []
    expr = _PARSER.parse(formula)
    for node in _walk(expr):
        if not isinstance(node, CleanedCall):
            continue
        call = node.op
        canonical = _CANON.get(call)
        if canonical is None:
            try:
                from factor_engine.cleaned_operators.registry import OperatorRegistry

                canonical = OperatorRegistry.resolve_canonical_optional(call)
            except Exception:  # noqa: BLE001
                canonical = None
        spec = _SPECS.get(canonical) if canonical else None
        if spec is None:
            issues.append({"op": call, "canonical": canonical, "kind": "OPERATOR_NOT_REGISTERED"})
            continue
        names = spec["param_names"]
        scalars = spec["scalar_params"]
        args = list(node.args or ())
        for k, _ in node.kwargs or ():
            if k not in names:
                issues.append({"op": canonical, "kind": "UNKNOWN_KWARG", "kwarg": k})
        if spec["variadic"]:
            continue
        if len(args) > len(names):
            issues.append(
                {
                    "op": canonical,
                    "kind": "TOO_MANY_POSITIONAL",
                    "given": len(args),
                    "declared": len(names),
                    "param_names": names,
                }
            )
        for i, a in enumerate(args[: len(names)]):
            pname = names[i]
            if pname in scalars and not isinstance(a, Literal):
                issues.append(
                    {
                        "op": canonical,
                        "kind": "SERIES_INTO_SCALAR_PARAM",
                        "position": i,
                        "param": pname,
                        "arg_type": type(a).__name__,
                        "param_names": names,
                    }
                )
    return issues


def _work(payload):
    shard, shards, manifest_dir, out_path = payload
    _init()
    grouped: collections.Counter = collections.Counter()
    samples: dict[str, list] = {}
    my_rows = 0
    parsed = 0
    flagged = 0
    parse_fail = collections.Counter()
    with gzip.open(out_path, "wt", encoding="utf-8") as dst:
        for path in sorted(Path(manifest_dir).glob("full-s*.jsonl.gz")):
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    r = json.loads(line)
                    if r.get("compile_status") != "COMPILED":
                        continue
                    idx = my_rows
                    my_rows += 1
                    if idx % shards != shard:
                        continue
                    try:
                        issues = _audit_formula(r.get("r57_formula") or "")
                        parsed += 1
                    except Exception as exc:  # noqa: BLE001
                        parse_fail[f"{type(exc).__name__}: {str(exc)[:120]}"] += 1
                        continue
                    if not issues:
                        continue
                    flagged += 1
                    seen = set()
                    for it in issues:
                        key = json.dumps({k: v for k, v in it.items() if k != "param_names"}, sort_keys=True)
                        if key in seen:
                            continue
                        seen.add(key)
                        grouped[f"{it['kind']}|{it['op']}"] += 1
                        samples.setdefault(key, []).append({"id": r.get("id"), "issue": it})
                    dst.write(json.dumps({"id": r.get("id"), "issues": issues}, ensure_ascii=False) + "\n")
    return {
        "shard": shard,
        "parsed": parsed,
        "flagged": flagged,
        "parse_fail": dict(parse_fail.most_common(20)),
        "grouped": dict(grouped.most_common(500)),
        "samples": {k: v[:2] for k, v in list(samples.items())[:200]},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--shards", type=int, default=8)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    payloads = [
        (i, args.shards, str(args.manifest_dir), str(args.outdir / f"arity2-s{i}.jsonl.gz"))
        for i in range(args.shards)
    ]
    parsed = flagged = 0
    grouped: collections.Counter = collections.Counter()
    parse_fail: collections.Counter = collections.Counter()
    samples: dict[str, list] = {}
    with ProcessPoolExecutor(max_workers=args.shards) as ex:
        for res in [f.result() for f in as_completed([ex.submit(_work, p) for p in payloads])]:
            parsed += res["parsed"]
            flagged += res["flagged"]
            grouped.update(res["grouped"])
            parse_fail.update(res["parse_fail"])
            for k, v in res["samples"].items():
                samples.setdefault(k, v)

    kinds: collections.Counter = collections.Counter()
    for k, v in grouped.items():
        kinds[k.split("|")[0]] += v
    summary = {
        "rows_parsed": parsed,
        "rows_flagged": flagged,
        "parse_failures": dict(parse_fail.most_common(20)),
        "totals_by_kind": dict(kinds.most_common()),
        "issues_by_kind_and_op": dict(grouped.most_common(500)),
        "samples": samples,
    }
    (args.outdir / "arity2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("rows_parsed:", parsed, "rows_flagged:", flagged)
    print("parse_failures:", dict(parse_fail.most_common(6)))
    print("totals_by_kind:", dict(kinds.most_common()))
    print()
    print("=== issue groups (kind|op) ===")
    for k, v in grouped.most_common(45):
        print(f"  {v:7d}  {k}")


if __name__ == "__main__":
    main()
