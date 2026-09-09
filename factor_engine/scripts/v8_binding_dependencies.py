"""Read-only callable dependency evidence, not a numerical certification gate.

Traverses concrete winning callables, captured factories and referenced Python
helpers. Unresolved dynamic state is reported, never silently certified.
"""
from functools import partial
from dataclasses import fields, is_dataclass
import hashlib
import inspect
import json
from pathlib import Path
import types


def binding_dependencies(operator, *, max_nodes=256, max_values=4096):
    if type(max_nodes) is not int or max_nodes < 0 or type(max_values) is not int or max_values < 0:
        raise ValueError("dependency budgets must be nonnegative integers")
    nodes = {}
    unresolved = set()
    active = set()
    visited_values = 0

    def visit(value):
        nonlocal visited_values
        visited_values += 1
        if visited_values > max_values or len(active) >= 64:
            unresolved.add("value-or-depth-budget-exhausted")
            return {"unresolved": "value-or-depth-budget-exhausted"}
        if value is None or type(value) in (bool, int, str):
            return value
        if type(value) is float:
            return {"float": value.hex()}
        if is_dataclass(value) and not isinstance(value, type):
            if id(value) in active:
                unresolved.add("cyclic-dataclass")
                return {"unresolved": "cyclic-dataclass"}
            if len(fields(value)) > max_values - visited_values:
                unresolved.add("value-or-depth-budget-exhausted")
                return {"unresolved": "value-or-depth-budget-exhausted"}
            active.add(id(value))
            result = visit({field.name: getattr(value, field.name) for field in fields(value)})
            active.remove(id(value))
            return result
        if isinstance(value, (tuple, list, dict)) and len(value) > max_values - visited_values:
            unresolved.add("value-or-depth-budget-exhausted")
            return {"unresolved": "value-or-depth-budget-exhausted"}
        if isinstance(value, (tuple, list)):
            if id(value) in active:
                unresolved.add("cyclic-container")
                return {"unresolved": "cyclic-container"}
            active.add(id(value))
            result = [visit(v) for v in value]
            active.remove(id(value))
            return result
        if isinstance(value, dict) and all(type(k) is str for k in value):
            if id(value) in active:
                unresolved.add("cyclic-mapping")
                return {"unresolved": "cyclic-mapping"}
            active.add(id(value))
            result = {k: visit(v) for k, v in sorted(value.items())}
            active.remove(id(value))
            return result
        if isinstance(value, partial):
            return {"partial": visit(value.func), "args": visit(value.args),
                    "keywords": visit(value.keywords)}
        if inspect.ismethod(value):
            value = value.__func__
        if isinstance(value, types.ModuleType):
            path = getattr(value, "__file__", None)
            record = {"module": value.__name__, "version": getattr(value, "__version__", None)}
            if path and value.__name__.startswith("factor_engine."):
                record["source_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
                unresolved.add("dynamic-module-attributes:" + value.__name__)
            return record
        if inspect.isfunction(value):
            key = value.__module__ + ":" + value.__qualname__
            # Multiple closure instances may share the same qualified name.
            # Return captured values separately so a factory parameter matters.
            if id(value) in active:
                return {"recursive": key}
            if len(nodes) + len(active) >= max_nodes:
                unresolved.add("node-budget-exhausted")
                return {"unresolved": key}
            active.add(id(value))
            try:
                source = inspect.getsource(value)
            except (OSError, TypeError):
                source = None
                unresolved.add("source-unavailable:" + key)
            closure = inspect.getclosurevars(value)
            # Attribute lookups cannot be statically resolved from a code name.
            # Keep this bound explicit instead of asserting full certification.
            if closure.unbound:
                unresolved.add("dynamic-lookups:" + key)
            payload = {"symbol": key, "source": source,
                       "defaults": visit(value.__defaults__),
                       "kwdefaults": visit(value.__kwdefaults__),
                       "captures": {k: visit(v) for k, v in sorted(closure.nonlocals.items())},
                       "globals": {k: visit(v) for k, v in sorted(closure.globals.items())}}
            active.remove(id(value))
            digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            nodes[digest] = payload
            return {"callable": digest}
        label = type(value).__module__ + ":" + type(value).__qualname__
        unresolved.add("unresolved-value:" + label)
        return {"unresolved": label}

    entrypoints = {}
    for name in ("calculate", "_calculate_series", "_calculate_scalar", "_fn", "_build_sql_specs"):
        function = getattr(operator, name, None)
        if callable(function):
            entrypoints[name] = visit(function)
            if name == "_build_sql_specs":
                # SQL registry entries are declarations, not execution kernels.
                # Observing this method does not traverse the runtime emitter.
                unresolved.add("sql-declaration-only-runtime-emitter-not-traced")
    if not entrypoints:
        unresolved.add("no-callable-entrypoints")
    payload = {"schema": "factor_engine.v8.callable_dependencies.v1",
               "entrypoints": entrypoints,
               "metadata": visit(getattr(operator, "metadata", None)),
               "nodes": dict(sorted(nodes.items()))}
    return {"dependency_hash": hashlib.sha256(
                json.dumps(payload, sort_keys=True).encode()).hexdigest(),
            "payload": payload, "unresolved": sorted(unresolved),
            "coverage": "PARTIAL" if unresolved else "RESOLVED_PYTHON_CALLABLE_GRAPH",
            "mathematical_certification": "NOT_CLAIMED"}


def main():
    import argparse
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--canonical", action="append")
    selection.add_argument("--all-canonicals", action="store_true")
    parser.add_argument("--max-nodes", type=int, default=256)
    parser.add_argument("--max-values", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    args = parser.parse_args()
    previous = {}
    if args.previous:
        with args.previous.open() as stream:
            for line in stream:
                row = json.loads(line)
                key = (row["canonical"], row["backend"])
                if key in previous:
                    raise ValueError("duplicate previous binding")
                previous[key] = row["dependency_hash"]
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    from factor_engine.backend.operator_semantic_version import semantic_version
    load_all()
    names = (sorted(OperatorRegistry.list_canonical()) if args.all_canonicals else
             sorted({OperatorRegistry.resolve_canonical(name) for name in args.canonical}))
    unknown = set(names) - set(OperatorRegistry.list_canonical())
    if unknown:
        raise ValueError(f"unknown canonicals: {sorted(unknown)}")
    with args.output.open("x", encoding="utf-8") as stream:
        for name in names:
            for backend in sorted(OperatorRegistry.backends_for(name)):
                # This is a read-only inventory, not production admission.
                # The default production gate returns None for research-only
                # operators; hashing None would invent an empty "resolved"
                # dependency graph instead of observing the actual winner.
                operator = OperatorRegistry.get(name, backend, mode="any")
                if operator is None:
                    raise RuntimeError(f"advertised binding unavailable: {name}/{backend}")
                record = binding_dependencies(operator, max_nodes=args.max_nodes,
                                              max_values=args.max_values)
                old = previous.get((name, backend))
                record.update({
                    "canonical": name, "backend": backend,
                    "implementation": type(operator).__module__ + ":" + type(operator).__qualname__,
                    "semantic_version": semantic_version(name),
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                    "prior_evidence": "STALE_REQUIRES_REVALIDATION" if old is not None
                        and old != record["dependency_hash"] else "NOT_CERTIFIED_BY_THIS_AUDIT",
                })
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
