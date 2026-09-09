"""
Coverage compiler for quant_evaluator metrics.

Scans REAL code — never hand-written claims — and produces a coverage matrix
CSV answering, per metric:

    metric | kernel | registry | artifact | test | report | prod_status

Each column is computed by actually inspecting the source tree:

    - ``kernel``      : the ``implementation_id`` from ``metrics.catalog`` is
                        resolved (module + attribute) against the real
                        ``quant_evaluator.metrics`` package.  ``YES`` only if
                        the import succeeds and the attribute exists.
    - ``registry``    : the metric is bound in ``registry.metrics`` (STABLE
                        spec with a ``compute_fn``) or appears in
                        ``registry.metrics.catalog_snapshot()``.
    - ``artifact``    : at least one consumer in ``reporting/`` or
                        ``metrics/`` serializes/consumes the metric's declared
                        ``artifact_kind`` via the canonical
                        ``contracts.metric_artifacts`` (either a concrete
                        artifact class reference or a MetricEvidence/ChartSpec
                        consumer mention).
    - ``test``        : the metric_id (or its implementation function name)
                        is mentioned in ``tests/``.
    - ``report``      : the metric's id or its implementation function name
                        is mentioned in ``reporting/`` (a panel consuming it).
    - ``prod_status`` : INVENTORY_WIRED when kernel+registry+artifact+test+report all
                        hold; GAP when at least kernel/registry hold but one
                        of artifact/test/report is missing; NOT_IMPL when the
                        kernel does not resolve.

The compiler is idempotent and deterministic (sorted rows), and refuses to
touch anything outside ``quant_evaluator``.
No row certifies mathematical correctness, executed assertions or production
admission. NumericalQualificationReceipt is independent of this inventory.
"""

from __future__ import annotations

import csv
import hashlib
import inspect
from pathlib import Path
import importlib
import os
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

_ARTIFACT_CLASS_RE = re.compile(
    r"(ScalarMetricArtifact|SeriesMetricArtifact|VectorMetricArtifact|"
    r"MatrixMetricArtifact|DistributionMetricArtifact|MetricArtifact|MetricEvidence)"
)

_METRIC_MENTION_RE = re.compile(r"[A-Za-z0-9_\.]+")


def canonical_entrypoint_inventory():
    """M22 source/registry/package/consumer graph, explicitly not certification.

    Use the real registry binding (including parameterized partials); keep
    static test references separate from independently executed golden runs.
    """
    import functools
    import tomllib
    from quant_evaluator.registry.metrics import catalog_snapshot
    root = Path(_pkg_root())
    repo = root.parent
    packages = set(tomllib.loads((root / "pyproject.toml").read_text())["tool"]["setuptools"]["packages"])
    test_roots = [root / "tests", repo / "integration_tests",
                  repo / "factor_assets/tests", repo / "factor_optimizer/tests",
                  repo / "factor_preprocess/tests"]
    texts = {str(p.relative_to(repo)): p.read_text(errors="replace")
             for folder in test_roots if folder.exists() for p in folder.rglob("test_*.py")}
    rows = []
    for metric_id, spec in sorted(catalog_snapshot().items()):
        fn = spec.compute_fn
        if fn is None:
            rows.append({"metric_id": metric_id, "status": "NO_BOUND_IMPLEMENTATION",
                         "golden_execution_status": "NOT_RUN"})
            continue
        base = fn.func if isinstance(fn, functools.partial) else fn
        base = getattr(base, "py_func", base)
        source = Path(inspect.getsourcefile(base)).resolve()
        module = base.__module__
        references = sorted(path for path, text in texts.items()
                            if metric_id in text or base.__name__ in text)
        rows.append({
            "metric_id": metric_id, "public_entry": "quant_evaluator.evaluate",
            "implementation": module + "." + base.__qualname__,
            "source_path": str(source.relative_to(repo)),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "registry_implementation_hash": spec.implementation_hash,
            "parameter_contract": str(inspect.signature(fn)),
            "bound_parameters": dict(fn.keywords or {}) if isinstance(fn, functools.partial) else {},
            "packaged": module.rpartition(".")[0] in packages,
            "wheel_path": module.replace(".", "/") + ".py",
            "static_test_references": references,
            "golden_execution_status": "NOT_RUN",
            "consumer_execution_status": "NOT_RUN",
            "status": "INVENTORY_ONLY",
        })
    return rows


def _pkg_root() -> str:
    """Return the absolute path of the installed quant_evaluator package."""
    import quant_evaluator

    return os.path.dirname(os.path.abspath(quant_evaluator.__file__))


def _scan_tree(root: str, *, include_subdirs: Tuple[str, ...] = ()) -> List[str]:
    """Return all .py files under root (excluding __pycache__ and build/)."""
    files: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune noise
        dirnames[:] = [
            d
            for d in dirnames
            if d not in ("__pycache__", "build", ".pytest_cache", ".git", "quant_evaluator.egg-info")
        ]
        if include_subdirs and not any(
            dirpath.startswith(os.path.join(root, s)) or dirpath == root
            for s in include_subdirs
        ):
            if dirpath != root:
                continue
        for fn in filenames:
            if fn.endswith(".py"):
                files.append(os.path.join(dirpath, fn))
    return sorted(files)


def _resolve_implementation(implementation_id: str) -> Tuple[bool, str]:
    """Resolve ``module.attr`` against the real codebase.

    Returns (resolved, detail).  Fail closed: any import error => False.
    """
    if not implementation_id or "." not in implementation_id:
        return False, f"malformed implementation_id {implementation_id!r}"
    module_path, _, attr = implementation_id.rpartition(".")
    if not module_path or not attr:
        return False, f"malformed implementation_id {implementation_id!r}"
    try:
        module = importlib.import_module(module_path)
    except Exception as exc:  # noqa: BLE001 - report, do not raise
        return False, f"import failed: {type(exc).__name__}: {exc}"
    if not hasattr(module, attr):
        return False, f"attribute {attr!r} missing in {module_path}"
    return True, ""


def _load_catalog_metric_ids() -> List[str]:
    """Load metric ids from ``metrics.catalog`` (the authoritative catalog)."""
    from quant_evaluator.metrics import catalog as catalog_module

    return catalog_module.list_all_metric_ids()


def _registry_metric_ids() -> Set[str]:
    """Collect metric ids bound in ``registry.metrics`` (real registry)."""
    ids: Set[str] = set()
    from quant_evaluator.registry.metrics import (
        list_metrics,
        catalog_snapshot,
        MetricStatus,
    )

    # The registry also binds metrics not present in metrics.catalog (e.g.
    # coverage, turnover, ic_ir, ic_std...).  Include every STABLE metric with
    # a real bound compute_fn.  catalog ids are the compilation universe; the
    # registry column simply reports whether the id is bound in the registry.
    ids.update(list_metrics())
    for name, spec in catalog_snapshot().items():
        if spec.status is MetricStatus.STABLE and spec.compute_fn is not None:
            ids.add(name)
    return ids


def _file_texts(files: List[str]) -> Dict[str, str]:
    """Read file contents as text (empty string on any read error)."""
    out: Dict[str, str] = {}
    for path in files:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                out[path] = fh.read()
        except OSError:
            out[path] = ""
    return out


def _mentioned(text: str, needle: str) -> bool:
    """True when ``needle`` appears as a real code token in ``text``.

    QE-P0-04: a mention inside a comment, docstring, or string literal is NOT
    a binding — it does not prove the metric is actually consumed.  We strip
    comments and string literals (via the tokenizer) before matching so the
    READY gate is not a false positive from prose.
    """
    if not needle:
        return False
    import tokenize
    from io import StringIO

    code_tokens: List[str] = []
    try:
        for tok in tokenize.generate_tokens(StringIO(text).readline):
            if tok.type == tokenize.COMMENT:
                continue
            if tok.type == tokenize.STRING:
                # Skip docstrings (triple-quoted) but keep real string literals
                # used in code (e.g. get_metric("pearson_ic")).
                if tok.string.startswith(('"""', "'''")):
                    continue
            code_tokens.append(tok.string)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # Unparseable text: fall back to the raw substring check (best effort).
        return False
    return needle in code_tokens or needle in " ".join(code_tokens)


def _scan_artifact_consumers() -> Tuple[Set[str], Set[str]]:
    """Scan reporting/ + metrics/ for canonical artifact-class consumers.

    Returns (kinds_mentioned, ids_mentioned).  ids_mentioned is the set of
    metric ids whose implementation function name (or the id itself) appears
    in the same file that references a canonical artifact class.
    """
    root = _pkg_root()
    kinds: Set[str] = set()
    ids_mentioned: Set[str] = set()
    candidate_files = _scan_tree(
        os.path.join(root, "reporting")
    ) + _scan_tree(os.path.join(root, "metrics"))
    texts = _file_texts(candidate_files)
    for path, text in texts.items():
        kinds.update(name for name in _ARTIFACT_CLASS_RE.findall(text) if _mentioned(text, name))
    # For each metric, does a file that mentions a canonical artifact class
    # also mention the metric's kernel function name?
    from quant_evaluator.metrics import catalog as catalog_module

    for metric_id in catalog_module.list_all_metric_ids():
        spec = catalog_module.get_metric_spec(metric_id)
        impl_fn = spec.implementation_id.rpartition(".")[2] if "." in spec.implementation_id else ""
        for path, text in texts.items():
            if any(_mentioned(text, name) for name in _ARTIFACT_CLASS_RE.findall(text)) and (
                _mentioned(text, metric_id) or _mentioned(text, impl_fn)
            ):
                ids_mentioned.add(metric_id)
                break
    return kinds, ids_mentioned


def compile_coverage_rows() -> List[Dict[str, str]]:
    """Compile the coverage matrix rows (one per catalog metric)."""
    from quant_evaluator.metrics import catalog as catalog_module

    root = _pkg_root()
    tests_files = _scan_tree(os.path.join(root, "tests"))
    reporting_files = _scan_tree(os.path.join(root, "reporting"))
    tests_texts = _file_texts(tests_files)
    reporting_texts = _file_texts(reporting_files)

    registry_ids = _registry_metric_ids()
    _, artifact_ids = _scan_artifact_consumers()

    rows: List[Dict[str, str]] = []
    for metric_id in catalog_module.list_all_metric_ids():
        spec = catalog_module.get_metric_spec(metric_id)
        impl_id = spec.implementation_id
        impl_fn = impl_id.rpartition(".")[2] if "." in impl_id else ""
        resolved, detail = _resolve_implementation(impl_id)

        # test mention: metric_id or kernel function name in tests/
        test_hit = any(
            _mentioned(text, metric_id) or _mentioned(text, impl_fn)
            for text in tests_texts.values()
        )
        # report mention: metric_id or kernel fn in reporting/
        report_hit = any(
            _mentioned(text, metric_id) or _mentioned(text, impl_fn)
            for text in reporting_texts.values()
        )
        registered = metric_id in registry_ids
        artifact_hit = metric_id in artifact_ids

        if not resolved:
            prod_status = "NOT_IMPL"
        elif resolved and registered and artifact_hit and test_hit and report_hit:
            prod_status = "INVENTORY_WIRED"
        else:
            prod_status = "GAP"

        rows.append(
            {
                "metric": metric_id,
                "kernel": "YES" if resolved else "NO",
                "registry": "YES" if registered else "NO",
                "artifact": "YES" if artifact_hit else "NO",
                "test": "YES" if test_hit else "NO",
                "report": "YES" if report_hit else "NO",
                "prod_status": prod_status,
            }
        )
    return rows


def _write_csv(rows: List[Dict[str, str]], out_path: str) -> None:
    """Write the coverage matrix CSV deterministically."""
    fieldnames = ["metric", "kernel", "registry", "artifact", "test", "report", "prod_status"]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r["metric"]):
            writer.writerow(row)


def compile_metric_coverage(out_path: Optional[str] = None) -> str:
    """Compile and write the coverage matrix. Returns the output path."""
    if out_path is None:
        root = _pkg_root()
        out_path = os.path.join(root, "docs", "METRIC_COVERAGE_COMPILER.csv")
    rows = compile_coverage_rows()
    _write_csv(rows, out_path)
    return out_path


if __name__ == "__main__":
    print(compile_metric_coverage())
