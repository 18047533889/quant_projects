# -*- coding: utf-8 -*-
"""Step 3 / batch 2: declare PhysicalImplementationSpec for the 15 highest-frequency
canonical operators whose polars slot is a GENUINE expression kernel but which had no
explicit spec (so production_mode classified them UNSUPPORTED).

Honesty rules applied:
  * Only operators whose registered kernel provably avoids a pandas round-trip are
    touched: verified by a RUNTIME probe (patched pl.DataFrame.to_pandas -> 0 calls)
    and by kernel-body inspection (pl.col/with_columns, no np./pd. tokens).
  * execution_kind is POLARS_NATIVE_EXPR only because the kernel builds a polars
    expression over columns.  Kernels that are per-column NumPy estimators are NOT
    included here (they would be POLARS_NUMPY_KERNEL).
  * supports_lazy / supports_streaming are left at their False defaults: these kernels
    accept a materialised pl.DataFrame and return an eager frame, so claiming lazy
    execution would overstate the implementation.
  * implementation_source_hash follows the same stable plaintext identity-label
    convention the previous step-3/batch-1 commit used.

Dry-run by default; pass --apply to write.
"""
from __future__ import annotations

import ast
import re
import sys

REPO = "/home/sunhaiwei/quant_projects"
APPLY = "--apply" in sys.argv

# (relative path, class name, canonical)
TARGETS = [
    ("factor_engine/cleaned_operators/common/cross_sectional.py", "RankPolars", "rank"),
    ("factor_engine/cleaned_operators/common/cross_sectional.py", "CsPctRankPolars", "cs_pct_rank"),
    ("factor_engine/cleaned_operators/common/cross_sectional.py", "CrossSectionalMeanPolars", "cs_mean"),
    ("factor_engine/cleaned_operators/common/polars_auto.py", "AndPolarsAuto", "and_"),
    ("factor_engine/cleaned_operators/common/time_series.py", "TSSumPolars", "ts_sum"),
    ("factor_engine/cleaned_operators/common/time_series.py", "TSZScorePolars", "ts_zscore"),
    ("factor_engine/cleaned_operators/common/time_series.py", "TSDeltaPolars", "ts_delta"),
    ("factor_engine/cleaned_operators/common/time_series.py", "TSDelayPolars", "ts_delay"),
    ("factor_engine/cleaned_operators/common/time_series.py", "TSMaxPolars", "ts_max"),
    ("factor_engine/cleaned_operators/common/time_series.py", "TSMinPolars", "ts_min"),
    ("factor_engine/cleaned_operators/common/polars_extended.py", "TanhPolars", "tanh"),
    ("factor_engine/cleaned_operators/common/polars_daily_native.py", "TSSharpeNative", "ts_sharpe"),
    ("factor_engine/cleaned_operators/common/polars_daily_native.py", "CSMadZscoreNative", "cs_mad_zscore"),
]

NOTE = ("Genuine polars expression kernel (pl.col/with_columns); runtime probe shows 0 "
        "pl.DataFrame.to_pandas calls and the kernel body has no pandas/NumPy term. "
        "Declared to connect the polars_long channel: execution_kind was previously "
        "absent, so canonical_polars_kind(production_mode=True) reported UNSUPPORTED.")

SPEC = '''{ind}_physical_spec = PhysicalImplementationSpec(
{ind}    canonical="{canon}", backend="polars",
{ind}    execution_kind=ExecutionKind.POLARS_NATIVE_EXPR,
{ind}    materializes_full_panel=True,
{ind}    supports_nulls=True, supports_nan=True, supports_inf=True,
{ind}    implementation_source_hash="{mod}:{cls}:v1",
{ind}    emitter_identity="polars_expr:{canon}",
{ind}    kernel_identity="{mod}:{cls}",
{ind}    parameter_domain_hash="{canon}:declared:v1",
{ind}    semantic_contract_hash="{canon}:polars_native_expr:v1",
{ind}    notes=({note!r}),
{ind})
'''


DYN_SPEC = '''{ind}_physical_spec = PhysicalImplementationSpec(
{ind}    canonical=canon, backend="polars",
{ind}    execution_kind=ExecutionKind.POLARS_NATIVE_EXPR,
{ind}    materializes_full_panel=True,
{ind}    supports_nulls=True, supports_nan=True, supports_inf=True,
{ind}    implementation_source_hash=f"common.polars_auto:{{canon}}:v1",
{ind}    emitter_identity=f"polars_expr:{{canon}}",
{ind}    kernel_identity=f"common.polars_auto:{{canon}}",
{ind}    parameter_domain_hash=f"{{canon}}:declared:v1",
{ind}    semantic_contract_hash=f"{{canon}}:polars_native_expr:v1",
{ind}    notes=(DYNAMIC_NOTE),
{ind})
'''

# operators built by the _register_compare factory: the class is created at runtime and
# renamed, so the spec must reference the enclosing `canon` parameter instead of a literal.
DYNAMIC_NOTE = ("Genuine polars expression kernel (pl.col comparison + with_columns); "
                "runtime probe shows 0 pl.DataFrame.to_pandas calls. Declared to connect "
                "the polars_long channel (execution_kind was previously absent).")

DYNAMIC_TARGETS = [
    ("factor_engine/cleaned_operators/common/polars_auto.py", "_register_compare",
     "_ComparePolars", ["gt", "lt"]),
]


def patch_dynamic(relpath, factory, clsname, canons):
    path = f"{REPO}/{relpath}"
    src = open(path).read()
    tree = ast.parse(src)
    fn_node = None
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == factory:
            fn_node = n
            break
    if fn_node is None:
        raise SystemExit(f"{relpath}: factory {factory} not found")
    target = None
    for n in ast.walk(fn_node):
        if isinstance(n, ast.ClassDef) and n.name == clsname:
            target = n
    if target is None:
        raise SystemExit(f"{relpath}: class {clsname} inside {factory} not found")
    body_src = ast.get_source_segment(src, target) or ""
    if "_physical_spec" in body_src:
        print(f"  SKIP dynamic {clsname} ({canons}): already has _physical_spec")
        return 0
    last = max(target.body, key=lambda s: getattr(s, "end_lineno", s.lineno))
    ind = " " * target.body[0].col_offset
    text = DYN_SPEC.format(ind=ind)
    text = text.replace("DYNAMIC_NOTE", f"DYN_NOTE_{factory}")
    out = src.splitlines(keepends=True)
    out.insert(last.end_lineno, text)
    new_src = "".join(out)
    # inject the note constant + contracts import at module level
    t2 = ast.parse(new_src)
    last_imp = None
    for n in t2.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            last_imp = n
    note_name = f"DYN_NOTE_{factory}"
    import_add = ""
    if re.search(r"^\s*(from|import).*PhysicalImplementationSpec", new_src, re.M) is None:
        import_add = ("from factor_engine.backend.contracts import (\n"
                      "    ExecutionKind, PhysicalImplementationSpec,\n)\n")
    decl = f"\n{note_name} = {DYNAMIC_NOTE!r}\n\n"
    add = decl + import_add
    out2 = new_src.splitlines(keepends=True)
    out2.insert(last_imp.end_lineno, add + decl)
    new_src = "".join(out2)
    new_src = new_src.replace(f"notes=({note_name})", f"notes=({note_name})")
    ast.parse(new_src)
    if APPLY:
        open(path, "w").write(new_src)
    print(f"  {'WROTE' if APPLY else 'DRY-RUN ok'} {relpath} (dynamic {clsname} covers {canons})")
    return len(canons)


def class_nodes(tree):
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            out.setdefault(node.name, []).append(node)
    return out


def module_has_names(src):
    has_spec = re.search(r"^\s*(from|import).*PhysicalImplementationSpec", src, re.M) is not None
    has_ek = re.search(r"^\s*(from|import).*ExecutionKind", src, re.M) is not None
    return has_spec, has_ek


def patch_file(relpath, file_targets):
    path = f"{REPO}/{relpath}"
    src = open(path).read()
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)
    nodes = class_nodes(tree)
    inserts = []
    for cls, canon in file_targets:
        cand = nodes.get(cls)
        if not cand:
            raise SystemExit(f"{relpath}: class {cls} not found")
        if len(cand) > 1:
            raise SystemExit(f"{relpath}: class {cls} defined {len(cand)} times - ambiguous")
        node = cand[0]
        # already patched?
        body_src = ast.get_source_segment(src, node) or ""
        if "_physical_spec" in body_src:
            print(f"  SKIP {cls} ({canon}): already has _physical_spec")
            continue
        last = max(node.body, key=lambda s: getattr(s, "end_lineno", s.lineno))
        insert_after = last.end_lineno            # 1-based
        first_stmt = node.body[0]
        ind = " " * (first_stmt.col_offset)
        mod_label = relpath.split("cleaned_operators/")[-1].replace(".py", "").replace("/", ".")
        text = SPEC.format(ind=ind, canon=canon, cls=cls, mod=mod_label, note=NOTE)
        inserts.append((insert_after, text, cls, canon))
        print(f"  PLAN {cls} ({canon}) insert after line {insert_after} indent={len(ind)} "
              f"module={mod_label}")

    if not inserts:
        print(f"  {relpath}: nothing to do")
        return 0

    out = list(lines)
    for insert_after, text, cls, canon in sorted(inserts, reverse=True):
        out.insert(insert_after, text)
    new_src = "".join(out)

    has_spec, has_ek = module_has_names(new_src)
    if not (has_spec and has_ek):
        # add a top-level import right after the last top-level import
        t2 = ast.parse(new_src)
        last_imp = None
        for n in t2.body:
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                last_imp = n
        if last_imp is None:
            raise SystemExit(f"{relpath}: no top-level import to anchor a new import")
        add = ("from factor_engine.backend.contracts import (\n"
               "    ExecutionKind, PhysicalImplementationSpec,\n"
               ")\n")
        out2 = new_src.splitlines(keepends=True)
        out2.insert(last_imp.end_lineno, add)
        new_src = "".join(out2)
        print(f"  added contracts import to {relpath}")

    # syntax check
    ast.parse(new_src)
    if APPLY:
        open(path, "w").write(new_src)
    print(f"  {'WROTE' if APPLY else 'DRY-RUN ok'} {relpath} (+{len(inserts)} specs)")
    return len(inserts)


if __name__ == "__main__":
    by_file = {}
    for rp, cls, canon in TARGETS:
        by_file.setdefault(rp, []).append((cls, canon))
    total = 0
    for rp, ft in by_file.items():
        print(rp)
        total += patch_file(rp, ft)
    for rp, fac, cls, canons in DYNAMIC_TARGETS:
        print(f"{rp}  [dynamic factory {fac}]")
        total += patch_dynamic(rp, fac, cls, canons)
    print(f"\nTOTAL specs planned: {total}   APPLY={APPLY}")
