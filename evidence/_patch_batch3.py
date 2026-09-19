# -*- coding: utf-8 -*-
"""Batch 3 of the "real polars / DuckDB backend coverage" work.

Declares ``PhysicalImplementationSpec(execution_kind=POLARS_NATIVE_EXPR)`` for 13
operators whose registered polars slot is a GENUINE polars expression kernel but
which carried no explicit spec, so ``canonical_polars_kind(production_mode=True)``
reported UNSUPPORTED (fail-closed) and the polars_long production channel stayed
closed for them.

Honesty rules (same contract as batch 1 / batch 2):
  * Only kernels whose body builds ``pl.Expr`` (pl.col / with_columns / group_by)
    and whose RUNTIME marshal probe records ZERO ``pl.DataFrame.to_pandas`` calls
    are declared.  The probe evidence is produced separately by
    ``evidence/_verify_batch_specs.py``.
  * ``supports_lazy`` / ``supports_streaming`` are left at their ``False`` defaults:
    these kernels accept a materialised eager ``pl.DataFrame`` and return an eager
    frame, so claiming lazy execution would overstate them.
  * NO kernel body is modified by this patch -- declarations only.
  * ``implementation_source_hash`` keeps the stable plaintext identity-label
    convention used by the two previous batches.

This script does not write the repository.  It emits a unified diff; the caller
applies it with ``git apply`` (dry-run first) so the working tree is only ever
touched through the normal patch path.
"""
from __future__ import annotations

import difflib
import os
import sys
import textwrap

REPO = "/home/sunhaiwei/quant_projects"

CONTRACTS_IMPORT = (
    "from factor_engine.backend.contracts import (\n"
    "    ExecutionKind,\n"
    "    PhysicalImplementationSpec,\n"
    ")\n"
)

SPEC_FN = '''def _batch3_native_spec(canonical: str, kernel: str) -> PhysicalImplementationSpec:
    """Explicit execution-kind contract for a genuine polars expression kernel.

    Built from the batch-3 evidence: the kernel body is ``pl.Expr`` construction
    (no pandas round-trip) and the runtime marshal probe records zero
    ``pl.DataFrame.to_pandas`` calls on real daily data.  Eager panel API, so
    ``supports_lazy`` / ``supports_streaming`` stay False.
    """
    return PhysicalImplementationSpec(
        canonical=canonical,
        backend="polars",
        execution_kind=ExecutionKind.POLARS_NATIVE_EXPR,
        materializes_full_panel=True,
        supports_nulls=True,
        supports_nan=True,
        supports_inf=True,
        implementation_source_hash=f"__LABEL__:{kernel}:v1",
        emitter_identity=f"polars_expr:{canonical}",
        kernel_identity=f"__LABEL__:{kernel}",
        parameter_domain_hash=f"{canonical}:declared:v1",
        semantic_contract_hash=f"{canonical}:polars_native_expr:v1",
        notes=_BATCH3_NOTE,
    )

'''

NOTE_CONST = '''_BATCH3_NOTE = (
    "Genuine polars expression kernel (pl.Expr over columns, no pandas round-trip); "
    "runtime marshal probe on real daily data records 0 pl.DataFrame.to_pandas "
    "calls. Eager panel API only: no lazy/streaming or production-parity claim."
)

'''

BLOCK_HEADER = (
    "# ---------------------------------------------------------------------------\n"
    "# R57 backend-coverage batch 3 — explicit execution-kind declarations.\n"
    "# These kernels are genuine polars expressions (pl.col / with_columns /\n"
    "# group_by over pl.Expr).  Previously they had no _physical_spec, so\n"
    "# canonical_polars_kind(production_mode=True) failed closed to UNSUPPORTED.\n"
    "# ---------------------------------------------------------------------------\n"
)


def helper_block(label: str, canon_kernel_pairs=None, extra_sets=(), emit_import=True) -> str:
    out = BLOCK_HEADER
    if canon_kernel_pairs is not None and emit_import:
        out += CONTRACTS_IMPORT + "\n"
    out += NOTE_CONST + "\n" + SPEC_FN.replace("__LABEL__", label)
    if canon_kernel_pairs is not None:
        out += "\n_NATIVE_SPECS: dict[str, PhysicalImplementationSpec] = {\n"
        out += "    _c: _batch3_native_spec(_c, _k)\n    for _c, _k in (\n"
        for c, k in canon_kernel_pairs:
            out += f'        ("{c}", "{k}"),\n'
        out += "    )\n}\n"
    for name, members in extra_sets:
        out += f'\n{name} = frozenset({sorted(members)!r})\n'
    return out


def read(rel: str) -> str:
    with open(os.path.join(REPO, rel), encoding="utf-8") as fh:
        return fh.read()


def write_patch(rel: str, old: str, new: str, chunks: list) -> None:
    diff = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{rel}",
        tofile=f"b/{rel}",
        n=3,
    )
    chunks.append("".join(diff))


def replace_once(text: str, old: str, new: str, what: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{what}: expected exactly 1 occurrence, found {n}")
    return text.replace(old, new)


# ---------------------------------------------------------------------------
# A. factory modules: inject the spec into the dynamically built class dict
# ---------------------------------------------------------------------------

FACTORY_TARGETS = [
    dict(
        rel="factor_engine/cleaned_operators/price_volume/polars_liquidity_v2.py",
        label="price_volume.polars_liquidity_v2",
        anchor="def _register(name: str, params:",
        key="name",
        pairs=[
            ("price_impact", "PolarsLiquidityV2_price_impact"),
            ("turnover_zscore", "PolarsLiquidityV2_turnover_zscore"),
            ("relative_volume", "PolarsLiquidityV2_relative_volume"),
            ("amihud_illiquidity", "PolarsLiquidityV2_amihud_illiquidity"),
        ],
    ),
    dict(
        rel="factor_engine/cleaned_operators/technical/polars_tech_misc.py",
        label="technical.polars_tech_misc",
        anchor="def _register(name: str, params:",
        key="name",
        pairs=[("efficiency_ratio", "PolarsTechMisc_efficiency_ratio")],
    ),
    dict(
        rel="factor_engine/cleaned_operators/fundamental/polars_fundamental.py",
        label="fundamental.polars_fundamental",
        anchor="def _register(name: str, params:",
        key="name",
        pairs=[("fin_quarter_from_cumulative", "PolarsFundamental_fin_quarter_from_cumulative")],
    ),
    dict(
        rel="factor_engine/cleaned_operators/intraday/polars_next_stage.py",
        label="intraday.polars_next_stage",
        anchor="def _mk(canonical: str, description: str, params: list[str], fn,",
        key="canonical",
        pairs=[
            ("intra_realized_skewness", "IntradayPolars_intra_realized_skewness"),
            ("intra_realized_kurtosis", "IntradayPolars_intra_realized_kurtosis"),
        ],
    ),
    dict(
        rel="factor_engine/cleaned_operators/intraday/polars_intraday_full.py",
        label="intraday.polars_intraday_full",
        # the shared ``_mk`` factory is first *called* long before the mid-file
        # contracts import, so the spec table must sit ahead of the first call
        # and the import has to live at the top of the module.
        anchor='_mk("intra_segment_return",',
        key="canonical",
        emit_import=False,
        drop_import_line=(
            "from factor_engine.backend.contracts import "
            "ExecutionKind, PhysicalImplementationSpec\n"
        ),
        top_import_after=(
            "from factor_engine.cleaned_operators.base_polars import "
            "OperatorMetadata, SeriesOperator, register_operator\n"
        ),
        pairs=[("intra_realized_variance", "IntradayPolarsFull_intra_realized_variance")],
    ),
]

CLASS_DICT_OLD = (
    '        {"metadata": metadata, "_calculate_series": _calculate_series, '
    '"__module__": __name__},'
)

CLASS_DICT_NEW = (
    "        {\n"
    '            "metadata": metadata,\n'
    '            "_calculate_series": _calculate_series,\n'
    '            "__module__": __name__,\n'
    '            **({"_physical_spec": _NATIVE_SPECS[__KEY__]}\n'
    "               if __KEY__ in _NATIVE_SPECS else {}),\n"
    "        },"
)


def patch_factory(spec: dict, chunks: list) -> int:
    rel = spec["rel"]
    src = read(rel)
    new = src
    # 0) optional: hoist a mid-file contracts import to the top import block
    if spec.get("drop_import_line"):
        new = replace_once(new, spec["drop_import_line"], "", f"{rel}: drop mid-file import")
    if spec.get("top_import_after"):
        new = replace_once(
            new,
            spec["top_import_after"],
            spec["top_import_after"] + CONTRACTS_IMPORT,
            f"{rel}: top-level contracts import",
        )
    # 1) helper block before the factory definition (or before the first call site)
    block = helper_block(
        spec["label"], spec["pairs"], emit_import=spec.get("emit_import", True)
    )
    idx = new.find(spec["anchor"])
    if idx < 0:
        raise SystemExit(f"{rel}: anchor {spec['anchor']!r} not found")
    line_start = new.rfind("\n", 0, idx) + 1
    new = new[:line_start] + block + "\n\n" + new[line_start:]
    # 2) class dict gains the optional spec
    new = replace_once(
        new,
        CLASS_DICT_OLD,
        CLASS_DICT_NEW.replace("__KEY__", spec["key"]),
        f"{rel}: class dict",
    )
    if src == new:
        raise SystemExit(f"{rel}: no change produced")
    write_patch(rel, src, new, chunks)
    print(f"  PLAN factory {rel}: {len(spec['pairs'])} spec(s)")
    return len(spec["pairs"])


# ---------------------------------------------------------------------------
# B. plain class-body modules: one spec attribute inside the class body
# ---------------------------------------------------------------------------

CLASS_TARGETS = [
    dict(
        rel="factor_engine/cleaned_operators/common/polars_ops.py",
        label="common.polars_ops",
        anchor='@register_operator(name="ts_pct",',
        cls="class TSPctPolars(SeriesOperator):",
        canonical="ts_pct",
        kernel="TSPctPolars",
        want_import=False,
    ),
    dict(
        rel="factor_engine/cleaned_operators/common/polars_daily_native.py",
        label="common.polars_daily_native",
        anchor='@register_operator(\n    name="signed_sqrt",',
        cls="class SignedSqrtNative(SeriesOperator):",
        canonical="signed_sqrt",
        kernel="SignedSqrtNative",
        want_import=False,
    ),
]


def patch_class(spec: dict, chunks: list) -> int:
    rel = spec["rel"]
    src = read(rel)
    new = src
    block = helper_block(spec["label"])
    idx = src.find(spec["anchor"])
    if idx < 0:
        raise SystemExit(f"{rel}: anchor not found")
    line_start = src.rfind("\n", 0, idx) + 1
    new = new[:line_start] + block + "\n" + new[line_start:]
    # spec attribute as the first statement after the class docstring
    cis = new.find(spec["cls"])
    if cis < 0:
        raise SystemExit(f"{rel}: class {spec['cls']!r} not found")
    ms = new.find("    metadata = OperatorMetadata(", cis)
    if ms < 0:
        raise SystemExit(f"{rel}: metadata assignment not found after {spec['cls']!r}")
    attr = (
        f'    _physical_spec = _batch3_native_spec("{spec["canonical"]}", '
        f'"{spec["kernel"]}")\n'
    )
    new = new[:ms] + attr + new[ms:]
    write_patch(rel, src, new, chunks)
    print(f"  PLAN class {rel}: {spec['canonical']}")
    return 1


# ---------------------------------------------------------------------------
# C. polars_auto.py: the shared _register_unary factory + cleanup of the
#    duplicated import/constant block left behind by the batch-2 injector
# ---------------------------------------------------------------------------

AUTO_REL = "factor_engine/cleaned_operators/common/polars_auto.py"
AUTO_DUP = '''from factor_engine.backend.contracts import (
    ExecutionKind, PhysicalImplementationSpec,
)

DYN_NOTE__register_compare = 'Genuine polars expression kernel (pl.col comparison + with_columns); runtime probe shows 0 pl.DataFrame.to_pandas calls. Declared to connect the polars_long channel (execution_kind was previously absent).'

'''

AUTO_OLD = "        _calculate_series = _unary_calc(expr_fn)\n"
AUTO_NEW = (
    "        if canonical in _NATIVE_UNARY_SPEC_CANONICALS:\n"
    "            _physical_spec = _batch3_native_spec(\n"
    "                canonical, f\"{canonical.title().replace('_', '')}PolarsAuto\"\n"
    "            )\n"
    "        _calculate_series = _unary_calc(expr_fn)\n"
)


def patch_auto(chunks: list) -> int:
    src = read(AUTO_REL)
    new = src
    # (1) drop the duplicated contracts import + duplicated note constant
    if new.count(AUTO_DUP) != 2:
        raise SystemExit(f"{AUTO_REL}: expected 2 copies of the batch-2 injector "
                         f"block, found {new.count(AUTO_DUP)}")
    first = new.find(AUTO_DUP)
    second = new.find(AUTO_DUP, first + 1)
    new = new[:second] + new[second + len(AUTO_DUP):]
    # (2) helper block ahead of the shared unary factory
    anchor = "def _register_unary(\n"
    idx = new.find(anchor)
    if idx < 0:
        raise SystemExit(f"{AUTO_REL}: _register_unary not found")
    line_start = new.rfind("\n", 0, idx) + 1
    block = helper_block(
        "common.polars_auto",
        None,
        extra_sets=[("_NATIVE_UNARY_SPEC_CANONICALS", ("sigmoid", "log_abs"))],
    )
    new = new[:line_start] + block + "\n\n" + new[line_start:]
    # (3) declare inside the per-canonical unary class body
    new = replace_once(new, AUTO_OLD, AUTO_NEW, f"{AUTO_REL}: _UnaryPolars body")
    write_patch(AUTO_REL, src, new, chunks)
    print(f"  PLAN auto {AUTO_REL}: sigmoid, log_abs (+ duplicate-import cleanup)")
    return 2


# ---------------------------------------------------------------------------
# D. evidence harness: allow a distinct output file per batch
# ---------------------------------------------------------------------------

VER_REL = "evidence/_verify_batch_specs.py"
VER_OLD = '    out = f"{REPO}/evidence/_verify_batch_specs.json"\n'
VER_NEW = ('    out = os.environ.get("VER_OUT") or '
           'f"{REPO}/evidence/_verify_batch_specs.json"\n')

# The batch-2 harness tested the absent-default sentinel with
# ``type(d).__name__ != "MISSING"``, but the live sentinel type is
# ``_MissingDefaultType`` — so the guard never fired and parameters such as
# ``window`` were passed through as the sentinel, aborting the run with
# OperatorParameterError.  Match the sentinel by substring instead.
KW_OLD = '        if d is not None and type(d).__name__ != "MISSING":\n'
KW_NEW = '        if d is not None and "missing" not in type(d).__name__.lower():\n'

# The batch-2 harness compared the two wide outputs as raw numpy blocks, i.e. by
# integer position.  A pivoting kernel is free to emit its columns in pivot
# discovery order rather than the input panel's order, so that comparison can
# match different symbols against each other.  (Found while verifying the
# intraday kernels, where it produced a spurious max-abs-dev of 26 and a false
# NaN-mask mismatch.)  Align by column name instead.
ARRAY_OLD = '''def as_array(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.to_numpy(dtype=float)
    if isinstance(obj, pl.DataFrame):
        return obj.select([c for c in obj.columns if c != "date"]).to_numpy().astype(float)
    raise TypeError(type(obj))
'''
ARRAY_NEW = '''def as_array(obj):
    """Column-name aligned view of a wide result.

    A pivoting kernel orders its output columns by pivot discovery order, which
    need not equal the input panel's column order; comparing the raw numpy blocks
    positionally would then compare different symbols against each other.  Sort
    both sides by column name so the comparison is symbol-for-symbol.
    """
    if isinstance(obj, pd.DataFrame):
        return obj.reindex(columns=sorted(map(str, obj.columns))).to_numpy(dtype=float)
    if isinstance(obj, pl.DataFrame):
        cols = sorted(c for c in obj.columns if c != "date")
        return obj.select(cols).to_numpy().astype(float)
    raise TypeError(type(obj))
'''


def patch_verifier(chunks: list) -> int:
    src = read(VER_REL)
    new = replace_once(src, VER_OLD, VER_NEW, f"{VER_REL}: output path")
    new = replace_once(new, KW_OLD, KW_NEW, f"{VER_REL}: missing-default sentinel")
    new = replace_once(new, ARRAY_OLD, ARRAY_NEW, f"{VER_REL}: column-name alignment")
    write_patch(VER_REL, src, new, chunks)
    print(f"  PLAN verifier {VER_REL}: VER_OUT + sentinel fix + column-name alignment")
    return 0


def main() -> int:
    chunks: list[str] = []
    total = 0
    for spec in FACTORY_TARGETS:
        total += patch_factory(spec, chunks)
    for spec in CLASS_TARGETS:
        total += patch_class(spec, chunks)
    total += patch_auto(chunks)
    patch_verifier(chunks)
    patch = "".join(chunks)
    if "--emit" in sys.argv:
        path = sys.argv[sys.argv.index("--emit") + 1]
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(patch)
        print(f"\nEMITTED {path} ({len(patch.splitlines())} lines)")
    else:
        sys.stdout.write(patch)
    print(f"\nTOTAL operator specs planned: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
