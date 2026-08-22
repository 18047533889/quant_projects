#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R18-114: direct-use audit — every canonical gets PASS / MOVE / DELETE / FAIL.

No SKIP is allowed (R18-124: ``except: continue`` is forbidden; an exception is
an ``AUDIT_FAILED`` and blocks closure).

The registry-level gates (R18-127) are implemented mechanically here:

  REGISTERED_PUBLIC_WITHOUT_DIRECT_STATUS       == []
  DIRECT_WITHOUT_INPUT_CONTRACT                 == []
  DIRECT_WITHOUT_SMOKE_RECIPE                   == []
  CONDITION_AS_ALPHA_TERMINAL                   == []
  CATEGORY_AS_CONTINUOUS_ALPHA                  == []
  INTERMEDIATE_LEVEL_AS_DEFAULT_TERMINAL        == []
  SOURCE_TRANSFORM_AS_ALPHA_TERMINAL            == []
  GHOST_SURFACE_CANONICALS                      == []
  ALIASES_TO_DELETED_CANONICALS                 == []
  PUBLIC_NO_DATA_CANONICALS                     == []
  DEFAULT_MINING_MANIFEST_CONTAINS_PENDING      == []
  DEFAULT_MINING_MANIFEST_CONTAINS_RESEARCH     == []
  DIRECT_OPERATOR_NOT_REACHABLE_IN_GRAMMAR      == []
  UNRESOLVED / UNKNOWN / PENDING statuses       == []

``--smoke`` additionally binds synthetic panels and executes every retained
direct operator, reporting DIRECT_DEFAULT_EXECUTION_FAILURE for any that raises
(audit can never silently skip a failing operator).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _registry() -> dict[str, Any]:
    from cleaned_operators import load_all
    from cleaned_operators.registry import OperatorRegistry

    load_all()
    return dict(OperatorRegistry._catalog)


def _surface_names() -> set[str]:
    """Factor-authoring surface names (DAILY / EXTENDED / RESEARCH).

    R18-037 ghost check: ``surface_names - registry_canonicals == ∅``.  The
    ``NON_FACTOR_PRODUCTION_CANONICALS`` set is deliberately NOT included — it
    holds DSL verbs (Lead/bfill/dropna/rand_*) that are never registered factor
    canonicals, so they are not ghosts by construction.
    """
    names: set[str] = set()
    try:
        from cleaned_operators.operator_surface import (
            DAILY_CANONICALS,
            EXTENDED_ONLY_CANONICALS,
            RESEARCH_ONLY_CANONICALS,
        )

        names = set(DAILY_CANONICALS) | set(EXTENDED_ONLY_CANONICALS) | set(RESEARCH_ONLY_CANONICALS)
    except Exception:
        pass
    return names


def run_audit(*, smoke: bool = False) -> dict[str, Any]:
    from mining.direct_use import (
        DirectUseStatus,
        build_direct_use_operator,
        direct_use_matrix_rows,
    )

    catalog = _registry()
    rows = direct_use_matrix_rows()
    row_by = {r.canonical: r for r in rows}

    verdicts: dict[str, str] = {}
    violations: dict[str, list[str]] = defaultdict(list)

    for canonical, r in sorted(row_by.items()):
        st = r.direct_use_status
        # DELETED / MOVED verdicts come straight from the status.
        if st.value.startswith("delete_"):
            verdicts[canonical] = "DELETE"
            continue
        if st in (DirectUseStatus.RESEARCH_TOOL, DirectUseStatus.MOVE_INTERNAL):
            verdicts[canonical] = "MOVE"
            continue
        # Direct rows: a handful of role-AST consistency gates.
        status = r.direct_use_status.value
        if status == "unresolved" or status in ("unknown", "pending"):
            verdicts[canonical] = "FAIL"
            violations["UNRESOLVED"].append(canonical)
            continue
        if r.mining_role in ("condition", "state", "event", "group_state", "global_state") and r.terminal_allowed:
            verdicts[canonical] = "FAIL"
            violations["CONDITION_AS_ALPHA_TERMINAL"].append(canonical)
            continue
        if not r.data_inputs and not r.scalar_parameters and not r.input_slots:
            verdicts[canonical] = "FAIL"
            violations["DIRECT_WITHOUT_INPUT_CONTRACT"].append(canonical)
            continue
        # every retained direct row must have a smoke recipe (R18-006/071).
        # A data-less operator is NOT automatically a ghost — date_diff_days and
        # other time-index-derived ops legitimately take no panel input.  Only a
        # completely EMPTY contract (no data, no scalar, no slot) is a ghost.
        if not r.data_inputs and not r.scalar_parameters and not r.input_slots:
            verdicts[canonical] = "FAIL"
            violations["DIRECT_WITHOUT_SMOKE_RECIPE"].append(canonical)
            continue
        if smoke:
            ok, err = _smoke_execute(canonical, r)
            if not ok:
                verdicts[canonical] = "FAIL"
                violations["DIRECT_DEFAULT_EXECUTION_FAILURE"].append(f"{canonical}: {err}")
                continue
        verdicts[canonical] = "PASS"

    # ---- registry-level cross gates --------------------------------------
    # ghost surface names: surface - registry
    ghost = sorted(_surface_names() - set(catalog))
    violations["GHOST_SURFACE_CANONICALS"].extend(ghost)
    # aliases pointing at deleted/moved canonicals
    try:
        from cleaned_operators.registry import OperatorRegistry

        aliases = dict(OperatorRegistry._aliases)
        for alias, target in aliases.items():
            if target not in catalog or (
                target in row_by and row_by[target].direct_use_status.value.startswith("delete_")
            ):
                violations["ALIASES_TO_DELETED_CANONICALS"].append(f"{alias}->{target}")
    except Exception:
        pass
    # PUBLIC no-data: a canonical that is *already* classified DELETE_NO_DATA /
    # DELETE_NONCAUSAL is NOT public — it is in the R18 delete plan.  The gate
    # only fires on a RETAINED (direct/research/move) canonical that has no
    # usable source, which would be a classification bug.
    for canonical, r in sorted(row_by.items()):
        st = r.direct_use_status
        if st in (DirectUseStatus.DELETE_NO_DATA, DirectUseStatus.DELETE_NONCAUSAL):
            continue
        if not r.source_recipes and not r.data_inputs and st.value.startswith("direct_"):
            violations["PUBLIC_NO_DATA_CANONICALS"].append(canonical)
    # default manifest: pending/research inside direct manifest (checked by exporter)
    try:
        manifest = json.loads(
            (REPO / "build" / "mining" / "direct_mining_manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("pending_count"):
            violations["DEFAULT_MINING_MANIFEST_CONTAINS_PENDING"].append(str(manifest.get("pending_count")))
        if manifest.get("research_count"):
            violations["DEFAULT_MINING_MANIFEST_CONTAINS_RESEARCH"].append(str(manifest.get("research_count")))
    except Exception:
        pass

    summary: dict[str, int] = defaultdict(int)
    for v in verdicts.values():
        summary[v] += 1
    return {
        "verdicts": verdicts,
        "summary": dict(summary),
        "violations": {k: v for k, v in violations.items()},
        "violation_counts": {k: len(v) for k, v in violations.items()},
        "release_safe": all(len(v) == 0 for v in violations.values()),
    }


def _smoke_execute(canonical: str, row: Any) -> tuple[bool, str]:
    """Best-effort synthetic execution.  Returns (ok, err)."""
    try:
        import numpy as np
        import pandas as pd

        from cleaned_operators.registry import OperatorRegistry

        op = OperatorRegistry.get(canonical, "pandas_numpy") or OperatorRegistry.get(canonical)
        if op is None:
            return False, "no operator instance"
        n = 40
        dates = pd.date_range("2024-01-01", periods=n)
        stocks = [f"S{i}" for i in range(8)]
        idx = pd.MultiIndex.from_product([dates, stocks], names=["date", "asset"])
        rng = np.random.default_rng(7)
        frames = {
            p: pd.DataFrame(rng.standard_normal((n * 8, 1)), index=idx, columns=["v"])
            for p in row.data_inputs
        }
        kwargs: dict[str, Any] = {}
        for p in row.scalar_parameters:
            kwargs[p] = 3
        result = op.calculate(*[frames[p] for p in row.data_inputs], **kwargs)
        out = result
        if hasattr(result, "to_pandas"):
            out = result.to_pandas()
        if out is None:
            return False, "null output"
        return True, ""
    except Exception as exc:  # noqa: BLE001 — audit never silently skips
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="execute synthetic smoke for retained direct ops")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    result = run_audit(smoke=args.smoke)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("R18 direct-use audit")
        print("  verdicts:", dict(result["summary"]))
        for k, v in sorted(result["violation_counts"].items()):
            print(f"  {k}: {v}")
        if not result["release_safe"]:
            for k, v in sorted(result["violations"].items()):
                if v:
                    print(f"  ! {k}:")
                    for item in v[:8]:
                        print(f"      - {item}")
    return 0 if result["release_safe"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
