#!/usr/bin/env python3
"""Profile cleaned_operators.load_all() bootstrap performance."""
import time
import sys
from typing import Callable
from dataclasses import dataclass


@dataclass
class TimingRecord:
    """Individual timing measurement."""
    operation: str
    duration_ms: float
    parent: str = ""


class BootstrapProfiler:
    """Instrument the bootstrap process with timing probes."""

    def __init__(self):
        self.timings: list[TimingRecord] = []
        self.start_time = time.perf_counter()

    def record(self, operation: str, duration_ms: float, parent: str = ""):
        """Record a timing measurement."""
        self.timings.append(TimingRecord(operation, duration_ms, parent))

    def time_operation(self, name: str, func: Callable, parent: str = ""):
        """Time a single operation."""
        start = time.perf_counter()
        result = func()
        duration_ms = (time.perf_counter() - start) * 1000
        self.record(name, duration_ms, parent)
        return result

    def report(self):
        """Generate detailed timing report."""
        total_ms = (time.perf_counter() - self.start_time) * 1000

        print("\n" + "="*80)
        print("BOOTSTRAP PERFORMANCE PROFILE")
        print("="*80)
        print(f"\nTotal bootstrap time: {total_ms:.1f} ms ({total_ms/1000:.2f}s)")
        print(f"Number of operations tracked: {len(self.timings)}")

        # Sort by duration
        sorted_timings = sorted(self.timings, key=lambda x: x.duration_ms, reverse=True)

        print("\n" + "-"*80)
        print("TOP 20 SLOWEST OPERATIONS")
        print("-"*80)
        print(f"{'Operation':<60} {'Time (ms)':>10} {'%':>8}")
        print("-"*80)

        for i, timing in enumerate(sorted_timings[:20], 1):
            pct = (timing.duration_ms / total_ms) * 100
            op_display = timing.operation[:58] if len(timing.operation) > 58 else timing.operation
            print(f"{i:2d}. {op_display:<57} {timing.duration_ms:10.1f} {pct:7.1f}%")

        # Group by category
        print("\n" + "-"*80)
        print("TIMING BY CATEGORY")
        print("-"*80)

        categories = {}
        for timing in self.timings:
            parent = timing.parent or "uncategorized"
            if parent not in categories:
                categories[parent] = []
            categories[parent].append(timing)

        for category in sorted(categories.keys()):
            timings_in_cat = categories[category]
            cat_total = sum(t.duration_ms for t in timings_in_cat)
            pct = (cat_total / total_ms) * 100
            print(f"\n{category}: {cat_total:.1f} ms ({pct:.1f}%)")
            if cat_total > 100:  # Only show details for categories > 100ms
                for timing in sorted(timings_in_cat, key=lambda x: x.duration_ms, reverse=True)[:5]:
                    print(f"  - {timing.operation}: {timing.duration_ms:.1f} ms")


# Monkey-patch the module loading to instrument it
profiler = BootstrapProfiler()


def profile_bootstrap():
    """Run instrumented bootstrap."""
    import importlib
    from cleaned_operators import _load_all_impl, BOOTSTRAP_MODULE_SPECS, BootstrapModuleRole

    # Track overall phases
    phase_start = time.perf_counter()

    # Evidence delta
    start = time.perf_counter()
    from backend.evidence_delta import install_evidence_delta
    install_evidence_delta()
    profiler.record("install_evidence_delta", (time.perf_counter() - start) * 1000, "bootstrap_phases")

    # Production signature
    start = time.perf_counter()
    from backend.production_signature_v2 import apply_production_signature_v2
    apply_production_signature_v2()
    profiler.record("apply_production_signature_v2", (time.perf_counter() - start) * 1000, "bootstrap_phases")

    # Registration audit
    start = time.perf_counter()
    from cleaned_operators.registration_audit import install_registration_audit
    install_registration_audit()
    profiler.record("install_registration_audit", (time.perf_counter() - start) * 1000, "bootstrap_phases")

    # Bootstrap module spec check
    start = time.perf_counter()
    from cleaned_operators import check_bootstrap_module_specs
    check_bootstrap_module_specs()
    profiler.record("check_bootstrap_module_specs", (time.perf_counter() - start) * 1000, "bootstrap_phases")

    # Module loading (the big one)
    module_load_start = time.perf_counter()
    module_count = 0
    slow_modules = []

    for spec in BOOTSTRAP_MODULE_SPECS:
        if spec.role is BootstrapModuleRole.RESEARCH_EXTENSION:
            continue

        start = time.perf_counter()
        try:
            __import__(spec.module, fromlist=["*"])
            duration = (time.perf_counter() - start) * 1000
            profiler.record(f"import {spec.module}", duration, "module_imports")
            module_count += 1

            if duration > 100:  # Track modules taking > 100ms
                slow_modules.append((spec.module, duration))
        except ImportError as exc:
            from cleaned_operators.common._polars_bridge import should_skip_optional_import_error
            if not should_skip_optional_import_error(exc, spec.module):
                raise

    module_load_total = (time.perf_counter() - module_load_start) * 1000
    profiler.record(f"ALL MODULE IMPORTS ({module_count} modules)", module_load_total, "bootstrap_phases")

    # Semantic certification
    start = time.perf_counter()
    from cleaned_operators.semantic_certification import snapshot_registered_statuses, stamp_compatibility_metadata
    snapshot_registered_statuses()
    stamp_compatibility_metadata()
    profiler.record("semantic_certification", (time.perf_counter() - start) * 1000, "bootstrap_phases")

    # Deduplication
    start = time.perf_counter()
    from cleaned_operators._dedupe import apply_operator_deduplication
    apply_operator_deduplication()
    profiler.record("apply_operator_deduplication", (time.perf_counter() - start) * 1000, "bootstrap_phases")

    # Operator overhaul
    start = time.perf_counter()
    from cleaned_operators.operator_overhaul import finalize_operator_overhaul
    finalize_operator_overhaul()
    profiler.record("finalize_operator_overhaul", (time.perf_counter() - start) * 1000, "bootstrap_phases")

    # LQTP policy
    start = time.perf_counter()
    from cleaned_operators.lqtp_policy_patch import apply_lqtp_policy_patch
    apply_lqtp_policy_patch()
    profiler.record("apply_lqtp_policy_patch", (time.perf_counter() - start) * 1000, "bootstrap_phases")

    # Research factor enable
    start = time.perf_counter()
    from cleaned_operators.research_factor_enable import enable_research_factor_runtime
    enable_research_factor_runtime()
    profiler.record("enable_research_factor_runtime", (time.perf_counter() - start) * 1000, "bootstrap_phases")

    # Layer governance
    start = time.perf_counter()
    from cleaned_operators.layer_governance import finalize_layer_governance
    finalize_layer_governance()
    profiler.record("finalize_layer_governance", (time.perf_counter() - start) * 1000, "bootstrap_phases")

    # Post governance
    start = time.perf_counter()
    from cleaned_operators.layer_governance_post import apply_post_governance
    apply_post_governance()
    profiler.record("apply_post_governance", (time.perf_counter() - start) * 1000, "bootstrap_phases")

    # Production hardening
    start = time.perf_counter()
    from cleaned_operators.production_hardening import apply_production_hardening
    apply_production_hardening()
    profiler.record("apply_production_hardening", (time.perf_counter() - start) * 1000, "bootstrap_phases")

    # Fiscal strict
    fiscal_start = time.perf_counter()
    from cleaned_operators import fiscal_strict, fiscal_event_ops, replace_backend, record_backend_replacement_after
    for _canonical in (
        "period_lag", "period_change", "period_average", "period_cagr",
        "quarter_from_cumulative", "ttm_from_quarterly", "ttm_from_cumulative",
        "yoy_by_period",
    ):
        for _backend in ("pandas_numpy", "polars"):
            _migration = replace_backend(_canonical, _backend, reason="test", source="test")
    fiscal_strict.register()
    fiscal_event_ops.register()
    profiler.record("fiscal_strict_registration", (time.perf_counter() - fiscal_start) * 1000, "bootstrap_phases")

    # Polars contracts
    start = time.perf_counter()
    from cleaned_operators.overhaul.cleanup import _attach_explicit_polars_contracts
    _attach_explicit_polars_contracts()
    profiler.record("_attach_explicit_polars_contracts", (time.perf_counter() - start) * 1000, "bootstrap_phases")

    # Registration audit finalize
    start = time.perf_counter()
    from cleaned_operators.registration_audit import finalize_registration_audit
    finalize_registration_audit()
    profiler.record("finalize_registration_audit", (time.perf_counter() - start) * 1000, "bootstrap_phases")

    # SQL backends
    start = time.perf_counter()
    from backend.sql_pushdown.sql_registry import register_sql_backends
    register_sql_backends()
    profiler.record("register_sql_backends", (time.perf_counter() - start) * 1000, "bootstrap_phases")

    # Polars gap coverage
    start = time.perf_counter()
    from cleaned_operators.polars_gap_coverage import register_polars_gap_coverage
    register_polars_gap_coverage()
    profiler.record("register_polars_gap_coverage", (time.perf_counter() - start) * 1000, "bootstrap_phases")

    # Fiscal SQL v2
    start = time.perf_counter()
    from backend.sql_pushdown.fiscal_v2 import apply_fiscal_sql_v2
    apply_fiscal_sql_v2()
    profiler.record("apply_fiscal_sql_v2", (time.perf_counter() - start) * 1000, "bootstrap_phases")

    # Evidence certification overlay
    start = time.perf_counter()
    from cleaned_operators.production_certification_overlay import apply_evidence_certification_overlay
    apply_evidence_certification_overlay()
    profiler.record("apply_evidence_certification_overlay", (time.perf_counter() - start) * 1000, "bootstrap_phases")

    # Contract hardening
    start = time.perf_counter()
    from cleaned_operators.contract_hardening import apply_final_contract_hardening
    apply_final_contract_hardening()
    profiler.record("apply_final_contract_hardening", (time.perf_counter() - start) * 1000, "bootstrap_phases")

    # Param role backfill
    start = time.perf_counter()
    from cleaned_operators.param_role_contract import backfill_scalar_roles
    backfill_scalar_roles()
    profiler.record("backfill_scalar_roles", (time.perf_counter() - start) * 1000, "bootstrap_phases")

    # Stateful migration
    start = time.perf_counter()
    from cleaned_operators.stateful_contract_migration import apply_stateful_contract_migration
    apply_stateful_contract_migration()
    profiler.record("apply_stateful_contract_migration", (time.perf_counter() - start) * 1000, "bootstrap_phases")

    # Axis effects
    start = time.perf_counter()
    from ir.types import register_axis_effects_for_surface
    register_axis_effects_for_surface()
    profiler.record("register_axis_effects_for_surface", (time.perf_counter() - start) * 1000, "bootstrap_phases")

    print(f"\nSlowest modules (> 100ms):")
    for mod, dur in sorted(slow_modules, key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {mod}: {dur:.1f} ms")


if __name__ == "__main__":
    print("Starting bootstrap profiling...")
    print("This will take ~95 seconds as observed...")

    try:
        profile_bootstrap()
        profiler.report()
    except Exception as e:
        print(f"\nERROR during profiling: {e}")
        import traceback
        traceback.print_exc()
        profiler.report()  # Still show what we measured
        sys.exit(1)
