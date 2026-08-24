#!/usr/bin/env python3
"""Profile cleaned_operators.load_all() with complete instrumentation."""
import time
import sys
from contextlib import contextmanager


class TimingCollector:
    """Collect timing measurements."""

    def __init__(self):
        self.timings = []
        self.start = time.perf_counter()

    @contextmanager
    def measure(self, name):
        """Context manager to measure a block."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            self.timings.append((name, duration_ms))
            if duration_ms > 1000:  # Print slow operations immediately
                print(f"  [{duration_ms:.1f}ms] {name}", flush=True)

    def report(self):
        """Generate report."""
        total_ms = (time.perf_counter() - self.start) * 1000
        print("\n" + "="*80)
        print(f"TOTAL BOOTSTRAP TIME: {total_ms:.1f} ms ({total_ms/1000:.2f}s)")
        print("="*80)

        sorted_timings = sorted(self.timings, key=lambda x: x[1], reverse=True)

        print("\nTOP 20 SLOWEST OPERATIONS:")
        print("-"*80)
        for i, (name, duration) in enumerate(sorted_timings[:20], 1):
            pct = (duration / total_ms) * 100
            print(f"{i:2d}. {name[:70]:<70} {duration:8.1f}ms {pct:5.1f}%")

        # Find operations over 5 seconds
        slow_ops = [(n, d) for n, d in self.timings if d > 5000]
        if slow_ops:
            print("\n" + "!"*80)
            print("CRITICAL: Operations taking > 5 seconds:")
            print("!"*80)
            for name, duration in slow_ops:
                print(f"  {name}: {duration:.1f}ms ({duration/1000:.2f}s)")


# Monkey-patch _load_all_impl to add timing
collector = TimingCollector()

original_load_all_impl = None


def instrumented_load_all_impl(*, include_research=True):
    """Instrumented version of _load_all_impl."""
    import importlib

    # Import the module to access its internals
    import factor_engine.cleaned_operators
    from factor_engine.cleaned_operators import (
        BOOTSTRAP_MODULE_SPECS,
        BootstrapModuleRole,
        check_bootstrap_module_specs,
    )
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    global _LOADED

    with collector.measure("install_evidence_delta"):
        from factor_engine.backend.evidence_delta import install_evidence_delta
        install_evidence_delta()

    with collector.measure("apply_production_signature_v2"):
        from factor_engine.backend.production_signature_v2 import apply_production_signature_v2
        apply_production_signature_v2()

    cleaned_operators._SIGNATURE_AUTHORITY_AVAILABLE = True

    with collector.measure("install_registration_audit"):
        from factor_engine.cleaned_operators.registration_audit import install_registration_audit
        install_registration_audit()

    with collector.measure("check_bootstrap_module_specs"):
        check_bootstrap_module_specs()

    # Module loading
    module_load_start = time.perf_counter()
    for spec in BOOTSTRAP_MODULE_SPECS:
        if spec.role is BootstrapModuleRole.RESEARCH_EXTENSION and not include_research:
            continue

        with collector.measure(f"import {spec.module}"):
            try:
                __import__(spec.module, fromlist=["*"])
            except ImportError as exc:
                from factor_engine.cleaned_operators.common._polars_bridge import should_skip_optional_import_error
                if not should_skip_optional_import_error(exc, spec.module):
                    raise

    module_total = (time.perf_counter() - module_load_start) * 1000
    collector.timings.append(("ALL_MODULE_IMPORTS_TOTAL", module_total))

    with collector.measure("snapshot_registered_statuses + stamp_compatibility_metadata"):
        from factor_engine.cleaned_operators.semantic_certification import (
            snapshot_registered_statuses,
            stamp_compatibility_metadata,
        )
        snapshot_registered_statuses()
        stamp_compatibility_metadata()

    with collector.measure("apply_operator_deduplication"):
        from factor_engine.cleaned_operators._dedupe import apply_operator_deduplication
        apply_operator_deduplication()

    with collector.measure("finalize_operator_overhaul"):
        from factor_engine.cleaned_operators.operator_overhaul import finalize_operator_overhaul
        finalize_operator_overhaul()

    with collector.measure("apply_lqtp_policy_patch"):
        from factor_engine.cleaned_operators.lqtp_policy_patch import apply_lqtp_policy_patch
        apply_lqtp_policy_patch()

    with collector.measure("enable_research_factor_runtime"):
        from factor_engine.cleaned_operators.research_factor_enable import enable_research_factor_runtime
        enable_research_factor_runtime()

    with collector.measure("finalize_layer_governance"):
        from factor_engine.cleaned_operators.layer_governance import finalize_layer_governance
        finalize_layer_governance()

    with collector.measure("apply_post_governance"):
        from factor_engine.cleaned_operators.layer_governance_post import apply_post_governance
        apply_post_governance()

    with collector.measure("apply_production_hardening"):
        from factor_engine.cleaned_operators.production_hardening import apply_production_hardening
        apply_production_hardening()

    with collector.measure("fiscal_strict_registration"):
        from factor_engine.cleaned_operators import fiscal_strict, fiscal_event_ops, replace_backend
        for _canonical in (
            "period_lag", "period_change", "period_average", "period_cagr",
            "quarter_from_cumulative", "ttm_from_quarterly", "ttm_from_cumulative",
            "yoy_by_period",
        ):
            for _backend in ("pandas_numpy", "polars"):
                replace_backend(_canonical, _backend, reason="test", source="test")
        fiscal_strict.register()
        fiscal_event_ops.register()

    with collector.measure("_attach_explicit_polars_contracts"):
        from factor_engine.cleaned_operators.overhaul.cleanup import _attach_explicit_polars_contracts
        _attach_explicit_polars_contracts()

    with collector.measure("finalize_registration_audit"):
        from factor_engine.cleaned_operators.registration_audit import finalize_registration_audit
        finalize_registration_audit()

    with collector.measure("register_sql_backends"):
        from factor_engine.backend.sql_pushdown.sql_registry import register_sql_backends
        register_sql_backends()

    with collector.measure("register_polars_gap_coverage"):
        from factor_engine.cleaned_operators.polars_gap_coverage import register_polars_gap_coverage
        register_polars_gap_coverage()

    with collector.measure("apply_fiscal_sql_v2"):
        from factor_engine.backend.sql_pushdown.fiscal_v2 import apply_fiscal_sql_v2
        apply_fiscal_sql_v2()

    with collector.measure("apply_evidence_certification_overlay"):
        from factor_engine.cleaned_operators.production_certification_overlay import apply_evidence_certification_overlay
        apply_evidence_certification_overlay()

    with collector.measure("apply_final_contract_hardening"):
        from factor_engine.cleaned_operators.contract_hardening import apply_final_contract_hardening
        apply_final_contract_hardening()

    with collector.measure("backfill_scalar_roles"):
        from factor_engine.cleaned_operators.param_role_contract import backfill_scalar_roles
        backfill_scalar_roles()

    with collector.measure("apply_stateful_contract_migration"):
        from factor_engine.cleaned_operators.stateful_contract_migration import apply_stateful_contract_migration
        apply_stateful_contract_migration()

    with collector.measure("register_axis_effects_for_surface"):
        from factor_engine.ir.types import register_axis_effects_for_surface
        register_axis_effects_for_surface()

    cleaned_operators._LOADED = True


if __name__ == "__main__":
    print("Starting instrumented bootstrap profiling...")
    print("="*80)

    # Monkey-patch before calling load_all
    import factor_engine.cleaned_operators
    cleaned_operators._load_all_impl = instrumented_load_all_impl

    try:
        print("\nCalling cleaned_operators.load_all(include_research=True)...")
        cleaned_operators.load_all(include_research=True)
        print("\n✓ Bootstrap completed successfully")
    except Exception as e:
        print(f"\n✗ Bootstrap failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        collector.report()
