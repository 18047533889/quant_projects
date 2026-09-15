from factor_engine.api.dsl_parser import parse_recommended_expr
from factor_engine.api.mining_integration import (
    default_mining_operator_allowlist,
    recommended_research_operator_allowlist,
    validate_recommended_factor_engine_dsl,
)
from factor_engine.api.operator_registry import (
    build_dsl_allowlist,
    build_recommended_authoring_allowlist,
)
from factor_engine.runtime.operator_snapshot import (
    get_cached_runtime_operator_snapshot,
    query_runtime_operator_snapshot,
)


def test_recommended_dsl_defaults_to_full_public_runtime_surface():
    recommended = build_recommended_authoring_allowlist()
    explicit = build_dsl_allowlist(surface="all")
    assert set(recommended) == set(explicit)
    name = next(iter(sorted(set(explicit) - set(build_dsl_allowlist(surface="daily")))))
    parse_recommended_expr(f"{name}(close)")
    assert validate_recommended_factor_engine_dsl(f"{name}(close)") == (True, "OK")


def test_snapshot_counts_and_direct_canonical_query_are_same_authority():
    snapshot = get_cached_runtime_operator_snapshot()
    counts = snapshot["counts"]
    assert counts["canonical"] == len(snapshot["operators"])
    assert counts["aliases"] == len(snapshot["aliases"])
    assert counts["public_recipes"] == len(snapshot["recipes"])
    row = snapshot["operators"][len(snapshot["operators"]) // 2]
    page = query_runtime_operator_snapshot(
        snapshot, canonical=row["canonical"], offset=0, limit=1
    )
    assert page["catalog_digest"] == snapshot["catalog_digest"]
    assert page["total_matches"] == 1
    assert page["items"][0] == row
    assert {"blockers", "warnings", "execution_state", "assurance"} <= row.keys()


def test_snapshot_not_run_is_never_reported_as_verified():
    snapshot = get_cached_runtime_operator_snapshot()
    for row in snapshot["operators"]:
        statuses = {b["evidence"]["status"] for b in row["backends"]}
        if "NOT_RUN" in statuses:
            assert row["assurance"] != "VERIFIED"
            assert row["execution_state"] != "EXECUTABLE_VERIFIED"


def test_recommended_research_catalog_is_not_a_fastpath_subset():
    recommended = set(recommended_research_operator_allowlist())
    legacy = set(default_mining_operator_allowlist())
    assert legacy <= recommended
    assert recommended
