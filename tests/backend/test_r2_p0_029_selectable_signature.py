"""R2-P0-029 selectable backend ABI parity gate."""
from __future__ import annotations

import pytest

from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry
from backend.operator_capability import UnsupportedOperatorBackendError, get_best_backend


@pytest.fixture(scope="module", autouse=True)
def _loaded() -> None:
    load_all()


def _params(operator: object) -> tuple[str, ...]:
    metadata = getattr(operator, "metadata", None)
    return tuple(getattr(metadata, "param_names", ()) or ())


def test_known_raw_mismatches_are_not_selectable_in_production() -> None:
    """Raw registrations cannot bypass the capability admission authority."""
    cases = {
        "benchmark_relative_price": (("price", "benchmark_price"), ("numerator", "denominator")),
        "float_share_ratio": (("float_shares", "total_shares"), ("numerator", "denominator")),
        "rolling_beta_to_market": (("ret", "benchmark_ret", "window"), ("y", "x", "window")),
    }
    for canonical, (reference_params, polars_params) in cases.items():
        reference = OperatorRegistry.get(canonical, "pandas_numpy", mode="research")
        polars = OperatorRegistry.get(canonical, "polars", mode="research")
        assert _params(reference) == reference_params
        assert _params(polars) == polars_params
        with pytest.raises(UnsupportedOperatorBackendError):
            get_best_backend(canonical, mode="production", prefer="polars")


def test_selectable_production_pairs_have_equal_ordered_signatures() -> None:
    """Compare metadata only after the production selector admits both sides."""
    mismatches: list[tuple[str, str, tuple[str, ...], tuple[str, ...]]] = []
    for canonical in OperatorRegistry.list_canonical():
        try:
            reference, _ = get_best_backend(canonical, mode="production", prefer="pandas_numpy")
        except UnsupportedOperatorBackendError:
            continue
        for backend in ("polars", "sql"):
            try:
                operator, selected_backend = get_best_backend(
                    canonical, mode="production", prefer=backend
                )
            except UnsupportedOperatorBackendError:
                continue
            # SQL registrations are marker operators; bound SQL parameters are
            # checked against canonical contracts rather than marker metadata.
            if selected_backend == "sql" and not _params(operator):
                continue
            if _params(operator) != _params(reference):
                mismatches.append(
                    (canonical, selected_backend, _params(reference), _params(operator))
                )
    assert not mismatches, mismatches
