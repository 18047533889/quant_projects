"""The standalone repository root is already the factor_engine package."""
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("relative", [
    "docs/operator_manifest.json",
    "docs/backend_evidence_manifest.json",
    "cleaned_operators/docs/operators_catalog.json",
])
def test_manifests_have_one_canonical_location(relative: str) -> None:
    assert (ROOT / relative).is_file()
    assert not (ROOT / "factor_engine" / relative).exists(), (
        f"Duplicate generated manifest: factor_engine/{relative}; "
        "resolve output paths from the package root, not the working directory"
    )


def test_package_root_is_not_duplicated() -> None:
    assert (ROOT / "__init__.py").is_file()
    assert (ROOT / "runtime" / "engine.py").is_file()
    assert not (ROOT / "factor_engine").exists()
