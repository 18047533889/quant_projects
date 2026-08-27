"""Smoke test for the independently-installable quant_platform package.

Verifies the pure-stdlib contracts layer imports and exposes its public API.
"""

import platform as _stdlib_platform

import quant_platform.app.contracts as contracts


def test_contracts_public_api_nonempty():
    assert len(contracts.__all__) > 0


def test_contracts_are_pure_stdlib():
    # The contracts layer must not pull in any third-party runtime deps.
    import sys

    third_party = {
        "numpy",
        "pandas",
        "pydantic",
        "fastapi",
        "sqlalchemy",
        "duckdb",
        "pyarrow",
    }
    loaded = set(sys.modules)
    assert not (third_party & loaded), f"third-party modules loaded: {third_party & loaded}"


def test_stdlib_platform_not_shadowed():
    # `import platform` must resolve to the stdlib module, not quant_platform.
    assert _stdlib_platform.__file__ is None or "quant_platform" not in (
        _stdlib_platform.__file__ or ""
    )
    assert hasattr(_stdlib_platform, "python_version")


def test_contracts_importable():
    from quant_platform.app.contracts import (
        ArtifactRef,
        EventEnvelope,
        JobSpec,
        Role,
        TimingContract,
    )

    assert ArtifactRef is not None
    assert EventEnvelope is not None
    assert JobSpec is not None
    assert Role is not None
    assert TimingContract is not None
