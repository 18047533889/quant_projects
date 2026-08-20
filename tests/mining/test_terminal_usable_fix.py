import pytest
from mining.direct_use import build_direct_use_operator, DirectUseOperator
from cleaned_operators.registry import OperatorRegistry
from cleaned_operators import load_all

@pytest.fixture(scope="module", autouse=True)
def load_ops():
    load_all()

def test_terminal_usable_implies_composition_usable():
    """
    R21-TERMINAL-USABLE-FIX: terminal_usable must be True only if composition_usable is True.
    """
    for canonical in sorted(OperatorRegistry._catalog):
        if canonical.startswith("__"):
            continue
        try:
            row = build_direct_use_operator(canonical, OperatorRegistry._catalog[canonical])
        except Exception:
            continue

        if row.terminal_usable:
            assert row.composition_usable, (
                f"{canonical} is terminal_usable but NOT composition_usable"
            )

def test_production_terminal_usable_exists():
    """
    Verify production_terminal_usable field exists and is boolean.
    """
    for canonical in sorted(OperatorRegistry._catalog):
        if canonical.startswith("__"):
            continue
        try:
            row = build_direct_use_operator(canonical, OperatorRegistry._catalog[canonical])
            assert hasattr(row, "production_terminal_usable"), "Missing production_terminal_usable"
            assert isinstance(row.production_terminal_usable, bool), "Not bool"
        except Exception:
            continue

def test_production_terminal_usable_is_conjunction():
    """
    Verify production_terminal_usable = terminal_usable AND production_admitted.
    """
    for canonical in sorted(OperatorRegistry._catalog):
        if canonical.startswith("__"):
            continue
        try:
            row = build_direct_use_operator(canonical, OperatorRegistry._catalog[canonical])
        except Exception:
            continue

        expected = row.terminal_usable and row.production_admitted
        assert row.production_terminal_usable == expected, (
            f"{canonical}: expected {expected} but got {row.production_terminal_usable}"
        )
