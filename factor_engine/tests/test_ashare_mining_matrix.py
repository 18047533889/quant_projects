"""Tests for A-share mining matrix generation."""
import pytest
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent  # factor_engine/
repo_root = project_root.parent  # quant_projects/ (evidence/ lives at repo root)
sys.path.insert(0, str(project_root))


def test_mining_matrix_import():
    """Test that the mining matrix generation script can be imported."""
    from scripts.generate_ashare_mining_matrix import (
        build_mining_matrix,
        classify_operator,
        verify_stock_specificity,
        verify_pit_safe,
        verify_prefix_causal,
        verify_non_degenerate,
        identify_ashare_operators,
        CanonicalClass,
    )
    assert True


def test_classify_operator():
    """Test operator classification."""
    from scripts.generate_ashare_mining_matrix import classify_operator, CanonicalClass

    # A-share terminal alpha
    assert classify_operator("pe_ratio") == CanonicalClass.ASHARE_DAILY_TERMINAL_ALPHA
    assert classify_operator("abnormal_turnover") == CanonicalClass.ASHARE_DAILY_TERMINAL_ALPHA

    # Condition
    assert classify_operator("is_trading") == CanonicalClass.CONDITION
    assert classify_operator("is_st") == CanonicalClass.CONDITION

    # Event
    assert classify_operator("price_gap_up") == CanonicalClass.EVENT
    assert classify_operator("volume_surge") == CanonicalClass.EVENT

    # State
    assert classify_operator("trend_state") == CanonicalClass.STATE
    assert classify_operator("volatility_regime") == CanonicalClass.STATE

    # Group context
    assert classify_operator("industry_rank") == CanonicalClass.GROUP_CONTEXT
    assert classify_operator("sector_rank") == CanonicalClass.GROUP_CONTEXT

    # Global context
    assert classify_operator("market_index") == CanonicalClass.GLOBAL_CONTEXT
    assert classify_operator("market_breadth") == CanonicalClass.GLOBAL_CONTEXT


def test_verify_stock_specificity():
    """Test stock-specificity verification."""
    from scripts.generate_ashare_mining_matrix import verify_stock_specificity

    # Stock-specific operators
    assert verify_stock_specificity("pe_ratio") is True
    assert verify_stock_specificity("abnormal_turnover") is True
    assert verify_stock_specificity("turnover_ratio") is True

    # Non-stock-specific operators (global context)
    assert verify_stock_specificity("market_index") is False


def test_verify_pit_safe():
    """Test PIT safety verification."""
    from scripts.generate_ashare_mining_matrix import verify_pit_safe

    # PIT-safe operators
    assert verify_pit_safe("pe_ratio") is True
    assert verify_pit_safe("abnormal_turnover") is True
    assert verify_pit_safe("turnover_ratio") is True
    assert verify_pit_safe("rsi") is True

    # Non-PIT-safe operators (events)
    assert verify_pit_safe("price_gap_up") is False


def test_verify_prefix_causal():
    """Test prefix-causal verification."""
    from scripts.generate_ashare_mining_matrix import verify_prefix_causal

    # Prefix-causal operators
    assert verify_prefix_causal("pe_ratio") is True
    assert verify_prefix_causal("abnormal_turnover") is True
    assert verify_prefix_causal("turnover_ratio") is True
    assert verify_prefix_causal("rsi") is True

    # Non-prefix-causal operators (events)
    assert verify_prefix_causal("price_gap_up") is False


def test_verify_non_degenerate():
    """Test non-degenerate verification."""
    from scripts.generate_ashare_mining_matrix import verify_non_degenerate

    # Non-degenerate operators
    assert verify_non_degenerate("pe_ratio") is True
    assert verify_non_degenerate("abnormal_turnover") is True
    assert verify_non_degenerate("turnover_ratio") is True
    assert verify_non_degenerate("rsi") is True
    assert verify_non_degenerate("industry_rank") is True

    # Degenerate operators (events)
    assert verify_non_degenerate("price_gap_up") is False


def test_identify_ashare_operators():
    """Test A-share operator identification."""
    from scripts.generate_ashare_mining_matrix import identify_ashare_operators

    operators = identify_ashare_operators()
    assert len(operators) > 0
    assert "pe_ratio" in operators
    assert "abnormal_turnover" in operators
    assert "limit_up_count" in operators
    assert "is_trading" in operators
    assert "industry_rank" in operators


def test_build_mining_matrix():
    """Test mining matrix building."""
    from scripts.generate_ashare_mining_matrix import build_mining_matrix

    matrix = build_mining_matrix()

    assert len(matrix.operators) == 88
    assert matrix.metadata["market"] == "ashare"
    assert matrix.metadata["frequency"] == "daily"
    assert matrix.metadata["signal_structure"] == "cross_sectional"

    # Check verification counts
    assert matrix.metadata["stock_specific_count"] == 55
    assert matrix.metadata["pit_safe_count"] == 42
    assert matrix.metadata["prefix_causal_count"] == 42
    assert matrix.metadata["non_degenerate_count"] == 46

    # Check canonical class distribution
    assert matrix.metadata["canonical_class_distribution"]["ASHARE_DAILY_TERMINAL_ALPHA"] == 49
    assert matrix.metadata["canonical_class_distribution"]["CONDITION"] == 11
    assert matrix.metadata["canonical_class_distribution"]["EVENT"] == 10
    assert matrix.metadata["canonical_class_distribution"]["STATE"] == 6
    assert matrix.metadata["canonical_class_distribution"]["GROUP_CONTEXT"] == 6
    assert matrix.metadata["canonical_class_distribution"]["GLOBAL_CONTEXT"] == 6


def test_evidence_generation():
    """Test evidence generation script."""
    from scripts.generate_ashare_mining_matrix_evidence import (
        build_evidence,
        classify_operator,
        verify_stock_specificity,
        verify_pit_safe,
        verify_prefix_causal,
        verify_non_degenerate,
    )

    evidence = build_evidence()

    assert len(evidence.operators) == 88
    assert evidence.metadata["market"] == "ashare"
    assert evidence.metadata["frequency"] == "daily"

    # Check verification results
    assert evidence.verification_results["stock_specific"]["count"] == 55
    assert evidence.verification_results["pit_safe"]["count"] == 42
    assert evidence.verification_results["prefix_causal"]["count"] == 42
    assert evidence.verification_results["non_degenerate"]["count"] == 46


def test_yaml_export():
    """Test YAML export."""
    import yaml
    yaml_path = repo_root / "evidence" / "r2" / "R21-ASHARE-MINING-MATRIX.yaml"

    assert yaml_path.exists()

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "metadata" in data
    assert "operators" in data
    assert "verification_results" in data

    assert len(data["operators"]) == 88
    assert data["metadata"]["market"] == "ashare"


def test_documentation_exists():
    """Test that documentation exists."""
    doc_path = project_root / "docs" / "ASHARE_MINING_MATRIX.md"
    assert doc_path.exists()

    with open(doc_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "ASHARE_DAILY_CROSS_SECTIONAL_MINING_MATRIX" in content
    assert "Market | ashare" in content
    assert "Frequency | daily" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
