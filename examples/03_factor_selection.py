"""
Example 03: Factor Selection Workflow (Standalone)

Demonstrates the FA (Factor Assets) workflow:
1. Register FactorAsset instances with identity and lineage
2. Attach EvidenceRef from QE evaluations
3. Apply selection gates (IC threshold, monotonicity, lifecycle state)
4. Build a FactorSet for downstream consumption

This standalone version includes inline implementations for demonstration.
"""

from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import json
import hashlib


class LifecycleState(Enum):
    """Factor lifecycle states."""
    DRAFT = "DRAFT"
    EVALUATED = "EVALUATED"
    VALIDATED = "VALIDATED"
    PRODUCTION = "PRODUCTION"
    DEPRECATED = "DEPRECATED"


@dataclass
class EvidenceRef:
    """Reference to evaluation evidence."""
    evidence_id: str
    factor_id: str
    created_at: datetime
    ic_mean: float
    rank_ic_ir: float
    monotonicity: float
    n_dates: int


@dataclass
class FactorAsset:
    """A registered factor with identity and evidence."""
    factor_id: str
    expression: str
    expression_hash: str
    description: str
    tags: List[str]
    lifecycle_state: LifecycleState
    evidence_refs: List[EvidenceRef]
    created_at: datetime


@dataclass
class FactorSet:
    """A collection of selected factors."""
    set_id: str
    factor_ids: List[str]
    selection_criteria: Dict
    created_at: datetime
    metadata: Dict


def create_factor_id(expression: str) -> tuple:
    """Generate canonical factor ID from expression."""
    expr_hash = hashlib.sha256(expression.encode()).hexdigest()[:16]
    factor_id = f"factor_{expr_hash}"
    return factor_id, expr_hash


def simulate_qe_evidence(factor_id: str, expression: str, ic_mean: float, rank_ic_ir: float) -> EvidenceRef:
    """Simulate an EvidenceRef from QE evaluation."""
    return EvidenceRef(
        evidence_id=f"ev_{factor_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        factor_id=factor_id,
        created_at=datetime.now(),
        ic_mean=ic_mean,
        rank_ic_ir=rank_ic_ir,
        monotonicity=0.85 if abs(ic_mean) > 0.03 else 0.60,
        n_dates=250,
    )


def main():
    """Run factor selection workflow."""

    print("=" * 70)
    print("EXAMPLE 03: Factor Selection Workflow")
    print("=" * 70)
    print()

    # Step 1: Define factors to register
    print("STEP 1: Define Factor Candidates")
    print("-" * 70)
    print()

    factors_to_register = [
        {
            "expression": "ts_rank(close / ts_delay(close, 20), 120)",
            "description": "120-day momentum with 20-day return normalization",
            "tags": ["momentum", "trend"],
            "ic_mean": 0.045,
            "rank_ic_ir": 1.2,
        },
        {
            "expression": "cs_zscore(log(pb_ratio))",
            "description": "Price-to-book value factor, cross-sectionally standardized",
            "tags": ["value", "fundamental"],
            "ic_mean": 0.038,
            "rank_ic_ir": 1.0,
        },
        {
            "expression": "ts_std(returns, 20) * sqrt(252)",
            "description": "Annualized 20-day return volatility",
            "tags": ["volatility", "risk"],
            "ic_mean": -0.025,
            "rank_ic_ir": 0.8,
        },
        {
            "expression": "rolling_corr(volume, abs(returns), 60)",
            "description": "60-day correlation between volume and absolute returns",
            "tags": ["liquidity", "microstructure"],
            "ic_mean": 0.018,
            "rank_ic_ir": 0.6,
        },
        {
            "expression": "cs_rank(roe) * cs_rank(roa)",
            "description": "Combined ROE and ROA ranking",
            "tags": ["quality", "fundamental"],
            "ic_mean": 0.052,
            "rank_ic_ir": 1.4,
        },
    ]

    print(f"Prepared {len(factors_to_register)} factor candidates")
    print()

    # Step 2: Register factors as assets
    print("STEP 2: Register Factor Assets")
    print("-" * 70)
    print()

    registered_assets: List[FactorAsset] = []

    for idx, factor_spec in enumerate(factors_to_register):
        print(f"Registering factor {idx + 1}/{len(factors_to_register)}")

        # Generate canonical identity
        factor_id, expr_hash = create_factor_id(factor_spec["expression"])

        # Simulate QE evaluation evidence
        evidence = simulate_qe_evidence(
            factor_id,
            factor_spec["expression"],
            ic_mean=factor_spec["ic_mean"],
            rank_ic_ir=factor_spec["rank_ic_ir"],
        )

        # Create asset
        asset = FactorAsset(
            factor_id=factor_id,
            expression=factor_spec["expression"],
            expression_hash=expr_hash,
            description=factor_spec["description"],
            tags=factor_spec["tags"],
            lifecycle_state=LifecycleState.EVALUATED,
            evidence_refs=[evidence],
            created_at=datetime.now(),
        )

        registered_assets.append(asset)

        print(f"  ✓ {asset.factor_id}")
        print(f"    Expression: {asset.expression[:60]}...")
        print(f"    IC Mean: {evidence.ic_mean:.4f}")
        print(f"    Rank IC IR: {evidence.rank_ic_ir:.2f}")
        print()

    print(f"Total registered: {len(registered_assets)} assets")
    print()

    # Step 3: Apply selection gates
    print("STEP 3: Apply Selection Gates")
    print("-" * 70)
    print()

    # Define selection criteria
    IC_THRESHOLD = 0.03
    RANK_IC_IR_THRESHOLD = 0.9
    MONOTONICITY_THRESHOLD = 0.75

    print("Selection criteria:")
    print(f"  - IC Mean >= {IC_THRESHOLD}")
    print(f"  - Rank IC IR >= {RANK_IC_IR_THRESHOLD}")
    print(f"  - Monotonicity >= {MONOTONICITY_THRESHOLD}")
    print(f"  - Lifecycle state: EVALUATED or VALIDATED")
    print()

    selected_assets: List[FactorAsset] = []

    for asset in registered_assets:
        evidence = asset.evidence_refs[0]

        # Apply gates
        ic_mean = evidence.ic_mean
        rank_ic_ir = evidence.rank_ic_ir
        monotonicity = evidence.monotonicity

        passes_ic = abs(ic_mean) >= IC_THRESHOLD
        passes_ir = rank_ic_ir >= RANK_IC_IR_THRESHOLD
        passes_mono = monotonicity >= MONOTONICITY_THRESHOLD
        passes_state = asset.lifecycle_state in [
            LifecycleState.EVALUATED,
            LifecycleState.VALIDATED,
        ]

        if passes_ic and passes_ir and passes_mono and passes_state:
            selected_assets.append(asset)
            print(f"  ✓ {asset.factor_id}")
            print(f"    IC: {ic_mean:.4f}, IR: {rank_ic_ir:.2f}, Mono: {monotonicity:.2f}")
        else:
            print(f"  ✗ {asset.factor_id}")
            reasons = []
            if not passes_ic:
                reasons.append(f"IC {ic_mean:.4f} < {IC_THRESHOLD}")
            if not passes_ir:
                reasons.append(f"IR {rank_ic_ir:.2f} < {RANK_IC_IR_THRESHOLD}")
            if not passes_mono:
                reasons.append(f"Mono {monotonicity:.2f} < {MONOTONICITY_THRESHOLD}")
            print(f"    Reasons: {', '.join(reasons)}")

    print()
    print(f"Selected: {len(selected_assets)}/{len(registered_assets)} factors")
    print()

    # Step 4: Build FactorSet
    print("STEP 4: Build FactorSet")
    print("-" * 70)
    print()

    if not selected_assets:
        print("No factors passed selection gates.")
        return

    factor_set = FactorSet(
        set_id="research_alpha_v1",
        factor_ids=[asset.factor_id for asset in selected_assets],
        selection_criteria={
            "ic_threshold": IC_THRESHOLD,
            "rank_ic_ir_threshold": RANK_IC_IR_THRESHOLD,
            "monotonicity_threshold": MONOTONICITY_THRESHOLD,
        },
        created_at=datetime.now(),
        metadata={
            "purpose": "Research alpha pool for model training",
            "created_by": "research_team",
            "n_factors": len(selected_assets),
        },
    )

    print(f"FactorSet ID: {factor_set.set_id}")
    print(f"Factor count: {len(factor_set.factor_ids)}")
    print()

    print("Factors in set:")
    for asset in selected_assets:
        print(f"  - {asset.factor_id}")
        print(f"    {asset.description}")
        print(f"    Tags: {', '.join(asset.tags)}")
    print()

    # Step 5: Save FactorSet
    print("STEP 5: Save FactorSet")
    print("-" * 70)
    print()

    output_dir = Path("/tmp/fa_example_03")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{factor_set.set_id}.json"

    factor_set_dict = {
        "set_id": factor_set.set_id,
        "factor_ids": factor_set.factor_ids,
        "selection_criteria": factor_set.selection_criteria,
        "created_at": factor_set.created_at.isoformat(),
        "metadata": factor_set.metadata,
        "factors": [
            {
                "factor_id": asset.factor_id,
                "expression": asset.expression,
                "description": asset.description,
                "tags": asset.tags,
                "evidence": {
                    "ic_mean": asset.evidence_refs[0].ic_mean,
                    "rank_ic_ir": asset.evidence_refs[0].rank_ic_ir,
                    "monotonicity": asset.evidence_refs[0].monotonicity,
                },
            }
            for asset in selected_assets
        ],
    }

    output_path.write_text(json.dumps(factor_set_dict, indent=2, ensure_ascii=False))
    print(f"FactorSet saved to: {output_path}")
    print()

    # Step 6: Understanding the workflow
    print("STEP 6: Understanding the Selection Workflow")
    print("-" * 70)
    print()
    print("The factor selection workflow ensures:")
    print()
    print("1. Identity Management")
    print("   - Canonical factor_id from expression hash")
    print("   - Prevents duplicate registration")
    print("   - Enables exact deduplication")
    print()
    print("2. Evidence Attachment")
    print("   - Links evaluation results to assets")
    print("   - Tracks metrics (IC, IR, monotonicity)")
    print("   - Preserves provenance chain")
    print()
    print("3. Selection Gates")
    print("   - IC threshold: minimum predictive power")
    print("   - IR threshold: minimum risk-adjusted performance")
    print("   - Monotonicity: ensure proper factor direction")
    print("   - Lifecycle gate: only production-ready factors")
    print()
    print("4. FactorSet Construction")
    print("   - Bundle selected factors")
    print("   - Document selection criteria")
    print("   - Enable versioning and reproducibility")
    print()
    print("The FactorSet flows to:")
    print("  - FP (Factor Preprocess) for transformation")
    print("  - Model training pipelines")
    print("  - Production deployment systems")
    print()

    print("=" * 70)
    print("Example 03 Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
