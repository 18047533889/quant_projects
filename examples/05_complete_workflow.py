"""
Example 05: Complete End-to-End Workflow (Standalone)

Demonstrates full integration of all four packages:
- FO (Factor Optimizer): Generate candidate factors
- QE (Quantitative Evaluator): Evaluate each candidate
- FA (Factor Assets): Register, track, and select factors
- FP (Factor Preprocess): Transform selected factors to model input

This is the complete research-to-production pipeline.
"""

from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import json
import numpy as np


class ResearchControl:
    """
    Research control system for full provenance tracking.
    Records all operations across FO, QE, FA, and FP.
    """

    def __init__(self, experiment_id: str, storage_root: Path):
        self.experiment_id = experiment_id
        self.storage_root = storage_root
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.events: List[Dict] = []
        self.started_at = datetime.now()

    def record(self, stage: str, **kwargs):
        """Record an event."""
        self.events.append({
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            **kwargs
        })

    def save(self) -> Path:
        """Save provenance log."""
        log_path = self.storage_root / f"{self.experiment_id}_provenance.json"
        summary = {
            "experiment_id": self.experiment_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": datetime.now().isoformat(),
            "total_events": len(self.events),
            "events": self.events,
        }
        log_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        return log_path


def simulate_qe_evaluation(expression: str) -> Tuple[float, float]:
    """Simulate QE evaluation."""
    np.random.seed(hash(expression) % 2**32)
    base_ic = (hash(expression) % 100) / 1000.0 - 0.02
    noise = np.random.randn() * 0.01
    ic_mean = base_ic + noise
    rank_ic_ir = ic_mean / 0.15
    return ic_mean, rank_ic_ir


def main():
    """Run complete end-to-end workflow."""

    print("=" * 70)
    print("EXAMPLE 05: Complete End-to-End Workflow")
    print("=" * 70)
    print()
    print("This demonstrates the full research-to-production pipeline:")
    print("  FO → QE → FA → FP → Model Input")
    print()

    # Initialize
    experiment_id = f"e2e_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    storage_root = Path(f"/tmp/e2e_example_05/{experiment_id}")
    rc = ResearchControl(experiment_id, storage_root)

    print(f"Experiment ID: {experiment_id}")
    print()

    # ====================
    # STAGE 1: FO - Generate Candidates
    # ====================
    print("=" * 70)
    print("STAGE 1: Factor Generation (FO)")
    print("=" * 70)
    print()

    seed_factors = [
        "ts_rank(close / ts_delay(close, 20), 120)",
        "cs_zscore(log(market_cap))",
        "ts_std(returns, 20)",
    ]

    candidates = seed_factors.copy()

    # Generate mutations
    print("Generating mutations from seeds...")
    mutations = [
        "ts_rank(close / ts_delay(close, 10), 120)",
        "ts_rank(close / ts_delay(close, 30), 120)",
        "ts_rank(close / ts_delay(close, 20), 60)",
        "ts_zscore(close / ts_delay(close, 20), 120)",
    ]
    candidates.extend(mutations)

    for parent, mutant in [(seed_factors[0], m) for m in mutations]:
        rc.record("FO_MUTATION", parent=parent, mutant=mutant)

    print(f"Generated {len(candidates)} total candidates")
    print()

    # ====================
    # STAGE 2: QE - Evaluate
    # ====================
    print("=" * 70)
    print("STAGE 2: Evaluation (QE)")
    print("=" * 70)
    print()

    evaluated = []

    for expr in candidates:
        ic, ir = simulate_qe_evaluation(expr)
        rc.record("QE_EVALUATION", expression=expr, ic_mean=ic, rank_ic_ir=ir)

        if ic > 0.01:  # Filter poor candidates
            evaluated.append({"expression": expr, "ic_mean": ic, "rank_ic_ir": ir})
            print(f"  ✓ {expr[:55]}... IC={ic:.4f}, IR={ir:.2f}")
        else:
            print(f"  ✗ {expr[:55]}... IC={ic:.4f} (rejected)")

    print()
    print(f"Evaluated {len(evaluated)}/{len(candidates)} candidates passed filter")
    print()

    # ====================
    # STAGE 3: FA - Register and Select
    # ====================
    print("=" * 70)
    print("STAGE 3: Registration and Selection (FA)")
    print("=" * 70)
    print()

    # Register assets
    print("Registering as assets...")
    registered = []
    for candidate in evaluated:
        factor_id = f"factor_{hash(candidate['expression']) % 100000:05d}"
        registered.append({**candidate, "factor_id": factor_id})
        rc.record("FA_REGISTRATION", factor_id=factor_id, expression=candidate["expression"])
        print(f"  Registered: {factor_id}")

    print()

    # Apply selection gates
    IC_THRESHOLD = 0.020
    IR_THRESHOLD = 0.12

    print(f"Applying selection gates (IC >= {IC_THRESHOLD}, IR >= {IR_THRESHOLD})...")
    selected = []

    for asset in registered:
        passes = abs(asset["ic_mean"]) >= IC_THRESHOLD and asset["rank_ic_ir"] >= IR_THRESHOLD

        if passes:
            selected.append(asset)
            rc.record("FA_SELECTION", factor_id=asset["factor_id"], selected=True)
            print(f"  ✓ {asset['factor_id']}: IC={asset['ic_mean']:.4f}, IR={asset['rank_ic_ir']:.2f}")
        else:
            rc.record("FA_SELECTION", factor_id=asset["factor_id"], selected=False)
            print(f"  ✗ {asset['factor_id']}: IC={asset['ic_mean']:.4f}, IR={asset['rank_ic_ir']:.2f}")

    print()
    print(f"Selected {len(selected)}/{len(registered)} factors")
    print()

    if not selected:
        print("No factors passed selection. Exiting.")
        rc.save()
        return

    # ====================
    # STAGE 4: FP - Preprocess
    # ====================
    print("=" * 70)
    print("STAGE 4: Preprocessing (FP)")
    print("=" * 70)
    print()

    # Simulate preprocessing
    n_dates = 250
    n_symbols = 500
    n_features = len(selected)

    print(f"Preprocessing {n_features} selected factors...")
    print()

    transforms = [
        "winsorize_0.01_0.99",
        "cs_zscore",
        "cs_rank",
        "rolling_zscore_60d",
        "ols_neutralize",
    ]

    for asset in selected:
        rc.record("FP_PREPROCESSING", factor_id=asset["factor_id"], transforms=transforms)
        print(f"  {asset['factor_id']}: Applied {len(transforms)} transforms")

    print()

    # Generate feature matrix
    print("Generating feature matrix...")
    np.random.seed(42)
    feature_matrix = np.random.randn(n_dates, n_symbols, n_features) * 0.3

    bundle_id = f"bundle_{experiment_id}"
    rc.record("FP_BUNDLE_CREATION", bundle_id=bundle_id, shape=feature_matrix.shape)

    print(f"Bundle ID: {bundle_id}")
    print(f"Shape: {feature_matrix.shape} (dates × symbols × features)")
    print()

    # Statistics
    missing_rate = 0.0  # No missing in synthetic data
    print(f"Missing rate: {missing_rate:.2%}")
    print(f"Mean: {feature_matrix.mean():.4f}")
    print(f"Std: {feature_matrix.std():.4f}")
    print()

    # ====================
    # STAGE 5: Save Results
    # ====================
    print("=" * 70)
    print("STAGE 5: Save Results and Provenance")
    print("=" * 70)
    print()

    # Save feature bundle
    bundle_path = storage_root / f"{bundle_id}.npz"
    np.savez_compressed(
        bundle_path,
        feature_matrix=feature_matrix,
        factor_ids=[a["factor_id"] for a in selected],
    )
    print(f"Feature bundle saved: {bundle_path}")

    # Save provenance
    provenance_path = rc.save()
    print(f"Provenance log saved: {provenance_path}")
    print()

    # ====================
    # SUMMARY
    # ====================
    print("=" * 70)
    print("WORKFLOW SUMMARY")
    print("=" * 70)
    print()

    print("Pipeline statistics:")
    print(f"  Candidates generated (FO):   {len(candidates)}")
    print(f"  Candidates evaluated (QE):   {len(evaluated)}")
    print(f"  Assets registered (FA):      {len(registered)}")
    print(f"  Factors selected (FA):       {len(selected)}")
    print(f"  Features in bundle (FP):     {n_features}")
    print()

    print("Event counts:")
    stage_counts = {}
    for event in rc.events:
        stage = event["stage"]
        stage_counts[stage] = stage_counts.get(stage, 0) + 1

    for stage, count in sorted(stage_counts.items()):
        print(f"  {stage}: {count}")
    print()

    # ====================
    # NEXT STEPS
    # ====================
    print("=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print()
    print("The FeatureBundle is now ready for:")
    print()
    print("1. Model Training")
    print("   - Load feature_matrix from bundle")
    print("   - Train predictive models (linear, tree, neural)")
    print("   - Validate out-of-sample")
    print()
    print("2. Backtesting")
    print("   - Simulate portfolio construction")
    print("   - Calculate PnL, Sharpe, drawdown")
    print("   - Compare to benchmark")
    print()
    print("3. Production Deployment")
    print("   - Deploy preprocessing pipeline")
    print("   - Set up real-time calculation")
    print("   - Monitor factor quality and drift")
    print()
    print("4. Continuous Improvement")
    print("   - Re-run FO with new data")
    print("   - Evaluate new candidates")
    print("   - Update FactorSet")
    print("   - Retrain and redeploy")
    print()

    print("=" * 70)
    print("PACKAGE COLLABORATION SUMMARY")
    print("=" * 70)
    print()
    print("This example demonstrated how the four packages work together:")
    print()
    print("FO (Factor Optimizer)")
    print("  - Generates candidate factors via mutation")
    print("  - Tracks search budget and complexity")
    print("  - No execution, delegates to QE")
    print()
    print("QE (Quantitative Evaluator)")
    print("  - Evaluates factors at multiple fidelities")
    print("  - Produces evidence bundles (IC, IR, monotonicity)")
    print("  - No storage, delegates to FA")
    print()
    print("FA (Factor Assets)")
    print("  - Registers factors with canonical identity")
    print("  - Attaches evidence from QE")
    print("  - Applies selection gates")
    print("  - Builds versioned FactorSets")
    print()
    print("FP (Factor Preprocess)")
    print("  - Transforms selected factors")
    print("  - Applies cross-sectional and time-series operations")
    print("  - Neutralizes exposures")
    print("  - Packages as FeatureBundle")
    print()
    print("Research Control")
    print("  - Tracks full provenance")
    print("  - Records all operations across packages")
    print("  - Enables reproducibility")
    print()

    print("=" * 70)
    print("Example 05 Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
