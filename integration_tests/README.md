# Integration Tests

Cross-package integration tests for the quantitative research platform.

## Packages Tested

1. **quant_evaluator (QE)** - Pure evidence/metric/diagnosis engine
2. **factor_preprocess (FP)** - Model-input preparation layer
3. **factor_assets (FA)** - Factor asset identity, registry, lifecycle
4. **factor_optimizer (FO)** - Evidence-guided factor mutation and search

## Test Suites

### 1. QE + FP Pipeline (`test_qe_fp_pipeline.py`)
Tests the flow from raw factors through preprocessing to evaluation:
- Import isolation for QE and FP
- FactorBatch and LabelBundle creation
- Preprocessing policy definition
- Contract compatibility between QE and FP
- Error taxonomy and validation

**Key workflows:**
- Creating evaluation requests with factor batches
- Defining preprocessing transforms
- Shape compatibility verification

### 2. FA + FO Workflow (`test_fa_fo_workflow.py`)
Tests asset lifecycle, registry, and optimizer coordination:
- Import isolation for FA and FO
- FactorAsset and FactorIdentity creation
- AssetRepository CRUD operations
- Lifecycle state transitions
- SeenIndex for tracking evaluated factors
- Lineage tracking for derived factors

**Key workflows:**
- Factor registration and retrieval
- Lifecycle progression (CANDIDATE → UNDER_TEST → APPROVED → PRODUCTION)
- Evidence reference storage
- Factor set management

### 3. QE + FA Evidence Flow (`test_qe_fa_evidence_flow.py`)
Tests the flow from evaluation results to evidence storage:
- Evaluation results to EvidenceRef conversion
- Batch evaluation creating multiple evidence records
- EvaluationBundle to asset attachment
- Asset lifecycle updates based on evidence
- QE diagnosis to FA metadata mapping
- Repository queries by evidence metrics

**Key workflows:**
- Evaluation → Evidence → Lifecycle update
- Multi-factor batch evaluation evidence storage
- Quality score and warning propagation

### 4. All Packages Integration (`test_all_packages_integration.py`)
Tests complete workflows across all four packages:
- All packages import together
- Complete factor lifecycle end-to-end
- Multi-factor batch processing
- Factor optimization search simulation
- Cross-package error handling
- Realistic research scenarios

**Key workflows:**
- Full pipeline: Create asset → Preprocess → Evaluate → Store evidence → Optimize
- Search iterations with seen cache
- End-to-end realistic scenarios with 252 days × 50 assets

### 5. Import Isolation (`test_import_isolation.py`)
Verifies packages can be imported independently:
- Each package imports without requiring others
- No circular dependencies
- Public API completeness
- Contracts are independent
- Cross-package integration is optional
- No hidden dependencies

**Key tests:**
- Standalone import for each package
- API export verification
- Module independence checks

## Running Tests

### Run all integration tests
```bash
cd /home/shw/quant_projects/integration_tests
pytest
```

### Run specific test suite
```bash
pytest test_qe_fp_pipeline.py
pytest test_fa_fo_workflow.py
pytest test_qe_fa_evidence_flow.py
pytest test_all_packages_integration.py
pytest test_import_isolation.py
```

### Run with markers
```bash
pytest -m import_isolation
pytest -m qe_fp
pytest -m all_packages
```

### Run with verbose output
```bash
pytest -v
```

### Run with coverage
```bash
pytest --cov=quant_evaluator --cov=factor_preprocess --cov=factor_assets --cov=factor_optimizer
```

## Test Structure

Each test suite follows this pattern:
1. **Import isolation tests** - Verify package can be imported independently
2. **Basic API tests** - Test public API contracts
3. **Integration tests** - Test cross-package workflows
4. **Error handling tests** - Verify error propagation
5. **Realistic scenarios** - End-to-end workflows

## Expected Results

All tests should pass, demonstrating:
- ✓ Each package is independently importable
- ✓ No circular dependencies
- ✓ Public APIs are complete and documented
- ✓ Contracts are well-defined and validated
- ✓ Cross-package integration works correctly
- ✓ Error handling is consistent across boundaries
- ✓ Realistic workflows execute successfully

## Dependencies

Required packages:
- pytest >= 7.0
- numpy >= 1.20
- All four packages (QE, FP, FA, FO) must be installed

Install all packages in development mode:
```bash
cd /home/shw/quant_projects/quant_evaluator && pip install -e .
cd /home/shw/quant_projects/factor_preprocess && pip install -e .
cd /home/shw/quant_projects/factor_assets && pip install -e .
cd /home/shw/quant_projects/factor_optimizer && pip install -e .
```

## Test Coverage

- **QE + FP**: 13 tests
- **FA + FO**: 16 tests
- **QE + FA**: 10 tests
- **All packages**: 7 tests
- **Import isolation**: 17 tests

**Total**: 63+ integration tests

## Architecture Principles Verified

1. **Loose Coupling**: Each package can be imported and used independently
2. **Clear Contracts**: Well-defined interfaces between packages
3. **Fail-Closed**: Validation happens at boundaries
4. **Evidence-Driven**: Evaluation results flow to asset registry
5. **Lifecycle Management**: Factor progression based on evidence
6. **Reproducibility**: All inputs are explicit, nothing inferred

## Notes

- Tests use synthetic data (random values) for speed
- No external data dependencies
- All timing/calendar contracts are explicit (no inference)
- Tests verify contracts, not implementation details
- Focus on integration points and workflows, not exhaustive unit coverage
