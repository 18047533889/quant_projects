# FactorOptimizer

**Status:** Initial bounded implementation

FactorOptimizer is an optional evidence-guided mutation and search package. It generates legal factor mutations, orchestrates FE compilation and QE evaluation, and maintains search state without owning operators, metrics, or execution kernels.

## Scope

- Typed mutation specifications with versioned grammar
- Parameter role/domain/type validation
- FE legality checking via adapter protocol
- QE evidence consumption via adapter protocol
- Search budget tracking (trials, evaluations, cost)
- Seen cache with FE canonical identity
- Conservative mutation validation with explicit legal/illegal classification

## Out of Scope (this version)

- Arbitrary Python code execution
- LLM-based proposal generation (stubs provided for future extension)
- Training or model execution
- Direct FE operator kernel implementation
- Direct QE metric implementation

## Installation

```bash
pip install -e .
```

Optional adapters:
```bash
pip install -e .[factor_engine,quant_evaluator]
```

## Package Boundaries

FactorOptimizer does NOT:
- Parse factor expressions (uses FE adapter)
- Compute factor values (uses FE adapter)
- Calculate evaluation metrics (uses QE adapter)
- Store raw factor values (uses FA adapter)
- Make final admission decisions (proposes only)

## Testing

```bash
pytest tests/
```
