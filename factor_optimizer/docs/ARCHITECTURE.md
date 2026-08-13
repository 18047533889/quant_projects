# FactorOptimizer Architecture

**Version:** 0.1.0  
**Last Updated:** 2026-08-14

## Overview

FactorOptimizer is an evidence-guided mutation and search orchestrator. It generates legal mutations, tracks search state, and coordinates with FE/QE through adapter protocols—without executing factors, computing metrics, or making final admission decisions.

## Design Principles

1. **No Direct Execution**: FO proposes mutations, FE validates/executes, QE evaluates
2. **Conservative Validation**: Fail-closed on legality checks
3. **Versioned Grammar**: Mutation specs include version for reproducibility
4. **Budget-Aware**: First-class resource tracking (trials, evals, cost, LLM)
5. **Protocol-Based**: FE/QE integration via adapters, not hard dependencies

## Module Organization

```
factor_optimizer/
├── contracts/              # Core data contracts
│   ├── candidate_mutation.py
│   ├── search_budget.py
│   └── trial.py
├── grammar/               # Mutation specifications
│   ├── mutation_spec.py
│   ├── registry.py
│   └── validation.py
├── seen/                  # Deduplication
│   └── identity.py
├── complexity/            # Cost estimation
│   └── profile.py
├── policy/               # Admission & repair
│   ├── decisions.py
│   └── repair.py
├── search/               # Search orchestration
│   ├── runner.py
│   ├── multifidelity.py
│   ├── pareto.py
│   ├── plateau.py
│   └── lineage.py
├── llm/                  # LLM proposals (stubs)
│   ├── prompts.py
│   ├── proposal.py
│   └── records.py
└── adapters/             # External integrations
    ├── factor_engine.py
    └── quant_evaluator.py
```

## Key Contracts

### CandidateMutation
Immutable mutation proposal with complete provenance, parameters, hypothesis, and expected signatures.

### Trial
Tracks mutation through lifecycle: PROPOSED → VALIDATING → LEGAL/ILLEGAL → EVALUATING → EVALUATED/FAILED.

### SearchBudget & BudgetTracker
Resource limits and usage tracking prevent runaway searches.

## Integration Boundaries

**FactorEngine (FE):**
- FO: "Is this mutation legal?"
- FE: "Yes/No + reason"
- FO never parses expressions

**QuantEvaluator (QE):**
- FO: "Evaluate this factor"
- QE: "Here's the evidence bundle"
- FO never computes metrics

**FactorAssets (FA):**
- FO: "Here's a promising candidate"
- FA: "Admitted/Rejected via gates"
- FO never makes final decisions

## Search Strategies

### Multi-Fidelity
Start cheap (L0=5% sample), promote winners to full evaluation (L3=100%).

### Pareto Frontier
Track non-dominated points in multi-objective space (IC vs turnover).

### Lineage Tree
Track which mutations are productive, prune unproductive branches.

### Plateau Detection
Stop early when no improvement over window.

---

**Last Updated:** 2026-08-14
