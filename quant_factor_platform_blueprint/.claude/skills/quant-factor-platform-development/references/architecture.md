# Architecture Rules

DataAccess owns data/PIT access. FactorEngine owns factor semantics/computation. QuantEvaluator owns evidence. FactorOptimizer owns mutation/search. FactorAssets owns governance/organization. FactorPreprocess owns model-input representation. Research Control owns only the trial ledger.

Never create circular dependencies or a universal common package. Cross-package integration uses public API, versioned contracts and optional adapters.
