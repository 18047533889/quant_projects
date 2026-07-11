# External factor-pack evaluation

`AutoFactorEvaluation-RECONSTRUCT` is a provider-neutral evaluation library. It does not own GTJA, fundamental, analyst, news/text, alternative-data, flow, or graph factors.

```text
quant_projects/
├── AutoFactorEvaluation-RECONSTRUCT/   # generic evaluation library
├── gtja191/                            # external factor pack
├── fundamental_factors/                # future external pack
├── news_text_factors/                  # future external pack
└── analyst_revision_factors/           # future external pack
```

Each factor project exposes a callable returning `evaluation.factor_pack.FactorPack`. The evaluator receives immutable factor definitions and owns:

- DataAccess snapshot binding;
- FactorEngine execution;
- PIT universe and tradability filtering;
- cross-sectional purification;
- train/validation/test separation;
- validation-only selection and FDR;
- cost-aware portfolio diagnostics;
- deterministic reports, resume, and staging materialization.

Generic invocation:

```bash
python -m evaluation.batch \
  --provider some_factor_project.provider:load_pack \
  --output-dir output/factor_evaluation \
  --synthetic
```

The core package must remain importable and testable without any specific factor project on `PYTHONPATH`.
