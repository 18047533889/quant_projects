# GTJA185 evaluation adapter

GTJA185 is owned by the sibling `gtja191` project. `autofactor.provider:load_pack` converts the canonical GTJA catalog into the provider-neutral `FactorPack` contract without copying formulas into AutoFactorEvaluation.

```bash
export PYTHONPATH="$PWD:$PWD/factor_engine:$PWD/AutoFactorEvaluation-RECONSTRUCT:$PWD/gtja191"
python gtja191/scripts/evaluate_with_autofactor.py --synthetic
```

For production, add the registered market dataset, PIT universe dataset, date range, and `--run-mode production`.

The provider is responsible for factor identity, formulas, source formulas, version, source hash, pack hash, and domain metadata. AutoFactorEvaluation remains responsible for the evaluation methodology and reporting.
