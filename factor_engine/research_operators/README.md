# Research operators

This package contains the **explicit opt-in surface** for utilities that are
useful in research but should not appear in normal daily-factor manifests.
The runtime implementations still live in `cleaned_operators`, so there is no
second execution engine or duplicated implementation.

Categories:

- statistical tests and probability distributions;
- matrix, decomposition, PCA and complex-number utilities;
- FFT, wavelet and filtering utilities;
- explicitly unsafe/non-causal/random helpers via `include_unsafe=True`.

```python
from research_operators import build_research_dsl_allowlist

research_dsl = build_research_dsl_allowlist()
unsafe_dsl = build_research_dsl_allowlist(include_unsafe=True)
```

These operators are excluded from `api.operator_registry.build_dsl_allowlist()`
and therefore cannot be submitted as normal production/research factor DSL.
