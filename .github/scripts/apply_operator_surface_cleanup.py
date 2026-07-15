from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FE = ROOT / "factor_engine"


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"expected migration anchor missing in {path}: {old[:80]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


OPERATOR_SURFACE = r'''# -*- coding: utf-8 -*-
"""Public DSL surface policy for daily factor generation.

The runtime registry remains the compatibility and execution index.  This
module controls which names may be used in newly submitted DSL formulas.
"""
from __future__ import annotations

from typing import Literal

OperatorSurface = Literal["daily", "research", "unsafe", "legacy", "all"]

# Operators that are non-causal, backward-looking from the future, or random.
# They remain importable only from ``research_operators`` with explicit unsafe
# opt-in, never from the normal factor DSL.
UNSAFE_CANONICALS: frozenset[str] = frozenset(
    {
        "Lead",
        "next",
        "bfill",
        "fillna_interpolate",
        "shuffle",
        "sample",
        "rand_exp",
        "rand_lognormal",
        "rand_normal",
        "rand_poisson",
        "rand_uniform",
    }
)

# Useful research/statistical utilities, but not operators whose normal output
# is a daily scalar factor panel.  They are exposed by ``research_operators``.
RESEARCH_ONLY_CANONICALS: frozenset[str] = frozenset(
    {
        # Statistical tests and diagnostics.
        "bartlett_test",
        "chi_square_test",
        "corr_test",
        "durbin_watson_test",
        "granger_causality",
        "jarque_bera_test",
        "kendall_corr_test",
        "kpss_test",
        "ks_test",
        "levene_test",
        "lilliefors_test",
        "spearman_corr_test",
        "stationarity_test",
        "ttest_one_sample",
        "ttest_paired",
        "ttest_two_samples",
        # Probability distributions.
        "cdf_chi2",
        "cdf_f",
        "cdf_normal",
        "cdf_t",
        "pdf_chi2",
        "pdf_f",
        "pdf_normal",
        "pdf_t",
        "quantile_normal",
        "quantile_t",
        # Complex numbers and linear algebra.
        "complex",
        "conj",
        "real",
        "imag",
        "polar",
        "phase",
        "eig",
        "svd",
        "pca",
        "lu_decompose",
        "qr_decompose",
        "mat_add",
        "mat_subtract",
        "mat_multiply",
        "mat_transpose",
        "mat_inverse",
        "mat_determinant",
        "mat_rank",
        "norm",
        "norm_l1",
        "norm_linf",
        # Signal processing.  Boundary and causality policy must be explicit.
        "fft",
        "ifft",
        "wavelet",
        "wavelet_denoise",
        "convolve",
        "correlate",
        "decimate",
        "filter_lowpass",
        "filter_highpass",
        "filter_bandpass",
        "filter_notch",
        "interpolate",
        "unwrap",
    }
)

# Old convenience canonicals retained for direct registry compatibility.  New
# formulas use the canonical on the right side of ``_dedupe.py`` instead.
LEGACY_ONLY_CANONICALS: frozenset[str] = frozenset(
    {
        "inv",
        "reciprocal",
        "fmax",
        "fmin",
        "sqr",
        "cube",
        "cumulative_max",
        "cumulative_mean",
        "cumulative_min",
    }
)

# Alias names that resolve to an allowed canonical after deduplication but must
# still be hidden from the normal daily DSL.
HIDDEN_DAILY_NAMES: frozenset[str] = frozenset(
    {
        "inv",
        "reciprocal",
        "fmax",
        "fmin",
        "sqr",
        "cube",
        "cumulative_max",
        "cumulative_mean",
        "cumulative_min",
    }
)


def classify_canonical(canonical: str) -> str:
    """Return the single public surface classification for a canonical name."""
    if canonical in UNSAFE_CANONICALS:
        return "unsafe"
    if canonical in RESEARCH_ONLY_CANONICALS:
        return "research"
    if canonical in LEGACY_ONLY_CANONICALS:
        return "legacy"
    return "daily"


def is_dsl_name_allowed(name: str, canonical: str, *, surface: OperatorSurface = "daily") -> bool:
    """Return whether a registry name is visible on the requested DSL surface."""
    if surface == "all":
        return True
    category = classify_canonical(canonical)
    if surface == "daily":
        return category == "daily" and name not in HIDDEN_DAILY_NAMES
    if surface == "research":
        return category == "research"
    if surface == "unsafe":
        return category == "unsafe"
    if surface == "legacy":
        return category == "legacy" or name in HIDDEN_DAILY_NAMES
    raise ValueError(f"unknown operator surface: {surface!r}")


def surface_summary(canonicals: list[str]) -> dict[str, int]:
    """Count canonical operators by surface classification."""
    out = {"daily": 0, "research": 0, "unsafe": 0, "legacy": 0}
    for canonical in canonicals:
        out[classify_canonical(canonical)] += 1
    return out
'''

RESEARCH_INIT = r'''"""Explicit research-only operator access.

This package is intentionally separate from ``api``.  Importing it is an
explicit acknowledgement that the selected operators are not part of the
normal daily-factor DSL and are not production-certified.
"""
from __future__ import annotations

from typing import Any

from backend.cleaned_bridge import build_cleaned_dsl_allowlist, ensure_cleaned_loaded
from cleaned_operators.operator_surface import (
    RESEARCH_ONLY_CANONICALS,
    UNSAFE_CANONICALS,
)


def build_research_dsl_allowlist(*, include_unsafe: bool = False) -> dict[str, Any]:
    """Build the research utility DSL surface, optionally including unsafe tools."""
    out = build_cleaned_dsl_allowlist(surface="research")
    if include_unsafe:
        out.update(build_cleaned_dsl_allowlist(surface="unsafe"))
    return out


def build_unsafe_dsl_allowlist() -> dict[str, Any]:
    """Build the explicit non-causal/random utility surface."""
    return build_cleaned_dsl_allowlist(surface="unsafe")


def get_research_operator(name: str, *, backend: str = "pandas_numpy"):
    """Return a research operator runtime after validating its surface."""
    ensure_cleaned_loaded()
    from cleaned_operators.registry import OperatorRegistry

    canonical = OperatorRegistry._aliases.get(name, name)
    if canonical not in RESEARCH_ONLY_CANONICALS:
        raise KeyError(f"{name!r} is not a research-only operator")
    operator = OperatorRegistry.get(canonical, backend=backend)
    if operator is None:
        raise KeyError(f"research operator {name!r} has no {backend!r} runtime")
    return operator


def get_unsafe_operator(name: str, *, backend: str = "pandas_numpy"):
    """Return an unsafe operator only through explicit opt-in."""
    ensure_cleaned_loaded()
    from cleaned_operators.registry import OperatorRegistry

    canonical = OperatorRegistry._aliases.get(name, name)
    if canonical not in UNSAFE_CANONICALS:
        raise KeyError(f"{name!r} is not an unsafe operator")
    operator = OperatorRegistry.get(canonical, backend=backend)
    if operator is None:
        raise KeyError(f"unsafe operator {name!r} has no {backend!r} runtime")
    return operator


__all__ = [
    "build_research_dsl_allowlist",
    "build_unsafe_dsl_allowlist",
    "get_research_operator",
    "get_unsafe_operator",
]
'''

RESEARCH_README = r'''# Research operators

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
'''

OPERATOR_SURFACES_DOC = r'''# Operator surfaces

`factor_engine` now separates operator **runtime availability** from operator
**DSL submission eligibility**.

| Surface | Purpose | Normal manifest access |
|---|---|---|
| `daily` | Causal scalar operators that generate daily factor panels | Yes |
| `research` | Statistical tests, distributions, matrix/PCA and signal processing | No; import `research_operators` explicitly |
| `unsafe` | Lead/backfill/random/non-deterministic helpers | No; explicit unsafe opt-in only |
| `legacy` | Redundant historical names retained for direct runtime compatibility | No; migrate to canonical names |

The runtime registry still contains compatibility implementations so old
results can be reproduced.  New formulas are validated only against the
`daily` surface.

## Canonical replacements

| Removed public DSL name | Use instead |
|---|---|
| `inv`, `reciprocal` | `inverse` |
| `fmax` | `maximum` |
| `fmin` | `minimum` |
| `sqr` | `square` or `power(x, 2)` |
| `cube` | `power(x, 3)` |
| `cumulative_max` | `expanding_max` |
| `cumulative_min` | `expanding_min` |
| `cumulative_mean` | `expanding_mean` |

`Lead`, `next`, `bfill`, interpolation using future observations, random
sampling and random-number operators are never available through the normal
factor DSL.
'''

API_OPERATOR_REGISTRY = r'''"""Public daily-factor DSL allowlist.

The registry contains additional compatibility and research runtimes, but new
factor formulas only see the causal daily-factor surface.
"""
from __future__ import annotations

from typing import Any, Callable

from api.columns import col
from backend.cleaned_bridge import build_cleaned_dsl_allowlist

STUB_IR_OPS: frozenset[str] = frozenset()


def build_dsl_allowlist() -> dict[str, Callable[..., Any]]:
    """Return the public causal daily-factor DSL function mapping."""
    allow: dict[str, Callable[..., Any]] = {"col": col}
    allow.update(build_cleaned_dsl_allowlist(set(), surface="daily"))
    return allow
'''

CATALOG_GENERATOR = r'''#!/usr/bin/env python3
"""Generate post-deduplication operator runtime and surface catalogs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

FE_ROOT = Path(__file__).resolve().parents[1]
OUT_MD = FE_ROOT / "cleaned_operators" / "docs" / "operators_catalog.md"
OUT_JSON = FE_ROOT / "cleaned_operators" / "docs" / "operators_catalog.json"


def _load():
    root = str(FE_ROOT.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    if str(FE_ROOT) not in sys.path:
        sys.path.insert(0, str(FE_ROOT))
    from cleaned_operators import load_all
    from cleaned_operators.operator_policy import infer_operator_policy
    from cleaned_operators.operator_surface import classify_canonical, surface_summary
    from cleaned_operators.registry import OperatorRegistry
    from backend.sql_pushdown.sql_registry import register_sql_backends

    load_all()  # includes apply_operator_deduplication()
    register_sql_backends()
    return OperatorRegistry, infer_operator_policy, classify_canonical, surface_summary


def _payload(registry, infer_policy, classify_canonical, surface_summary) -> dict:
    rows: list[dict] = []
    for canonical in registry.list_canonical():
        operator = registry.get(canonical)
        if operator is None:
            continue
        meta = dict(registry.catalog().get(canonical) or {})
        policy = infer_policy(operator, canonical=canonical)
        rows.append(
            {
                "canonical": canonical,
                "surface": classify_canonical(canonical),
                "aliases": sorted(meta.get("aliases") or []),
                "backends": sorted(meta.get("backends") or []),
                "pit_safe": bool(policy.pit_safe) if policy else None,
                "scope": str(policy.scope) if policy else None,
                "lookback": policy.lookback_window if policy else None,
                "min_periods": policy.min_periods if policy else None,
                "lag": policy.lag if policy else None,
            }
        )
    return {
        "schema_version": 2,
        "canonical_count": len(rows),
        "surface_counts": surface_summary([row["canonical"] for row in rows]),
        "operators": rows,
    }


def _render(payload: dict) -> str:
    counts = payload["surface_counts"]
    lines = [
        "# Operators Catalog（自动生成）",
        "",
        "> 从 `load_all()` 去重后的最终 runtime registry 生成。",
        "> 日常因子 DSL 仅使用 `surface=daily`；其他工具见 `research_operators/`。",
        "",
        "## 摘要",
        "",
        f"- canonical 总数：{payload['canonical_count']}",
        f"- daily：{counts['daily']}",
        f"- research：{counts['research']}",
        f"- unsafe：{counts['unsafe']}",
        f"- legacy：{counts['legacy']}",
        "",
        "| canonical | surface | backends | aliases | pit_safe | scope | lookback | min_periods | lag |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in payload["operators"]:
        lines.append(
            "| {canonical} | {surface} | {backends} | {aliases} | {pit_safe} | {scope} | {lookback} | {min_periods} | {lag} |".format(
                canonical=row["canonical"],
                surface=row["surface"],
                backends=", ".join(row["backends"]),
                aliases=", ".join(row["aliases"]),
                pit_safe=row["pit_safe"],
                scope=row["scope"],
                lookback=row["lookback"],
                min_periods=row["min_periods"],
                lag=row["lag"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    registry, infer_policy, classify_canonical, surface_summary = _load()
    payload = _payload(registry, infer_policy, classify_canonical, surface_summary)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(_render(payload), encoding="utf-8")
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_MD} and {OUT_JSON}")


if __name__ == "__main__":
    main()
'''

DSL_REFERENCE = r'''# DSL 算子白名单参考

> **权威枚举**：[`dsl_allowlist.json`](dsl_allowlist.json)，由 `build_dsl_allowlist()` 导出。  
> **完整 runtime 与分类**：[`../cleaned_operators/docs/operators_catalog.md`](../cleaned_operators/docs/operators_catalog.md)。  
> **分层说明**：[`operator_surfaces.md`](operator_surfaces.md)。

## 日常因子 DSL

正常 manifest 只能使用 `surface=daily` 的算子，即能够生成按
`(date, instrument)` 对齐的因子值且默认满足因果约束的算子。代表性能力：

- 算术与安全数学：`add`, `subtract`, `multiply`, `divide`, `safe_div`, `log`, `sqrt`, `power`；
- 时序：`delay`, `ts_delta`, `ts_mean`, `ts_std`, `ts_sum`, `ts_min`, `ts_max`, `ts_corr`, `ts_rank`；
- 截面：`rank`, `zscore`, `winsorize`, `normalize`, `cs_demean`, `cs_regression`；
- 分组：`group_rank`, `group_neutralize`, `group_zscore`, `group_mean`；
- 技术与领域算子：保留能直接生成标量因子面板的技术、基本面和微观结构算子。

## 不在公共 DSL 中

- 非因果/随机：`Lead`, `next`, `bfill`, `fillna_interpolate`, `shuffle`, `sample`, `rand_*`；
- 研究诊断：假设检验、PDF/CDF、矩阵分解、PCA、复数、FFT、wavelet 和 filtering；
- 冗余旧名：`inv`, `reciprocal`, `fmax`, `fmin`, `sqr`, `cube`, `cumulative_*`。

研究工具通过 `research_operators` 显式访问，不得写入正常因子 manifest。

```bash
cd factor_engine
PYTHONPATH=. python scripts/export_dsl_allowlist.py
PYTHONPATH=. python scripts/generate_operators_catalog.py
```
'''

PUBLIC_FORMULA_VALIDATOR = r'''#!/usr/bin/env python3
"""Validate repository factor definitions against the public daily DSL."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterator

import yaml

ROOT = Path(__file__).resolve().parents[2]
FE = ROOT / "factor_engine"
for path in (ROOT, FE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from api.dsl_parser import parse_expr  # noqa: E402

FORMULA_KEYS = {"expr", "formula", "dsl_formula", "expression"}
SKIP_PARTS = {"archive", "output", "reports", ".git"}
ROOTS = (
    ROOT / "gtja191" / "candidate_pool",
    ROOT / "week2_pv_factors" / "candidate_pool",
    FE / "examples" / "configs",
)


def walk(value: Any, *, key: str = "") -> Iterator[str]:
    if isinstance(value, dict):
        for child_key, child in value.items():
            if child_key in FORMULA_KEYS and isinstance(child, str) and child.strip():
                yield child.strip()
            else:
                yield from walk(child, key=str(child_key))
    elif isinstance(value, list):
        for child in value:
            yield from walk(child, key=key)


def load(path: Path) -> Any:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> int:
    checked = 0
    failures: list[str] = []
    for root in ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in {".json", ".yaml", ".yml"}:
                continue
            if any(part in SKIP_PARTS for part in path.parts):
                continue
            try:
                payload = load(path)
            except Exception:
                continue
            for formula in walk(payload):
                checked += 1
                try:
                    parse_expr(formula)
                except Exception as exc:
                    failures.append(f"{path.relative_to(ROOT)}: {type(exc).__name__}: {exc}: {formula}")
    if failures:
        print("\n".join(failures[:100]))
        raise SystemExit(f"{len(failures)} public factor formulas failed validation")
    print(f"validated {checked} repository factor formulas against the daily DSL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

SURFACE_TEST = r'''from __future__ import annotations

import pytest

from api.operator_registry import build_dsl_allowlist
from backend.cleaned_bridge import build_cleaned_dsl_allowlist
from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry
from research_operators import (
    build_research_dsl_allowlist,
    build_unsafe_dsl_allowlist,
)


def test_daily_surface_keeps_factor_primitives() -> None:
    public = build_dsl_allowlist()
    for name in ("col", "abs", "rank", "zscore", "ts_mean", "ts_sum", "ts_corr", "where", "power"):
        assert name in public


@pytest.mark.parametrize(
    "name",
    [
        "Lead", "next", "bfill", "FillBackward", "fillna_interpolate",
        "shuffle", "sample", "rand_normal", "rand_uniform",
        "jarque_bera_test", "ttest_one_sample", "pdf_normal", "pca",
        "mat_inverse", "fft", "wavelet", "filter_lowpass",
        "inv", "reciprocal", "fmax", "fmin", "sqr", "cube",
        "cumulative_max", "cumulative_mean", "cumulative_min",
    ],
)
def test_removed_names_are_not_in_public_daily_dsl(name: str) -> None:
    assert name not in build_dsl_allowlist()


def test_research_tools_are_explicit_and_unsafe_is_separate() -> None:
    public = build_dsl_allowlist()
    research = build_research_dsl_allowlist()
    unsafe = build_unsafe_dsl_allowlist()
    assert "jarque_bera_test" in research
    assert "pca" in research
    assert "fft" in research
    assert "jarque_bera_test" not in public
    assert "next" in unsafe
    assert "bfill" in unsafe
    assert "rand_normal" in unsafe
    assert "next" not in research


def test_all_runtime_surface_remains_available_for_compatibility() -> None:
    all_runtime = build_cleaned_dsl_allowlist(surface="all")
    assert "fft" in all_runtime
    assert "next" in all_runtime
    assert "cube" in all_runtime


def test_duplicate_canonicals_are_merged_to_one_runtime() -> None:
    load_all()
    expected = {
        "inv": "inverse",
        "reciprocal": "inverse",
        "fmax": "maximum",
        "fmin": "minimum",
        "sqr": "square",
        "cumulative_max": "expanding_max",
        "cumulative_mean": "expanding_mean",
        "cumulative_min": "expanding_min",
    }
    canonicals = set(OperatorRegistry.list_canonical())
    for alias, canonical in expected.items():
        assert OperatorRegistry._aliases.get(alias) == canonical
        assert alias not in canonicals
        assert OperatorRegistry.get(canonical) is not None


def test_api_dynamic_import_cannot_bypass_public_surface() -> None:
    import api

    with pytest.raises(AttributeError):
        getattr(api, "fft")
    with pytest.raises(AttributeError):
        getattr(api, "next")
    assert callable(getattr(api, "ts_mean"))
'''

PERMANENT_WORKFLOW = r'''name: Operator Surface Gate

on:
  push:
    branches: [main, agent/operator-surface-cleanup]
    paths:
      - "factor_engine/**"
      - "gtja191/**"
      - "week2_pv_factors/**"
      - ".github/workflows/operator-surface-gate.yml"
  pull_request:
    paths:
      - "factor_engine/**"
      - "gtja191/**"
      - "week2_pv_factors/**"
      - ".github/workflows/operator-surface-gate.yml"

permissions:
  contents: read

concurrency:
  group: operator-surface-${{ github.ref }}
  cancel-in-progress: true

jobs:
  public-dsl:
    runs-on: ubuntu-latest
    timeout-minutes: 35
    env:
      PYTHONPATH: ${{ github.workspace }}:${{ github.workspace }}/factor_engine:${{ github.workspace }}/gtja191
      FACTOR_ENGINE_DISABLE_NUMBA: "1"
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          python -m pip install -r requirements.txt
      - name: Validate operator surfaces and generated catalogs
        run: |
          python factor_engine/scripts/generate_operators_catalog.py
          python factor_engine/scripts/export_dsl_allowlist.py
          git diff --exit-code -- \
            factor_engine/cleaned_operators/docs/operators_catalog.md \
            factor_engine/cleaned_operators/docs/operators_catalog.json \
            factor_engine/docs/dsl_allowlist.json
          pytest -q factor_engine/tests/test_operator_surface.py
          python factor_engine/scripts/validate_public_factor_formulas.py
      - name: Factor pack and engine regression
        run: |
          pytest -q \
            gtja191/tests/test_full_catalog_factor_engine.py \
            gtja191/tests/test_gtja191.py \
            factor_engine/tests/test_gtja_compat_semantics.py \
            factor_engine/tests/test_data_scope_stability.py
'''

# New policy and explicit research package.
write(FE / "cleaned_operators" / "operator_surface.py", OPERATOR_SURFACE)
write(FE / "research_operators" / "__init__.py", RESEARCH_INIT)
write(FE / "research_operators" / "README.md", RESEARCH_README)
write(FE / "docs" / "operator_surfaces.md", OPERATOR_SURFACES_DOC)
write(FE / "api" / "operator_registry.py", API_OPERATOR_REGISTRY)
write(FE / "scripts" / "generate_operators_catalog.py", CATALOG_GENERATOR)
write(FE / "scripts" / "validate_public_factor_formulas.py", PUBLIC_FORMULA_VALIDATOR)
write(FE / "tests" / "test_operator_surface.py", SURFACE_TEST)
write(FE / "docs" / "dsl_operators_reference.md", DSL_REFERENCE)
write(ROOT / ".github" / "workflows" / "operator-surface-gate.yml", PERMANENT_WORKFLOW)

# Public allowlist filtering in the cleaned bridge.
bridge = FE / "backend" / "cleaned_bridge.py"
text = bridge.read_text(encoding="utf-8")
start = text.index("def build_cleaned_dsl_allowlist")
end = text.index("\ndef build_production_dsl_allowlist", start)
new_block = r'''def build_cleaned_dsl_allowlist(
    skip: set[str] | None = None,
    *,
    surface: str = "daily",
) -> dict[str, Any]:
    """Build a surface-filtered DSL mapping from the runtime registry.

    ``daily`` is the normal factor submission surface.  ``research``,
    ``unsafe`` and ``legacy`` are explicit opt-in surfaces; ``all`` exists only
    for compatibility tooling and runtime audits.
    """
    ensure_cleaned_loaded()
    from cleaned_operators.operator_surface import is_dsl_name_allowed
    from cleaned_operators.registry import OperatorRegistry
    from api.cleaned_ops import make_cleaned_call_factory

    skip = skip or set()
    out: dict[str, Any] = {}
    for canon in OperatorRegistry.list_canonical():
        if canon in skip or canon in out:
            continue
        if OperatorRegistry.get(canon) is None:
            continue
        if not is_dsl_name_allowed(canon, canon, surface=surface):
            continue
        out[canon] = make_cleaned_call_factory(canon)
    for alias, canon in OperatorRegistry._aliases.items():
        if alias in skip or alias in out:
            continue
        if OperatorRegistry.get(canon) is None:
            continue
        if not is_dsl_name_allowed(alias, canon, surface=surface):
            continue
        out[alias] = make_cleaned_call_factory(canon)
    return out

'''
text = text[:start] + new_block + text[end + 1 :]
text = text.replace(
    "full = build_cleaned_dsl_allowlist(skip)",
    'full = build_cleaned_dsl_allowlist(skip, surface="daily")',
)
bridge.write_text(text, encoding="utf-8")

# ``from api import name`` must not bypass the daily allowlist.
api_init = FE / "api" / "__init__.py"
replace_once(
    api_init,
    '''    from backend.cleaned_bridge import ensure_cleaned_loaded\n\n    ensure_cleaned_loaded()\n    from cleaned_operators.registry import OperatorRegistry\n\n    canon = OperatorRegistry._aliases.get(name, name)\n    if OperatorRegistry.get(canon) is None:\n        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")\n    factory = make_cleaned_call_factory(name)\n''',
    '''    from api.operator_registry import build_dsl_allowlist\n\n    if name not in build_dsl_allowlist():\n        raise AttributeError(\n            f"module {__name__!r} has no public daily-factor operator {name!r}"\n        )\n    factory = make_cleaned_call_factory(name)\n''',
)

# Merge unused duplicate canonicals while retaining old names as registry aliases.
dedupe = FE / "cleaned_operators" / "_dedupe.py"
replace_once(dedupe, '    "reciprocal": "inv",\n', '    "inv": "inverse",\n    "reciprocal": "inverse",\n')
replace_once(
    dedupe,
    '    "if_else": "where",\n',
    '    "if_else": "where",\n    "fmax": "maximum",\n    "fmin": "minimum",\n    "sqr": "square",\n    "cumulative_max": "expanding_max",\n    "cumulative_mean": "expanding_mean",\n    "cumulative_min": "expanding_min",\n',
)
replace_once(
    dedupe,
    '    "reciprocal",\n',
    '    "inv",\n    "reciprocal",\n    "fmax",\n    "fmin",\n    "sqr",\n    "cumulative_max",\n    "cumulative_mean",\n    "cumulative_min",\n',
)

# Update the top-level guide without publishing a misleading raw runtime count.
readme = FE / "README.md"
text = readme.read_text(encoding="utf-8")
old = "**第 31 版起**，算子 runtime 统一在 **`cleaned_operators/`**（约 **440+** 个已实现 canonical + **116** 个别名）。DSL 白名单由 **`api.operator_registry.build_dsl_allowlist()`** 自动生成（`col` + 全部有 pandas runtime 的名字）。"
new = "算子 runtime 统一在 **`cleaned_operators/`**；正常 manifest 只暴露可生成日频标量因子值的 **daily DSL surface**。统计检验、矩阵/PCA、信号处理和非因果工具移至 **`research_operators/`** 显式调用。最终数量与分类以 `cleaned_operators/docs/operators_catalog.json` 为准。"
if old not in text:
    raise RuntimeError("README operator surface anchor missing")
readme.write_text(text.replace(old, new, 1), encoding="utf-8")

print("operator surface migration applied")
