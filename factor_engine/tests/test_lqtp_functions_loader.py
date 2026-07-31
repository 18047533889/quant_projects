from __future__ import annotations

from pathlib import Path

import pytest


def test_external_functions_yaml_alias_and_template(tmp_path: Path) -> None:
    from api.operator_registry import build_dsl_allowlist
    from api.lqtp_functions_loader import augment_from_functions_yaml
    from api.columns import col

    path = tmp_path / "functions.yaml"
    path.write_text(
        """
aliases:
  rolling_mean_external: ts_mean
templates:
  delta_plus_one:
    params: [x, n]
    expression: ts_delta(x, n) + 1
""",
        encoding="utf-8",
    )
    base = build_dsl_allowlist(surface="compat_research", dialect="lqtp")
    allow = augment_from_functions_yaml(base, path=path)
    assert allow["rolling_mean_external"](col("close"), 5).op == "ts_mean"
    expr = allow["delta_plus_one"](col("close"), 5)
    assert expr.op == "add"
    assert expr.args[0].op == "ts_delta"


def test_external_template_rejects_unbound_fields_and_unknown_functions(tmp_path: Path) -> None:
    from api.operator_registry import build_dsl_allowlist
    from api.lqtp_functions_loader import augment_from_functions_yaml, LQTPFunctionsConfigError
    from api.columns import col

    base = build_dsl_allowlist(surface="compat_research", dialect="lqtp")
    bad_field = tmp_path / "bad_field.yaml"
    bad_field.write_text(
        "templates:\n  bad:\n    params: [x]\n    expression: x + hidden_column\n",
        encoding="utf-8",
    )
    allow = augment_from_functions_yaml(base, path=bad_field)
    with pytest.raises(LQTPFunctionsConfigError, match="unbound bare name"):
        allow["bad"](col("close"))

    bad_call = tmp_path / "bad_call.yaml"
    bad_call.write_text(
        "templates:\n  bad:\n    params: [x]\n    expression: system_exec(x)\n",
        encoding="utf-8",
    )
    allow = augment_from_functions_yaml(base, path=bad_call)
    with pytest.raises(LQTPFunctionsConfigError, match="unregistered function"):
        allow["bad"](col("close"))
