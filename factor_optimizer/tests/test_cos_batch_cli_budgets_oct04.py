"""CLI budget arguments reach the loader before any COS work starts."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


EXAMPLE_PATH = Path(__file__).parents[1] / "examples" / "cos_batch_audit.py"
SPEC = importlib.util.spec_from_file_location("cos_batch_cli_budget_audit", EXAMPLE_PATH)
EXAMPLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXAMPLE)
MIB = 1024**2


def _run_cli(monkeypatch, capsys, argv):
    calls = []

    def loader(n_factors=2, n_assets=256, *, include_lineages=False,
               manifest_uri=None, max_factor_bytes=8*MIB,
               max_batch_factor_bytes=128*MIB, coverage_policy="isolate"):
        calls.append({
            "n_factors": n_factors, "n_assets": n_assets,
            "include_lineages": include_lineages, "manifest_uri": manifest_uri,
            "max_factor_bytes": max_factor_bytes,
            "max_batch_factor_bytes": max_batch_factor_bytes,
            "coverage_policy": coverage_policy,
        })
        return object(), object(), {}, {}

    monkeypatch.setattr(EXAMPLE, "load_cos_sample", loader)
    monkeypatch.setattr(EXAMPLE, "diagnose_training_batch", lambda *a, **kw: {})
    monkeypatch.setattr(sys, "argv", ["cos_batch_audit.py", *argv])
    EXAMPLE.main()
    json.loads(capsys.readouterr().out)
    return calls


@pytest.mark.parametrize(
    "argv,expected_factor,expected_batch",
    [([], 8*MIB, 128*MIB),
     (["--max-factor-mib", "128", "--max-batch-mib", "2048"],
      128*MIB, 2048*MIB)],
)
def test_cli_budgets_reach_loader_and_keep_defaults(
        monkeypatch, capsys, argv, expected_factor, expected_batch):
    calls = _run_cli(monkeypatch, capsys, argv)

    assert len(calls) == 1
    assert calls[0]["max_factor_bytes"] == expected_factor
    assert calls[0]["max_batch_factor_bytes"] == expected_batch


@pytest.mark.parametrize(
    "argv",
    [
        ["--max-factor-mib", "129"],
        ["--max-batch-mib", "2049"],
        ["--max-factor-mib", "0"],
        ["--max-batch-mib", "0"],
    ],
)
def test_cli_rejects_out_of_range_budgets_before_loader(monkeypatch, capsys, argv):
    calls = []
    monkeypatch.setattr(EXAMPLE, "load_cos_sample", lambda **kw: calls.append(kw))
    monkeypatch.setattr(EXAMPLE, "diagnose_training_batch", lambda *a, **kw: {})
