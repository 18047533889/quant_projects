"""Focused tests for the local-only research COS CLI preflight."""
from __future__ import annotations

import os

from quant_evaluator.scripts import research_cos_cli_preflight as preflight


def test_configured_absolute_executable_resolves_without_execution_or_disclosure(
        tmp_path, monkeypatch, capsys):
    executable = tmp_path / "safe-test-cli"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o700)
    monkeypatch.setenv("DATA_ACCESS_COS_CLI", str(executable))

    result = preflight.preflight_research_cos_cli()

    assert result == {
        "configured_cli_present": True,
        "configured_cli_resolvable": True,
        "reason_type": "resolved",
    }
    assert str(executable) not in repr(result)
    assert capsys.readouterr().out == ""


def test_missing_configured_cli_fails_closed_without_disclosing_entry(monkeypatch):
    configured = "research-cli-definitely-absent-oct04"
    monkeypatch.setenv("DATA_ACCESS_COS_CLI", configured)

    result = preflight.preflight_research_cos_cli()

    assert result == {
        "configured_cli_present": True,
        "configured_cli_resolvable": False,
        "reason_type": "not_found_or_not_executable",
    }
    assert configured not in repr(result)


def test_non_executable_absolute_cli_is_not_resolvable(tmp_path, monkeypatch):
    executable = tmp_path / "not-executable"
    executable.write_text("placeholder", encoding="utf-8")
    executable.chmod(0o600)
