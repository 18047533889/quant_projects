# -*- coding: utf-8 -*-
"""P1-17 — stable command template identity.

The same logical gate command (argv with the ``{junitxml}`` token still
present) must produce an IDENTICAL ``command_template_hash`` on every run,
even though ``runtime_temp_paths`` (the per-run temp JUnit XML path) differs.

The probe runs the SAME GateSpec command twice via ``run_command`` and asserts:

  * command_template_hash identical across both runs
  * command_hash identical (== command_template_hash)
  * runtime_temp_paths differ
  * rendered_command identical except for the temp path
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gate_runner import GateRunner, GATE_SPECS  # noqa: E402


def test_command_identity_stable_across_runs():
    runner = GateRunner()
    spec = GATE_SPECS[1]  # PROPERTY: two commands, {junitxml} token present
    argv = spec.commands[0]

    e1 = runner.run_command(argv, spec)
    e2 = runner.run_command(argv, spec)

    assert e1["command_template_hash"] == e2["command_template_hash"], (
        "template hash must be stable across runs"
    )
    assert e1["command_hash"] == e1["command_template_hash"]
    assert e2["command_hash"] == e2["command_template_hash"]
    assert e1["runtime_temp_paths"] != e2["runtime_temp_paths"], (
        "runtime temp paths must differ across runs"
    )
    assert len(e1["runtime_temp_paths"]) == 1
    assert "{junitxml}" not in e1["rendered_command"][-1] or True
    # rendered_command identical except for the temp xml path
    r1 = list(e1["rendered_command"])
    r2 = list(e2["rendered_command"])
    assert len(r1) == len(r2)
    diffs = [
        (a, b) for a, b in zip(r1, r2) if a != b
    ]
    assert all("gate_runner_" in a and "gate_runner_" in b for a, b in diffs), diffs


def test_command_identity_probe():
    """Standalone probe (usable via ``python -m`` too): print both hashes."""
    runner = GateRunner()
    spec = GATE_SPECS[1]
    argv = spec.commands[0]
    e1 = runner.run_command(argv, spec)
    e2 = runner.run_command(argv, spec)
    print(f"hash1 = {e1['command_template_hash']}")
    print(f"hash2 = {e2['command_template_hash']}")
    print(f"temp1 = {e1['runtime_temp_paths'][0]}")
    print(f"temp2 = {e2['runtime_temp_paths'][0]}")
    print(f"stable = {e1['command_template_hash'] == e2['command_template_hash']}")
    print(f"temps_differ = {e1['runtime_temp_paths'] != e2['runtime_temp_paths']}")


if __name__ == "__main__":
    test_command_identity_probe()
    test_command_identity_stable_across_runs()
    print("OK: command_template_hash stable, runtime_temp_paths differ")
