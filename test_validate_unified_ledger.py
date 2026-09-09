import json, subprocess, sys
import pytest
from pathlib import Path

ROOT=Path(__file__).resolve().parent
VALIDATOR=ROOT/"validate_unified_ledger.py"
SPEC=ROOT/"docs/V3_REMEDIATION_SPEC_20260906.md"
GOOD=ROOT/"evidence/r2/v3_177_unified_ledger.json"

def run(path):
    return subprocess.run([sys.executable,str(VALIDATOR),str(path),str(SPEC)],capture_output=True,text=True)

def test_validator_accepts_truthful_complete_inventory():
    assert run(GOOD).returncode == 0

def test_validator_rejects_missing_177_id(tmp_path):
    d=json.loads(GOOD.read_text()); d["items"].pop(); p=tmp_path/"bad.json"; p.write_text(json.dumps(d))
    result=run(p); assert result.returncode != 0 and "missing=" in result.stdout

def test_validator_rejects_false_closed_and_missing_evidence(tmp_path):
    d=json.loads(GOOD.read_text()); d["items"][0]["status"]="CLOSED"; d["items"][0]["tests"]=[]
    p=tmp_path/"bad.json"; p.write_text(json.dumps(d)); result=run(p)
    assert result.returncode != 0 and "falsely CLOSED" in result.stdout

@pytest.mark.parametrize("status", ["VERIFIED", "INTEGRATION_VERIFIED", "ALREADY_FIXED_VERIFIED"])
def test_qualified_verified_also_requires_public_path_tests_and_history(tmp_path, status):
    d=json.loads(GOOD.read_text()); d["items"][0]["status"]=status; d["items"][0]["public_paths"]=[]
    p=tmp_path/"bad.json"; p.write_text(json.dumps(d)); result=run(p)
    assert result.returncode != 0 and f"falsely {status}" in result.stdout
