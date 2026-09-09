#!/usr/bin/env python3
"""Build clean local wheels and compare their imports with checkout imports."""

from __future__ import annotations

import argparse
import ast
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


PROJECTS = {
    "data-access": ("data_access", "data_access"),
    "factor-engine": ("factor_engine", "factor_engine"),
    "factor-preprocess": ("factor_preprocess", "factor_preprocess"),
    "quant_evaluator": ("quant_evaluator", "quant_evaluator"),
    "factor-optimizer": ("factor_optimizer", "factor_optimizer"),
    "factor_assets": ("factor_assets", "factor_assets"),
    "quant-modeling": ("modeling", "modeling"),
    "quant-platform": ("quant_platform", "quant_platform"),
}


def run(argv, *, cwd, env=None):
    return subprocess.run(argv, cwd=cwd, env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def compare_package_sources(installed_root: Path, source_root: Path) -> dict:
    """Compare every installed Python implementation with its checkout file."""
    differences, digests = [], {}
    for installed in sorted(installed_root.rglob('*.py')):
        relative = installed.relative_to(installed_root)
        source = source_root / relative
        digest = hashlib.sha256(installed.read_bytes()).hexdigest()
        digests[str(relative)] = digest
        if not source.is_file():
            differences.append({'file': str(relative), 'reason': 'missing_checkout_source'})
        elif hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            differences.append({'file': str(relative), 'reason': 'different_implementation'})
    return {'status': 'PASS' if digests and not differences else 'BLOCKED',
            'compared_files': len(digests), 'differences': differences,
            'installed_source_sha256': digests}


def parse_probe_output(output: str) -> dict:
    """Noise before the final JSON receipt is allowed; malformed receipts fail."""
    try:
        receipt = json.loads(output.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {}
    return receipt if isinstance(receipt, dict) else {}


def reconcile_collected_tests(source: str, nodeids: list[str]) -> dict:
    """Compare pytest's real collection against definitions, before parametrization."""
    expected, duplicates = [], []

    def visit(body, prefix=""):
        seen = set()
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                name = prefix + node.name
                if name in seen:
                    duplicates.append(name)
                seen.add(name)
                expected.append(name)
            elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                visit(node.body, prefix + node.name + "::")

    visit(ast.parse(source).body)
    observed = {nodeid.split("::", 1)[1].split("[", 1)[0]
                for nodeid in nodeids if "::" in nodeid}
    missing, unexpected = sorted(set(expected) - observed), sorted(observed - set(expected))
    return {"status": "PASS" if expected and not duplicates and not missing and not unexpected else "BLOCKED",
            "definition_count": len(expected), "collected_case_count": len(nodeids),
            "collected_definition_count": len(observed), "missing": missing,
            "unexpected": unexpected, "duplicate_definitions": duplicates,
            "nodeids": nodeids}


def collect_parity_evidence(repo: Path) -> dict:
    """Required FE bridge tests must actually be discoverable, without skips."""
    relative = "factor_preprocess/tests/test_fe_operator_parity.py"
    path = repo / relative
    before = path.read_bytes()
    code = '''
import json, pytest
class Receipt:
    def __init__(self): self.nodeids=[]; self.failures=[]; self.skips=[]
    def pytest_collection_finish(self, session):
        self.nodeids=[item.nodeid for item in session.items]
    def pytest_collectreport(self, report):
        if report.failed: self.failures.append(str(report.longrepr))
        if report.skipped: self.skips.append(str(report.longrepr))
p=Receipt()
rc=pytest.main(["--collect-only", "-q", %r], plugins=[p])
print(json.dumps({"exit_code":int(rc),"nodeids":p.nodeids,"failures":p.failures,"skips":p.skips}))
''' % relative
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(repo / "factor_optimizer"), str(repo / "factor_preprocess"), str(repo)])
    result = run([sys.executable, "-c", code], cwd=repo, env=env)
    receipt = parse_probe_output(result.stdout)
    evidence = reconcile_collected_tests(before.decode(), receipt.get("nodeids", []))
    evidence.update({"file": relative, "source_sha256": hashlib.sha256(before).hexdigest(),
                     "collection": receipt, "process_exit_code": result.returncode})
    if (result.returncode != 0 or receipt.get("exit_code") != 0
            or receipt.get("failures") or receipt.get("skips") or before != path.read_bytes()):
        evidence["status"] = "BLOCKED"
    return evidence


def probe(python: Path, modules, *, cwd: Path, target: Path, env=None):
    code = """
import importlib, importlib.metadata, json, sys
sys.path.insert(0, %r)
items={}
for dist,module in %r:
    try:
        obj=importlib.import_module(module)
        try: version=importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError: version=None
        items[dist]={'status':'IMPORTED','module':module,'file':getattr(obj,'__file__',None),'version':version}
    except Exception as exc:
        items[dist]={'status':'BLOCKED','module':module,'error':type(exc).__name__+': '+str(exc)}
print(json.dumps(items,sort_keys=True))
""" % (str(target), list(modules))
    result = run([str(python), "-I", "-c", code], cwd=cwd, env=env)
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except Exception:
        payload = {"probe": {"status": "BLOCKED", "output": result.stdout}}
    return result.returncode, payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    evidence = {"schema_version": 1, "suite": "QAT-07-import-matrix", "repo": str(repo)}
    duplicates = []
    for path in repo.rglob("test*.py"):
        if any(part in {"build", "dist", ".venv"} for part in path.parts):
            continue
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except (OSError, SyntaxError):
            continue
        scopes = [("module", tree.body)] + [
            (node.name, node.body) for node in tree.body if isinstance(node, ast.ClassDef)
        ]
        for scope, body in scopes:
            seen = {}
            for node in body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                    if node.name in seen:
                        duplicates.append({"file": str(path.relative_to(repo)), "scope": scope,
                                           "name": node.name, "lines": [seen[node.name], node.lineno]})
                    seen[node.name] = node.lineno
    evidence["duplicate_test_definitions"] = duplicates
    evidence["required_parity_collection"] = collect_parity_evidence(repo)
    with tempfile.TemporaryDirectory(prefix="qat07-wheel-") as raw:
        tmp = Path(raw); wheelhouse = tmp / "wheelhouse"; wheelhouse.mkdir()
        builds = {}
        for dist, (project, _) in PROJECTS.items():
            project_dir = repo / project
            result = run([sys.executable, "-m", "build", "--wheel", "--no-isolation",
                          "--outdir", str(wheelhouse)], cwd=project_dir)
            builds[dist] = {"exit_code": result.returncode, "output": result.stdout[-4000:]}
        evidence["wheel_builds"] = builds

        target = tmp / "combined-target"
        wheels = sorted(str(p) for p in wheelhouse.glob("*.whl"))
        install = run([sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(target), *wheels], cwd=tmp)
        code, installed = probe(Path(sys.executable), [(d, m) for d, (_, m) in PROJECTS.items()], cwd=tmp, target=target)
        evidence["combined_wheels"] = {
            "install_target": str(target),
            "install_exit_code": install.returncode,
            "install_output": install.stdout[-4000:],
            "probe_exit_code": code, "imports": installed,
        }

        fp_wheel = next(iter(wheelhouse.glob("factor_preprocess-*.whl")), None)
        standalone = tmp / "fp-standalone-target"
        standalone_install = run([sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(standalone), str(fp_wheel)], cwd=tmp) if fp_wheel else None
        fp_probe = """
import json,sys
sys.path.insert(0,%r)
from factor_preprocess.registry.transforms import get_default_registry
op=get_default_registry().get_execution('forward_fill')
inner=getattr(op,'executor',op)
print(json.dumps({'execution_module':type(inner).__module__,'execution_type':type(inner).__name__,
                  'real_fe_loaded':type(inner).__name__=='FeOperatorExecutor'}))
""" % str(standalone)
        fp_result = run([sys.executable, "-I", "-c", fp_probe], cwd=tmp) if standalone_install else None
        evidence["factor_preprocess_standalone"] = {
            "install_target": str(standalone),
            "install_exit_code": standalone_install.returncode if standalone_install else None,
            "probe_exit_code": fp_result.returncode if fp_result else None,
            "probe_output": fp_result.stdout[-4000:] if fp_result else "wheel unavailable",
            "certification": "RESEARCH_ONLY" if fp_result and fp_result.returncode == 0 else "BLOCKED",
        }

        combo_probe = """
import json, pandas as pd, sys
sys.path.insert(0,%r)
from factor_preprocess.registry.transforms import get_default_registry
frame=pd.DataFrame({'date':[1,2,3,4],'asset_id':[1,1,1,1],'value':[1.0,float('nan'),float('nan'),4.0]})
op=get_default_registry().get_execution('forward_fill'); out=op(values=frame,max_lag=1)
inner=getattr(op,'executor',op)
print(json.dumps({'execution_type':type(inner).__name__,'execution_module':type(inner).__module__,
                  'real_fe_loaded':type(inner).__name__=='FeOperatorExecutor',
                  'effective_parameters':op.effective_parameters,'lag_guard_observed':bool(pd.isna(out.iloc[2])),
                  'output_values':[None if pd.isna(value) else float(value) for value in out]}))
""" % str(target)
        combo = run([sys.executable, "-I", "-c", combo_probe], cwd=tmp)
        combo_receipt = parse_probe_output(combo.stdout)
        evidence["factor_preprocess_with_real_fe"] = {
            "probe_exit_code": combo.returncode, "probe_output": combo.stdout[-4000:],
            "certification": "VERIFIED" if combo.returncode == 0
            and combo_receipt.get('real_fe_loaded') is True
            and combo_receipt.get('lag_guard_observed') is True
            and combo_receipt.get('effective_parameters') == {'max_periods': 1}
            and combo_receipt.get('output_values') == [1.0, 1.0, None, 4.0] else "BLOCKED",
        }

        mono_env = os.environ.copy()
        mono_env["PYTHONPATH"] = os.pathsep.join([str(repo / "factor_optimizer"), str(repo / "factor_preprocess"), str(repo)])
        mono_env["PYTHONNOUSERSITE"] = "1"
        mono_code = """
import importlib,json
mods=%r
print(json.dumps({m:importlib.import_module(m).__file__ for m in mods},sort_keys=True))
""" % [module for _, module in PROJECTS.values()]
        mono = run([sys.executable, "-c", mono_code], cwd=tmp, env=mono_env)
        evidence["monorepo"] = {"probe_exit_code": mono.returncode, "probe_output": mono.stdout[-8000:]}
        mono_paths = parse_probe_output(mono.stdout)
        implementation_comparison = {}
        for dist, (_, module) in PROJECTS.items():
            installed_file = installed.get(dist, {}).get('file')
            source_file = mono_paths.get(module)
            if not installed_file or not source_file:
                implementation_comparison[dist] = {'status': 'BLOCKED', 'reason': 'missing_import_path'}
                continue
            wheel_root, source_root = Path(installed_file).resolve().parent, Path(source_file).resolve().parent
            if not wheel_root.is_relative_to(target.resolve()) or not source_root.is_relative_to(repo):
                implementation_comparison[dist] = {'status': 'BLOCKED', 'reason': 'import_outside_declared_roots'}
                continue
            implementation_comparison[dist] = compare_package_sources(wheel_root, source_root)
        evidence['implementation_comparison'] = implementation_comparison
        mono_computation = run([sys.executable, '-c', combo_probe.replace(
            repr(str(target)), repr(str(repo / 'factor_preprocess')), 1)], cwd=tmp, env=mono_env)
        mono_receipt = parse_probe_output(mono_computation.stdout)
        evidence['effective_execution_parity'] = {
            'status': 'PASS' if mono_computation.returncode == 0 and mono_receipt == combo_receipt
            and mono_receipt.get('lag_guard_observed') is True else 'BLOCKED',
            'checkout': mono_receipt, 'wheel': combo_receipt,
        }

    all_built = all(v["exit_code"] == 0 for v in evidence["wheel_builds"].values())
    all_imported = all(v.get("status") == "IMPORTED" for v in evidence["combined_wheels"]["imports"].values())
    real_fe = evidence["factor_preprocess_with_real_fe"]["certification"] == "VERIFIED"
    mono_ok = evidence["monorepo"]["probe_exit_code"] == 0
    same_sources = all(row['status'] == 'PASS' for row in evidence['implementation_comparison'].values())
    same_execution = evidence['effective_execution_parity']['status'] == 'PASS'
    collected = evidence["required_parity_collection"]["status"] == "PASS"
    evidence["status"] = "PASS" if all_built and all_imported and real_fe and mono_ok and not duplicates and same_sources and same_execution and collected else "BLOCKED"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0 if evidence["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
