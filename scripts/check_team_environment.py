"""Read-only checks for the locked core environment and published source trees."""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import platform
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
CORE = ("data_access", "factor_engine", "quant_evaluator", "factor_preprocess",
        "factor_optimizer", "factor_assets", "quant_platform", "modeling")
MIRRORS = set(CORE) | {"vectorbt_qs", "riskfolio_qs", "alphaprobe", "platform_web", "lightgbm_qs"}
SHA = re.compile(r"[0-9a-f]{40}")


def git(root: Path, *args: str) -> str:
    p = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if p.returncode:
        raise ValueError(f"Git check failed: {args[0]}")
    return p.stdout.strip()


def read_pins(path: Path) -> dict[str, str]:
    pins = {}
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+)", line)
        if not match:
            raise ValueError(f"Non-exact constraint in {path.name}: {line}")
        name, version = match.groups()
        name = re.sub(r"[-_.]+", "-", name).lower()
        if name in pins:
            raise ValueError(f"Duplicate constraint: {name}")
        pins[name] = version
    if not pins:
        raise ValueError("Empty external constraint file")
    return pins


def editable_matches(raw: str | None, expected: Path) -> bool:
    try:
        d = json.loads(raw or "null")
        if not isinstance(d, dict) or d.get("dir_info", {}).get("editable") is not True:
            return False
        u = urlparse(d.get("url", ""))
        return (u.scheme == "file" and u.netloc in ("", "localhost")
                and Path(unquote(u.path)).resolve() == expected.resolve())
    except (ValueError, TypeError, AttributeError):
        return False


def receipt_errors(root: Path, receipt: dict) -> list[str]:
    errors = []
    if not isinstance(receipt, dict):
        return ["Receipt must be an object"]
    if receipt.get("success") is not True or receipt.get("dryRun") is not False:
        errors.append("Receipt must describe a successful real publication, not a dry-run")
    if not SHA.fullmatch(str(receipt.get("rootSHA", ""))):
        errors.append("Receipt has no valid source commit")
    entries = receipt.get("mirrors")
    if not isinstance(entries, list) or any(not isinstance(e, dict) for e in entries):
        return errors + ["Receipt mirror list is invalid"]
    names = [e.get("name") for e in entries]
    if any(not isinstance(n, str) for n in names) or len(names) != 13 or set(names) != MIRRORS:
        return errors + ["Receipt must contain each of the 13 mirrors exactly once"]
    for entry in entries:
        name = entry["name"]
        source = entry.get("sourceTree")
        if entry.get("status") not in ("pushed", "unchanged"):
            errors.append(f"{name}: publication was not completed")
        if not SHA.fullmatch(str(entry.get("mirrorCommit", ""))):
            errors.append(f"{name}: missing mirror commit")
        if not SHA.fullmatch(str(source)) or source != entry.get("verifiedRemoteTree"):
            errors.append(f"{name}: source and verified mirror trees do not match")
        try:
            if git(root, "rev-parse", f"HEAD:{name}") != source:
                errors.append(f"{name}: current committed source differs from published tree")
        except ValueError as exc:
            errors.append(f"{name}: {exc}")
    return errors


def isolated_import_errors(root: Path) -> list[str]:
    # Repo cwd/PYTHONPATH can hide broken editable installs or stale wheel caches.
    probe = """import importlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); failures=[]
for name in sys.argv[2:]:
    try:
        path=getattr(importlib.import_module(name),'__file__',None)
        if path is None or not pathlib.Path(path).resolve().is_relative_to(root/name):
            failures.append(name)
    except Exception:
        failures.append(name)
print(json.dumps(failures))
"""
    result = subprocess.run([sys.executable, "-I", "-c", probe, str(root), *CORE],
                            cwd=root.anchor, capture_output=True, text=True)
    if result.returncode:
        return ["Isolated import probe failed"]
    try:
        failed = json.loads(result.stdout)
        if not isinstance(failed, list) or any(n not in CORE for n in failed):
            raise ValueError("invalid import report")
    except (ValueError, TypeError):
        return ["Isolated import probe returned invalid evidence"]
    return [f"{name}: isolated import did not resolve to this checkout's source" for name in failed]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-only", action="store_true",
                        help="Check clean source and publication receipt only; does not certify dependencies")
    args = parser.parse_args(argv)
    errors = []
    try:
        if git(ROOT, "status", "--porcelain", "--untracked-files=normal"):
            errors.append("Source checkout is dirty; commit identity cannot certify uncommitted files")
        receipt = json.loads((ROOT / "evidence/team_sync_20260909/push_both_result.json").read_text())
        errors.extend(receipt_errors(ROOT, receipt))
    except (OSError, ValueError) as exc:
        errors.append(f"Source evidence check failed: {exc}")

    if not args.source_only:
        if sys.version_info[:3] != (3, 12, 3) or platform.system() != "Linux" or platform.machine() != "x86_64":
            errors.append("Core lock target is CPython 3.12.3 / Linux x86_64; other platforms require separate validation")
        if platform.python_implementation() != "CPython":
            errors.append("Core profile requires CPython")
        try:
            expected = read_pins(ROOT / "requirements/core-py312-linux.txt")
            for directory in CORE:
                project = tomllib.loads((ROOT / directory / "pyproject.toml").read_text())["project"]
                name = project["name"]
                expected[name] = project["version"]
                try:
                    dist = md.distribution(name)
                    if not editable_matches(dist.read_text("direct_url.json"), ROOT / directory):
                        errors.append(f"{name}: not installed editable from this checkout's {directory}")
                except md.PackageNotFoundError:
                    errors.append(f"Missing internal source install: {name}")
            for name, wanted in expected.items():
                try:
                    found = md.version(name)
                except md.PackageNotFoundError:
                    errors.append(f"Missing package: {name}=={wanted}")
                else:
                    if found != wanted:
                        errors.append(f"{name}: expected {wanted}, found {found}")
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"Dependency manifest check failed: {exc}")
        check = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
        if check.returncode:
            errors.append("pip check failed (run it directly for package conflicts)")
        errors.extend(isolated_import_errors(ROOT))

    scope = "SOURCE ONLY" if args.source_only else "CORE ENVIRONMENT AND SOURCE"
    print(f"{scope}: {'FAILED' if errors else 'PASSED'}")
    for error in errors:
        print(f"- {error}")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
