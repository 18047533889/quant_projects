"""Verify explicit setuptools package payload, bytes and runtime resources.

Usage: python scripts/verify_wheel_payload.py PACKAGE_ROOT WHEEL
Python 3.11+ build-time tooling; runtime minimum is not changed.
"""
from pathlib import Path
from zipfile import ZipFile
import fnmatch
import hashlib
import json
import sys
import tomllib


def verify(root: Path, wheel: Path) -> dict:
    config = tomllib.loads((root / "pyproject.toml").read_text())
    setup = config["tool"]["setuptools"]
    packages = setup["packages"]
    if not isinstance(packages, list) or not packages:
        raise ValueError("explicit nonempty package list required")
    mappings = setup["package-dir"]
    exclude = setup.get("exclude-package-data", {})
    expected = {}
    for package in packages:
        prefix = max((p for p in mappings if package == p or package.startswith(p + ".")), key=len)
        source = root / mappings[prefix] / package[len(prefix):].lstrip(".").replace(".", "/")
        if not source.is_dir():
            raise ValueError(f"declared package source missing: {package}")
        patterns = exclude.get("*", []) + exclude.get(package, [])
        files = list(source.glob("*.py"))
        for pattern in setup.get("package-data", {}).get(package, []):
            files.extend(source.glob(pattern))
        for file in files:
            if not file.is_file():
                continue
            relative = file.relative_to(source).as_posix()
            if any(fnmatch.fnmatch(relative, p) or fnmatch.fnmatch(file.name, p) for p in patterns):
                continue
            expected[package.replace(".", "/") + "/" + relative] = file
    if not any(n.endswith(".py") for n in expected):
        raise ValueError("vacuous source enumeration")
    roots = {p.split(".")[0] for p in packages}
    with ZipFile(wheel) as archive:
        names = archive.namelist()
        for required in ("METADATA", "RECORD"):
            if not any(n.endswith(".dist-info/" + required) for n in names):
                raise ValueError(f"missing wheel integrity metadata: {required}")
        if len(names) != len(set(names)):
            raise ValueError("duplicate archive entries")
        actual_py = {n for n in names if n.endswith(".py") and n.split("/")[0] in roots}
        expected_py = {n for n in expected if n.endswith(".py")}
        missing = set(expected) - set(names)
        extra_py = actual_py - expected_py
        mismatched = {n for n, p in expected.items() if n in names and archive.read(n) != p.read_bytes()}
        if missing or extra_py or mismatched:
            raise ValueError(json.dumps({"missing": sorted(missing), "extra_py": sorted(extra_py),
                                        "byte_mismatch": sorted(mismatched)}))
        if any("__pycache__/" in n or n.endswith(".pyc") for n in names):
            raise ValueError("bytecode shipped")
    return {"status": "PASS", "python_files": len(expected_py), "resources": len(expected)-len(expected_py),
            "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest()}


if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1]), Path(sys.argv[2])), sort_keys=True))
