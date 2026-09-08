"""Keep the explicit setuptools package manifest complete."""

from pathlib import Path
import tomllib


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_EXCLUDED_TOP_LEVEL = {"audit", "benchmarks", "build", "scripts", "tests"}


def test_every_source_package_is_declared() -> None:
    declared = set(
        tomllib.loads((_PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]
        ["setuptools"]["packages"]
    )
    discovered = {
        "factor_engine"
        if init.parent == _PACKAGE_ROOT
        else "factor_engine." + ".".join(init.parent.relative_to(_PACKAGE_ROOT).parts)
        for init in _PACKAGE_ROOT.rglob("__init__.py")
        if init.parent == _PACKAGE_ROOT
        or init.parent.relative_to(_PACKAGE_ROOT).parts[0] not in _EXCLUDED_TOP_LEVEL
    }

    assert discovered == declared, (
        f"undeclared source packages: {sorted(discovered - declared)}; "
        f"declared packages missing from source: {sorted(declared - discovered)}"
    )
