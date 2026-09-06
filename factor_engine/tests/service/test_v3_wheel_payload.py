from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts.verify_wheel_payload import verify


@pytest.mark.parametrize("mutation", [None, "missing", "extra", "wrong_bytes", "resource", "duplicate"])
def test_wheel_exact_payload(tmp_path, mutation):
    (tmp_path / "pyproject.toml").write_text('''[tool.setuptools]
packages = ["example"]
[tool.setuptools.package-dir]
example = "."
[tool.setuptools.package-data]
example = ["profile.yaml"]
''')
    (tmp_path / "__init__.py").write_text("VALUE = 1\n")
    (tmp_path / "profile.yaml").write_text("limit: 2\n")
    wheel = tmp_path / "example.whl"
    with ZipFile(wheel, "w") as archive:
        archive.writestr("example-1.dist-info/METADATA", "Name: example\n")
        archive.writestr("example-1.dist-info/RECORD", "")
        if mutation != "missing":
            archive.writestr("example/__init__.py", "VALUE = 0\n" if mutation == "wrong_bytes" else "VALUE = 1\n")
        if mutation != "resource":
            archive.writestr("example/profile.yaml", "limit: 2\n")
        if mutation == "extra":
            archive.writestr("example/stale.py", "")
        if mutation == "duplicate":
            archive.writestr("example/__init__.py", "VALUE = 1\n")
    if mutation is None:
        assert verify(tmp_path, wheel)["python_files"] == 1
    else:
        with pytest.raises(ValueError):
            verify(tmp_path, wheel)
