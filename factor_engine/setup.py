"""Apply the declared package exclusions to Python modules as well as data.

Setuptools exclude-package-data does not filter build_py's module list. Without
this narrow hook, co-located test_*.py files enter independent runtime wheels
despite the existing explicit exclusion contract in pyproject.toml.
"""
from fnmatch import fnmatch
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class DeclaredPayloadBuildPy(build_py):
    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        exclusions = self.distribution.exclude_package_data or {}
        # Setuptools normalizes the TOML wildcard key to the empty string.
        patterns = (list(exclusions.get("", ())) + list(exclusions.get("*", ()))
                    + list(exclusions.get(package, ())))
        return [entry for entry in modules
                if not any(fnmatch(Path(entry[2]).name, pattern) for pattern in patterns)]


if __name__ == "__main__":
    setup(cmdclass={"build_py": DeclaredPayloadBuildPy})
