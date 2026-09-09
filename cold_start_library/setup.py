"""Build hook for the single authoritative cold-start catalog artifact."""

from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPyWithCatalog(build_py):
    def run(self) -> None:
        super().run()
        source = Path(__file__).parent / "library" / "production_default_core_v9.json"
        target = Path(self.build_lib) / "cold_start_library" / "library" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        self.copy_file(str(source), str(target))


setup(cmdclass={"build_py": BuildPyWithCatalog})
