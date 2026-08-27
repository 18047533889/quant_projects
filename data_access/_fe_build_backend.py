"""P0-04 private PEP 517 backend for DataAccess.

Wraps setuptools' backend and regenerates ``data_access/_build_info.py`` from
the live Git tree immediately before each build, so an installed wheel always
carries build facts matching the exact revision it was built from.
"""
from __future__ import annotations

import os
import subprocess
import sys

from setuptools import build_meta as _orig

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REGEN_SCRIPT = os.path.join(_REPO_ROOT, "scripts", "regenerate_build_info.py")


def _regenerate_build_info() -> None:
    """Run the regeneration script if it exists; otherwise keep the fallback."""
    if not os.path.exists(_REGEN_SCRIPT):
        return
    subprocess.run(
        [sys.executable, _REGEN_SCRIPT],
        cwd=_REPO_ROOT,
        check=True,
        env={**os.environ, "QUANT_BUILD_FORCE": "1"},
    )


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _regenerate_build_info()
    return _orig.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory, config_settings=None):
    _regenerate_build_info()
    return _orig.build_sdist(sdist_directory, config_settings)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    _regenerate_build_info()
    return _orig.prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    _regenerate_build_info()
    return _orig.prepare_metadata_for_build_editable(metadata_directory, config_settings)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    _regenerate_build_info()
    return _orig.build_editable(wheel_directory, config_settings, metadata_directory)


def get_requires_for_build_wheel(config_settings=None):
    return _orig.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_sdist(config_settings=None):
    return _orig.get_requires_for_build_sdist(config_settings)


def get_requires_for_build_editable(config_settings=None):
    return _orig.get_requires_for_build_editable(config_settings)