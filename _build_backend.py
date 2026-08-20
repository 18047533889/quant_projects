"""Custom build backend to exclude test files from wheel."""
import os
from setuptools import build_meta as _orig

# Re-export the standard setuptools backend
prepare_metadata_for_build_wheel = _orig.prepare_metadata_for_build_wheel
build_sdist = _orig.build_sdist
get_requires_for_build_wheel = _orig.get_requires_for_build_wheel
get_requires_for_build_sdist = _orig.get_requires_for_build_sdist

# Editable install support (PEP 660)
try:
    prepare_metadata_for_build_editable = _orig.prepare_metadata_for_build_editable
    get_requires_for_build_editable = _orig.get_requires_for_build_editable
    build_editable = _orig.build_editable
except AttributeError:
    # Fallback for older setuptools
    pass


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """Build wheel and remove test files."""
    import tempfile
    import shutil
    from pathlib import Path

    # Build to temp location
    with tempfile.TemporaryDirectory() as tmpdir:
        wheel_path = _orig.build_wheel(tmpdir, config_settings, metadata_directory)
        wheel_file = Path(tmpdir) / wheel_path

        # Unpack, filter, repack
        import zipfile
        final_path = Path(wheel_directory) / wheel_path

        with zipfile.ZipFile(wheel_file, 'r') as zin:
            with zipfile.ZipFile(final_path, 'w', zipfile.ZIP_DEFLATED) as zout:
                for item in zin.infolist():
                    # Skip test files and __pycache__
                    if (item.filename.startswith('test_') or
                        '/test_' in item.filename or
                        '_test.py' in item.filename or
                        '__pycache__' in item.filename or
                        item.filename.endswith('.pyc')):
                        continue

                    data = zin.read(item.filename)
                    zout.writestr(item, data)

        return wheel_path
