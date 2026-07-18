from __future__ import annotations

import subprocess
import sys


def test_data_access_and_dataaccess_share_one_module_object():
    import data_access
    import dataaccess

    assert dataaccess is data_access
    assert dataaccess.get_store is data_access.get_store


def test_dataaccess_can_be_imported_first_without_circular_failure():
    code = (
        "import dataaccess, data_access; "
        "assert dataaccess is data_access; "
        "assert dataaccess.get_store is data_access.get_store"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
