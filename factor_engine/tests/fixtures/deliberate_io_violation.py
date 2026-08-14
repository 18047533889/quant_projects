#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Negative test fixture: deliberate I/O boundary violation.

This test file MUST be caught by the boundary checker when run with --strict
and placed outside the tests/ directory. It demonstrates that the checker
correctly detects unauthorized I/O operations.

This file should be moved to a temporary location for negative testing.
"""

import duckdb
import pandas as pd


def violate_boundary_with_duckdb():
    """Deliberate violation: direct duckdb.connect."""
    # This MUST be detected by check_dataaccess_io_boundary.py
    conn = duckdb.connect(":memory:")  # VIOLATION
    return conn


def violate_boundary_with_pandas():
    """Deliberate violation: direct pd.read_parquet."""
    # This MUST be detected by check_dataaccess_io_boundary.py
    df = pd.read_parquet("some_file.parquet")  # VIOLATION
    return df


def violate_boundary_with_write():
    """Deliberate violation: direct df.to_parquet."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    # This MUST be detected by check_dataaccess_io_boundary.py
    df.to_parquet("output.parquet")  # VIOLATION
