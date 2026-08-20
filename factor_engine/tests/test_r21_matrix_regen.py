# -*- coding: utf-8 -*-
"""R21-MATRIX-REGEN-V2: cell escaping + emitted-table well-formedness guards.

Covers review finding R21-MT-P2a (raw ``|`` inside parameter_domain_hash broke
the matrix table at line 3476) plus a general guard: every data row of every
emitted markdown table must have exactly the header's column count.
"""
from __future__ import annotations

import os

for _var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "POLARS_MAX_THREADS",
):
    os.environ[_var] = "1"

import sys
from pathlib import Path

FE_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(FE_ROOT.parent), str(FE_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.generate_physical_implementation_matrix import (  # noqa: E402
    _cell,
    write_matrix_doc,
)

MATRIX_DOC = FE_ROOT / "docs" / "PHYSICAL_IMPLEMENTATION_MATRIX.md"


def test_cell_escapes_pipe_and_newline() -> None:
    """A value containing ``|``/newline renders as one well-formed cell."""
    assert _cell("{a|b}") == r"{a\|b}"
    assert _cell("x\ny") == "x y"
    assert _cell("x\ry\tz") == "x y z"
    # unaffected pass-throughs
    assert _cell(True) == "yes"
    assert _cell(False) == "no"
    assert _cell(None) == "-"
    assert _cell("") == "-"
    assert _cell("plain_value") == "plain_value"


def _markdown_table_rows(text: str) -> list[tuple[int, list[str], list[str]]]:
    """Return (line_number, header_cells, data_rows) per markdown table.

    A table is a header row, a ``|---|`` separator, then data rows starting
    with ``|``.  Cell counting strips escaped ``\\|`` before splitting so an
    escaped pipe never inflates the column count.
    """

    def split_cells(line: str) -> list[str]:
        # protect escaped pipes, split on raw ones, then restore
        protected = line.replace("\\|", "\x00")
        cells = protected.split("|")
        return [c.replace("\x00", "\\|") for c in cells]

    tables: list[tuple[int, list[str], list[str]]] = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if (
            stripped.startswith("|")
            and i + 1 < len(lines)
            and set(lines[i + 1].strip()) <= {"|", "-", ":", " "}
            and "---" in lines[i + 1]
        ):
            header = split_cells(stripped)
            data: list[str] = []
            j = i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                data.append(lines[j].strip())
                j += 1
            tables.append((i + 1, header, data))
            i = j
        else:
            i += 1
    return tables


def test_emitted_matrix_tables_well_formed(tmp_path: Path, monkeypatch) -> None:
    """Every data row of every table has exactly the header's cell count.

    Includes a synthetic row whose parameter_domain_hash contains a raw ``|``
    to pin the R21-MT-P2a regression.
    """
    import scripts.generate_physical_implementation_matrix as gen

    rows = [
        {
            "canonical": "synthetic_pipe_op",
            "registry_slot": "primary",
            "physical_backend": "pandas_numpy",
            "implementation_id": "synthetic-impl-1",
            "execution_kind": "vectorized",
            "implemented": True,
            "selectable": True,
            "spec_complete": True,
            "oracle_passed": "NOT_RUN",
            "edge_passed": False,
            "parity_passed": True,
            "production_evidence_passed": False,
            "production_admitted": False,
            "source_hash": "abc123",
            "semantic_hash": "def456",
            "parameter_domain_hash": "{window|lambda|gap}",  # R21-MT-P2a shape
            "current_head_sha": "synthetic",
        },
        {
            "canonical": "synthetic_plain_op",
            "registry_slot": "primary",
            "physical_backend": "polars",
            "implementation_id": "",
            "execution_kind": "undeclared",
            "implemented": True,
            "selectable": False,
            "spec_complete": False,
            "oracle_passed": "NOT_RUN",
            "edge_passed": False,
            "parity_passed": False,
            "production_evidence_passed": False,
            "production_admitted": False,
            "source_hash": "",
            "semantic_hash": "",
            "parameter_domain_hash": "",
            "current_head_sha": "synthetic",
        },
    ]
    direct_rows = [
        {
            "canonical": "synthetic_pipe_op",
            "direct_use_status": "mining_visible",
            "mining_visible": True,
            "composition_usable": True,
            "terminal_usable": False,
            "production_admitted": False,
            "directly_usable": False,
        },
        {
            "canonical": "synthetic_plain_op",
            "direct_use_status": "not_mining_visible",
            "mining_visible": False,
            "composition_usable": False,
            "terminal_usable": False,
            "production_admitted": False,
            "directly_usable": False,
        },
    ]
    summary = {
        "total_rows": 2,
        "total_canonicals": 2,
        "backends": {},
        "direct_use": {
            "total_canonicals": 2,
            "mining_visible": 1,
            "composition_usable": 1,
            "terminal_usable": 0,
            "production_admitted": 0,
            "directly_usable": 0,
        },
    }
    target = tmp_path / "MATRIX.md"
    monkeypatch.setattr(gen, "MATRIX_DOC", target)
    write_matrix_doc(rows, direct_rows, summary, "synthetic")
    text = target.read_text(encoding="utf-8")

    tables = _markdown_table_rows(text)
    assert tables, "emitted doc must contain at least the two data tables"
    assert len(tables) == 2
    for line_no, header, data in tables:
        n_cols = len([c for c in header if c.strip()])
        for data_row in data:
            # count raw pipes after protecting escapes: must equal header pipes
            protected = data_row.replace("\\|", "\x00")
            raw_pipes = protected.count("|")
            header_protected = "|".join(header).replace("\\|", "\x00")
            header_pipes = header_protected.count("|")
            assert raw_pipes == header_pipes, (
                f"line {line_no}: row '{data_row}' has {raw_pipes} pipes "
                f"vs header {header_pipes}"
            )
    # the R21-MT-P2a regression specifically: escaped pipe present in output
    assert r"{window\|lambda\|gap}" in text


def test_generated_matrix_doc_on_disk_well_formed() -> None:
    """The real generated artifact on disk must have consistent column counts."""
    if not MATRIX_DOC.is_file():
        raise AssertionError(
            "docs/PHYSICAL_IMPLEMENTATION_MATRIX.md missing — run the generator"
        )
    text = MATRIX_DOC.read_text(encoding="utf-8")
    tables = _markdown_table_rows(text)
    assert tables, "generated matrix must contain tables"
    bad: list[str] = []
    for line_no, header, data in tables:
        header_pipes = "|".join(header).replace("\\|", "\x00").count("|")
        for data_row in data:
            protected = data_row.replace("\\|", "\x00")
            if protected.count("|") != header_pipes:
                bad.append(f"line {line_no + 1}: {data_row[:80]}")
    assert not bad, f"rows with pipe-count mismatch: {bad[:5]}"


def test_generated_matrix_doc_covers_new_event_state_canonicals() -> None:
    """The 16 new event/state canonicals (plus alias) appear in the matrix."""
    expected = [
        "event_streak",
        "event_cluster_duration",
        "event_decay_window",
        "signed_event_rate",
        "positive_event_rate",
        "negative_event_rate",
        "positive_event_age",
        "negative_event_age",
        "signed_event_decay",
        "event_direction_imbalance",
        "event_flip_density",
        "state_age",
        "state_persistence",
        "state_transition_count",
        "state_transition_rate",
        "state_flip_density",
    ]
    text = MATRIX_DOC.read_text(encoding="utf-8")
    missing = [c for c in expected if f"| {c} |" not in text]
    assert not missing, f"canonicals missing from generated matrix: {missing}"
    # state_episode_duration is an ALIAS of state_age (one canonical, alias-safe),
    # so it must NOT appear as its own canonical row; assert the live binding.
    assert not [
        ln for ln in text.split("\n") if ln.startswith("| state_episode_duration |")
    ], "state_episode_duration is an alias of state_age — must not be its own row"
    import scripts.generate_physical_implementation_matrix as gen

    gen._bootstrap()
    from cleaned_operators.registry import OperatorRegistry as _Reg

    assert _Reg._aliases.get("state_episode_duration") == "state_age"


def test_generated_matrix_doc_has_no_unescaped_pipe_in_hash_cells() -> None:
    """No raw (unescaped) pipe splits any row into more cells than its header.

    Splits each data row on raw pipes (escaped ``\\|`` protected) and requires
    the resulting cell count to match the header exactly — the direct
    statement of the R21-MT-P2a regression.
    """
    text = MATRIX_DOC.read_text(encoding="utf-8")
    for line_no, header, data in _markdown_table_rows(text):
        header_cells = len([c for c in header if c.strip()])
        for data_row in data:
            protected = data_row.replace("\\|", "\x00")
            cells = protected.split("|")
            assert len(cells) == header_cells + 2, (
                f"line {line_no}: row splits into {len(cells)} cells "
                f"vs header {header_cells + 2}"
            )
