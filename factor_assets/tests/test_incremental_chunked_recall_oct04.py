"""Behavioral tests for bounded exact scans of oversized member groups."""

from decimal import Decimal, localcontext
from types import SimpleNamespace

import numpy as np
import pytest

from factor_assets.clustering.incremental_recall import (
    scan_exact_member_winner, unit_cosine_scores,
)
from factor_assets.similarity.unit_vectors import _unit_rows


def _member(fid, values):
    return SimpleNamespace(factor_id=fid, embedding=values)



def _scan(mapping, member_ids, query, **kwargs):
    return scan_exact_member_winner(
        mapping, member_ids, np.asarray(query, dtype=np.float64),
        to_embedding=lambda value: np.asarray(value.embedding, dtype=np.float64),
        normalize_rows=_unit_rows,
        **kwargs,
    )


def test_chunk_scan_returns_exact_winner_and_first_member_across_chunk_tie():
    mapping = {
        "first": _member("first", [1.0, 0.0]),
        "middle": _member("middle", [0.0, 1.0]),
        "tied-later": _member("tied-later", [1.0, 0.0]),
        "negative": _member("negative", [-1.0, 0.0]),
    }
    assert _scan(mapping, list(mapping), [1.0, 0.0], max_chunk_rows=2) == (
        "first", 1.0)


def test_chunk_scan_does_not_treat_adjacent_float_scores_as_ties():
    low = 0.5
    high = np.nextafter(low, 1.0)
    mapping = {
        "lower-first": _member("lower-first", [low, np.sqrt(1.0 - low * low)]),
        "higher-second": _member("higher-second", [high, np.sqrt(1.0 - high * high)]),
    }
    assert _scan(mapping, list(mapping), [1.0, 0.0], max_chunk_rows=1) == (
        "higher-second", high)


def test_chunk_scan_checks_nonwinner_rows_in_later_chunks():
    mapping = {
        "winner": _member("winner", [1.0, 0.0]),
        "invalid-later": _member("invalid-later", [2.0, 0.0]),
    }
    with pytest.raises(ValueError, match="unit cosine scores"):
        scan_exact_member_winner(
            mapping, list(mapping), np.array([1.0, 0.0]),
            to_embedding=lambda value: np.asarray(value.embedding, dtype=np.float64),
            normalize_rows=lambda matrix: np.asarray(matrix, dtype=np.float64),
            max_chunk_rows=1,
        )


def test_chunk_scan_skips_missing_members_and_rejects_miskeyed_fingerprints():
    mapping = {
        "winner": _member("winner", [1.0, 0.0]),
        "orthogonal": _member("orthogonal", [0.0, 1.0]),
    }
    assert _scan(mapping, ["missing", "winner", "orthogonal"], [1.0, 0.0],
                 max_chunk_rows=1) == ("winner", 1.0)
    with pytest.raises(ValueError, match="mapping key"):
        _scan({"alias": _member("actual", [1.0, 0.0])}, ["alias"], [1.0, 0.0])


def test_chunk_scan_empty_members_returns_none():
    assert _scan({}, [], [1.0, 0.0]) is None


def test_chunk_scan_caps_row_count_and_matrix_payload():
    width = 8
    ids = [f"m-{i}" for i in range(11)]
    mapping = {
        fid: _member(fid, np.arange(1, width + 1, dtype=np.float64))
        for fid in ids
    }
    observed = []

    def normalize(matrix):
        observed.append((matrix.shape[0], matrix.nbytes))
        return _unit_rows(matrix)

    result = scan_exact_member_winner(
        mapping, ids, np.ones(width, dtype=np.float64) / np.sqrt(width),
        to_embedding=lambda value: np.asarray(value.embedding, dtype=np.float64),
        normalize_rows=normalize,
        max_chunk_rows=5,
        max_chunk_bytes=width * 8 * 3 * 2,
    )
    assert result is not None
    assert len(observed) == 6
    assert max(rows for rows, _ in observed) <= 5
    assert observed == [(2, width * 8 * 2)] * 5 + [(1, width * 8)]
    assert sum(rows for rows, _ in observed) == len(ids)


def test_chunk_scan_rejects_invalid_query_before_scoring():
    with pytest.raises(ValueError, match="query"):
        _scan({"m": _member("m", [1.0, 0.0])}, ["m"], [[1.0, 0.0]])


def test_chunk_scan_matches_full_matrix_argmax_on_seeded_inputs():
    rng = np.random.default_rng(20261004)
    ids = [f"seeded-{index}" for index in range(53)]
    mapping = {
        fid: _member(fid, rng.normal(size=17))
        for fid in ids
    }
    query = rng.normal(size=17)
    query /= np.linalg.norm(query)
    full_matrix = _unit_rows(np.vstack([
        np.asarray(mapping[fid].embedding, dtype=np.float64) for fid in ids
    ]))
    full_scores = unit_cosine_scores(query, full_matrix)
    expected_index = int(np.argmax(full_scores))
    expected = (ids[expected_index], float(full_scores[expected_index]))

    for chunk_rows in (1, 7, 16, 128):
        actual = _scan(mapping, ids, query, max_chunk_rows=chunk_rows)
        assert actual[0] == expected[0]
        assert actual[1] == pytest.approx(expected[1], rel=0.0, abs=5e-16)


@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_chunk_scan_rejects_nonfinite_nonwinner_in_later_chunk(invalid):
    mapping = {
        "winner": _member("winner", [1.0, 0.0]),
        "invalid-later": _member("invalid-later", [invalid, 0.0]),
    }
    with pytest.raises(ValueError, match="only finite values"):
        _scan(mapping, list(mapping), [1.0, 0.0], max_chunk_rows=1)


def test_chunk_scan_rejects_member_width_mismatch():
    mapping = {"bad-width": _member("bad-width", [1.0, 0.0, 0.0])}
    with pytest.raises(ValueError, match="matching the query width"):
        _scan(mapping, list(mapping), [1.0, 0.0], max_chunk_rows=1)


def test_chunk_scan_enforces_row_count_when_byte_budget_allows_more():
    ids = [f"wide-{i}" for i in range(8)]
    mapping = {
        fid: _member(fid, [1.0, 0.0])
        for fid in ids
    }
    observed = []

    def normalize(matrix):
        observed.append(matrix.shape[0])
        return _unit_rows(matrix)

    result = scan_exact_member_winner(
        mapping, ids, np.array([1.0, 0.0]),
        to_embedding=lambda value: np.asarray(value.embedding, dtype=np.float64),
        normalize_rows=normalize,
        max_chunk_rows=3,
        max_chunk_bytes=4096,
    )
    assert result == (ids[0], 1.0)
    assert observed == [3, 3, 2]


def test_near_equal_chunk_winner_and_floor_match_decimal_oracle():
    low = 0.5
    high = np.nextafter(low, 1.0)
    mapping = {
        "lower-first": _member(
            "lower-first", [low, np.sqrt(1.0 - low * low)]),
        "higher-second": _member(
            "higher-second", [high, np.sqrt(1.0 - high * high)]),
    }
    with localcontext() as context:
        context.prec = 120
        decimal_scores = {}
        for fid in mapping:
            x, y = (Decimal.from_float(float(value))
                    for value in mapping[fid].embedding)
            decimal_scores[fid] = x / (x * x + y * y).sqrt()

    assert decimal_scores["higher-second"] > decimal_scores["lower-first"]
    oracle_score = float(decimal_scores["higher-second"])
    for chunk_rows in (1, 2):
        result = _scan(mapping, list(mapping), [1.0, 0.0],
                       max_chunk_rows=chunk_rows)
        assert result[0] == "higher-second"
        assert result[1] == pytest.approx(oracle_score, rel=0.0, abs=5e-16)
        lower_floor = oracle_score - 4e-16
        upper_floor = oracle_score + 4e-16
        assert (result[1] < lower_floor) is False
        assert (result[1] < upper_floor) is True


def test_chunk_scan_byte_budget_boundary_limits_to_one_row():
    ids = [f"boundary-{i}" for i in range(3)]
    mapping = {fid: _member(fid, [1.0, 0.0]) for fid in ids}
    observed = []

    def normalize(matrix):
        observed.append(matrix.shape[0])
        return _unit_rows(matrix)

    scan_exact_member_winner(
        mapping, ids, np.array([1.0, 0.0]),
        to_embedding=lambda value: np.asarray(value.embedding, dtype=np.float64),
        normalize_rows=normalize,
        max_chunk_rows=3,
        max_chunk_bytes=3 * 2 * 8 * 2 - 1,
    )
    assert observed == [1, 1, 1]


def test_equal_member_tie_keeps_first_when_chunk_shape_changes_blas_rounding():
    rng = np.random.default_rng(4)
    for _ in range(3):
        vector = rng.normal(size=17)
        query = _unit_rows(rng.normal(size=(1, 17)))[0]
    mapping = {
        "first": _member("first", vector),
        "middle": _member("middle", -query),
        "last": _member("last", vector),
    }
    with localcontext() as context:
        context.prec = 120
        vector_dec = [Decimal.from_float(float(value)) for value in vector]
        query_dec = [Decimal.from_float(float(value)) for value in query]
        dot = sum(x * y for x, y in zip(vector_dec, query_dec))
        norm_vector = sum(x * x for x in vector_dec).sqrt()
        norm_query = sum(y * y for y in query_dec).sqrt()
        oracle = dot / (norm_vector * norm_query)
    result = _scan(mapping, list(mapping), query, max_chunk_rows=2)
    assert result[0] == "first"
    assert result[1] == pytest.approx(float(oracle), rel=0.0, abs=5e-16)
    assert (result[1] < float(oracle) - 4e-16) is False
    assert (result[1] < float(oracle) + 4e-16) is True
