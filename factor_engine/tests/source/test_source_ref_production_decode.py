# -*- coding: utf-8 -*-
"""R24-139/140: production SourceRef decode is STRICT — unknown transform,
unapproved dialect, or a non-SourceRef string all hard-fail, never fall back."""
from __future__ import annotations

import pytest

from api.source_ref import (
    decode_source_ref_production,
    encode_source_ref,
    make_source_ref,
)


def test_non_sourceref_rejected_in_production() -> None:
    with pytest.raises(ValueError, match="not a SourceRef"):
        decode_source_ref_production("close")


def test_old_unapproved_dialect_rejected() -> None:
    spec = make_source_ref("StockDailyBar", "close", dialect_version="1999-01-01")
    with pytest.raises(ValueError, match="dialect_version"):
        decode_source_ref_production(encode_source_ref(spec))


def test_approved_dialect_accepted() -> None:
    spec = make_source_ref("StockDailyBar", "close", dialect_version="2026-08-10")
    d = decode_source_ref_production(encode_source_ref(spec))
    assert d.table == "StockDailyBar"


def test_unknown_transform_rejected_in_production() -> None:
    # The transform contract doesn't know "bogus_transform" → production rejects.
    spec = make_source_ref("StockDailyBar", "close", transform="bogus_transform")
    with pytest.raises(ValueError, match="not a declared production transform"):
        decode_source_ref_production(encode_source_ref(spec))
