# -*- coding: utf-8 -*-
"""R24-141..143: minute transform wall-clock parameters are strictly validated
(HH:MM syntax, start<end) — never accepted as opaque strings."""
from __future__ import annotations

import pytest

from factor_engine.api.source_ref import make_source_ref


def test_valid_minute_at_accepted() -> None:
    make_source_ref("StockMinuteBar", "close", transform="minute_at",
                    transform_params={"hhmm": "09:30"})


@pytest.mark.parametrize("bad", ["930", "25:00", "09:70", "9:3", "", "09-30"])
def test_invalid_hhmm_rejected(bad: str) -> None:
    with pytest.raises(ValueError, match="HH:MM"):
        make_source_ref("StockMinuteBar", "close", transform="minute_at",
                        transform_params={"hhmm": bad})


def test_valid_minute_range_accepted() -> None:
    make_source_ref("StockMinuteBar", "close", transform="minute_range",
                    transform_params={"start": "09:30", "end": "11:29"})


def test_minute_range_start_after_end_rejected() -> None:
    with pytest.raises(ValueError, match="earlier than end"):
        make_source_ref("StockMinuteBar", "close", transform="minute_range",
                        transform_params={"start": "11:30", "end": "09:30"})


def test_minute_range_bad_bound_rejected() -> None:
    with pytest.raises(ValueError, match="HH:MM"):
        make_source_ref("StockMinuteBar", "close", transform="minute_range",
                        transform_params={"start": "0930", "end": "11:30"})
