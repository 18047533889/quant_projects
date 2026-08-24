# -*- coding: utf-8 -*-
"""R24-119/120 + R24-245: NextTradingOpen.resolve() must NOT substitute the
decision time for an unknown publication reference.  A row with no PubDate must
fail closed — never resolve to the next-session open of the decision date."""
from __future__ import annotations

import pandas as pd
import pytest

from factor_engine.ir.types import NextTradingOpen


class _FakeCalendar:
    def next_open(self, date):
        return pd.Timestamp("2026-08-11 09:30:00")


def test_next_trading_open_with_known_pubdate() -> None:
    row = pd.Series({"PubDate": pd.Timestamp("2026-08-10")})
    out = NextTradingOpen().resolve(row, calendar=_FakeCalendar())
    assert out == pd.Timestamp("2026-08-11 09:30:00")


def test_unknown_pubdate_does_not_use_decision_time() -> None:
    # R24-245 golden: row without PubDate + decision_context=2026-08-10 must
    # fail, NEVER resolve to 2026-08-11 open.
    from types import SimpleNamespace

    row = pd.Series({})
    decision_context = SimpleNamespace(asof=pd.Timestamp("2026-08-10"))
    with pytest.raises(NotImplementedError, match="decision time"):
        NextTradingOpen().resolve(row, calendar=_FakeCalendar(), decision_context=decision_context)
