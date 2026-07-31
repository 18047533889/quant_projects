# -*- coding: utf-8 -*-
"""Install the session-clock v2 intraday feature runtime after legacy bootstrap."""
from __future__ import annotations


def install_intraday_clock_runtime() -> None:
    from .lqtp_logical_source_v2 import LQTPLogicalDataSource
    if getattr(LQTPLogicalDataSource,"_intraday_clock_v2_installed",False):
        return
    from .intraday_feature_runtime_v2 import load_intraday_feature
    previous=LQTPLogicalDataSource._minute_daily
    def patched(self,field,transform,params):
        if transform=="intraday_feature":
            return load_intraday_feature(self,dict(params))
        return previous(self,field,transform,params)
    LQTPLogicalDataSource._minute_daily=patched
    LQTPLogicalDataSource._intraday_clock_v2_installed=True
