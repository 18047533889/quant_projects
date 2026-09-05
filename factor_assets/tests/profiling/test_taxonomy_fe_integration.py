# -*- coding: utf-8 -*-
"""End-to-end taxonomy tests through the real FE static-analysis pipeline.

These exercise the deterministic classifier against the actual
``factor_engine.api.static_analysis.analyze_factor_definition`` output.  FE is
an optional dependency of factor_assets, so the module self-skips when it is
unavailable; in this repo the FE package is present and these run for real.
"""

import pytest

from factor_assets.profiling.taxonomy import classify_factor_taxonomy

factor_engine = pytest.importorskip(
    "factor_engine.api.static_analysis",
    reason="factor_engine optional dependency not installed",
)
analyze_factor_definition = factor_engine.analyze_factor_definition


def _mech(art):
    return tuple(t.tag for t in art.mechanism_tags)


class TestRealDslPipeline:
    def test_momentum(self):
        art = classify_factor_taxonomy(
            analyze_factor_definition("rank(ts_mean(close, 5) / ts_std(close, 20))")
        )
        assert "MOMENTUM" in _mech(art)
        assert art.data_domains == ("PRICE",)
        assert "RANKED" in art.structure_tags
        assert "TIME_SERIES" in art.structure_tags
        assert art.display_family == "PV_ONLY"

    def test_reversal(self):
        art = classify_factor_taxonomy(
            analyze_factor_definition("rank(-ts_mean(close, 5) / ts_std(close, 20))")
        )
        assert "REVERSAL" in _mech(art)

    def test_volatility(self):
        art = classify_factor_taxonomy(analyze_factor_definition("rank(ts_std(close, 20))"))
        assert "VOLATILITY" in _mech(art)

    def test_liquidity_field(self):
        art = classify_factor_taxonomy(analyze_factor_definition("rank(turnover_ratio)"))
        assert "LIQUIDITY" in _mech(art)
        assert "LIQUIDITY" in art.data_domains

    def test_price_volume_mixed(self):
        art = classify_factor_taxonomy(
            analyze_factor_definition("rank(ts_corr(close, volume, 10))")
        )
        assert "INTERACTION" in _mech(art)
        assert set(art.data_domains) == {"PRICE", "VOLUME"}
        assert "MIXED_DOMAIN" in art.structure_tags

    def test_fundamental_quality(self):
        art = classify_factor_taxonomy(analyze_factor_definition("rank(roe)"))
        assert "QUALITY" in _mech(art)
        assert art.data_domains == ("FUNDAMENTAL.QUALITY",)

    def test_fundamental_value(self):
        art = classify_factor_taxonomy(analyze_factor_definition("rank(pe_ratio)"))
        assert "VALUE" in _mech(art)

    def test_determinism_through_real_pipeline(self):
        dsl = "rank(ts_mean(close, 5) / ts_std(close, 20))"
        a1 = classify_factor_taxonomy(analyze_factor_definition(dsl))
        a2 = classify_factor_taxonomy(analyze_factor_definition(dsl))
        assert a1.content_hash == a2.content_hash
        assert a1.to_dict() == a2.to_dict()

    def test_structure_full_set_through_real_pipeline(self):
        art = classify_factor_taxonomy(
            analyze_factor_definition("rank(winsorize(zscore(ts_ema(close, 5))))")
        )
        assert art.structure_tags == (
            "TIME_SERIES", "CROSS_SECTIONAL", "RANKED", "ZSCORED",
            "WINSORIZED", "SMOOTHED",
        )
