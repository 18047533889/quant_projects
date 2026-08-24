from __future__ import annotations

from types import SimpleNamespace

import factor_engine.cleaned_operators.model_lane as model_lane


def _authority(canonical, classification, *, path):
    return SimpleNamespace(
        __file__=path,
        classification_of=lambda _: classification,
        LEGACY_LOCAL_PREDICTIVE_CANONICALS={canonical: classification},
    )


def test_missing_or_ambiguous_legacy_authority_fails_closed(monkeypatch, tmp_path):
    canonical = "panel_rolling_pcr_forecast"
    monkeypatch.setattr(
        model_lane.importlib,
        "import_module",
        lambda _: _authority(canonical, object(), path=tmp_path / "legacy.py"),
    )

    assert model_lane.assign_model_lane(canonical) is None


def test_legacy_authority_remains_the_lane_source(monkeypatch):
    canonical = "panel_rolling_pcr_forecast"
    classification = object()
    expected_path = (
        model_lane.Path(model_lane.__file__).resolve().parent.parent
        / "modeling"
        / "legacy.py"
    )
    monkeypatch.setattr(
        model_lane.importlib,
        "import_module",
        lambda _: _authority(canonical, classification, path=expected_path),
    )

    assert model_lane.assign_model_lane(canonical) == "LEGACY_LOCAL_PREDICTIVE"


def test_legacy_authority_failure_does_not_fallback(monkeypatch):
    monkeypatch.setattr(
        model_lane.importlib,
        "import_module",
        lambda _: (_ for _ in ()).throw(RuntimeError("ambiguous authority")),
    )

    assert model_lane.assign_model_lane("panel_rolling_pcr_forecast") is None
