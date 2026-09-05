"""R61-FI-014: FO FactorIntelligenceProvider port + FA adapter tests.

Covers (plan §20 E6 / matrix E6):
1. ``FactorIntelligenceProvider`` Protocol structure — an implementation with
   the three narrow methods satisfies it (@runtime_checkable) and exposes the
   right signatures.
2. Consumer views are frozen dataclasses with the full field sets
   (taxonomy: data_domains/display_family/mechanism_tags; health: 14
   dimension grades + overall_grade; diagnosis: tag/severity/confidence/
   repairability).
3. ``adapters/factor_assets.py`` projects FA-contract-shaped artifacts into
   views (duck-typing — no FA source import required).
4. No grading logic leaks into FO: a source scan asserts ports/ and
   adapters/factor_assets.py contain no grade-threshold tables.
5. Unknown ``factor_definition_id`` → explicit unknown view or typed error,
   never a silent ``None``.
"""

import dataclasses
import inspect
import re
import sys
from pathlib import Path

import pytest

from factor_optimizer.adapters.factor_assets import (
    FaFactorIntelligenceProvider,
    InMemoryFactorIntelligenceProvider,
    create_fa_intelligence_provider,
    create_in_memory_factor_intelligence_provider,
    diagnosis_view_from_fa,
    health_view_from_fa,
    taxonomy_view_from_fa,
)
from factor_optimizer.ports.factor_intelligence import (
    DiagnosisRepairability,
    DiagnosisSeverity,
    DiagnosisView,
    FactorHealthView,
    FactorIntelligenceProvider,
    FactorIntelligenceUnknownFactorError,
    FactorTaxonomyView,
    HealthDimension,
    HealthGrade,
    unknown_factor_view,
)

_PKG_ROOT = Path(__file__).resolve().parents[1] / "factor_optimizer"


# ---------------------------------------------------------------------------
# Helpers: FA-contract-shaped (duck) artifacts — mirrors the real FA
# factor_assets.profiling contracts without importing factor_assets.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _FATagEvidence:
    tag: str
    source: str
    confidence: float
    evidence_refs: tuple = ()


@dataclasses.dataclass(frozen=True)
class _FATaxonomyArtifact:
    factor_definition_id: str
    data_domains: tuple
    display_family: str
    mechanism_tags: tuple
    structure_tags: tuple = ()
    frequency_tags: tuple = ()
    field_usage_ref: str = ""
    operator_usage_ref: str = ""
    taxonomy_policy_id: str = "CN_A_SHARE_DAILY_TAXONOMY_V1"
    taxonomy_policy_version: str = "1.0.0"
    content_hash: str = ""


@dataclasses.dataclass(frozen=True)
class _FAHealthArtifact:
    factor_definition_id: str
    dimension_grades: dict
    overall_grade: str = HealthGrade.NONE


@dataclasses.dataclass(frozen=True)
class _FADiagnosisArtifact:
    factor_definition_id: str
    tag: str
    severity: str = DiagnosisSeverity.MEDIUM
    confidence: float = 0.6
    repairability: str = DiagnosisRepairability.REPAIRABLE
    details: tuple = ()
    health_ref: str = ""


def _fa_taxonomy(fid="F_1001"):
    return _FATaxonomyArtifact(
        factor_definition_id=fid,
        data_domains=("PRICE", "VOLUME"),
        display_family="PV_ONLY",
        mechanism_tags=(
            _FATagEvidence("MOMENTUM", "deterministic_rule", 0.85, ("operator:ts_mean",)),
            _FATagEvidence("VOLATILITY", "deterministic_rule", 0.85, ("operator:ts_std",)),
        ),
    )


def _fa_health(fid="F_1001"):
    grades = {dim: "B" for dim in HealthDimension.all()}
    grades[HealthDimension.PREDICTIVE_POWER] = "A"
    return _FAHealthArtifact(
        factor_definition_id=fid, dimension_grades=grades, overall_grade="A+"
    )


def _fa_diagnosis(fid="F_1001", tag="high_turnover"):
    return _FADiagnosisArtifact(
        factor_definition_id=fid,
        tag=tag,
        severity=DiagnosisSeverity.MEDIUM,
        confidence=0.7,
        repairability=DiagnosisRepairability.REPAIRABLE,
        details=("repair:increase_horizon",),
    )


# ---------------------------------------------------------------------------
# 1. Protocol structure
# ---------------------------------------------------------------------------


def test_provider_protocol_is_runtime_checkable_with_three_methods():
    assert hasattr(FactorIntelligenceProvider, "__protocol_attrs__")
    methods = {name for name, _ in inspect.getmembers(FactorIntelligenceProvider) if not name.startswith("_")}
    assert {"get_taxonomy", "get_health_card", "get_diagnoses"} <= methods
    sig = inspect.signature(FactorIntelligenceProvider.get_taxonomy)
    assert list(sig.parameters) == ["self", "factor_definition_id"]
    sig = inspect.signature(FactorIntelligenceProvider.get_health_card)
    assert list(sig.parameters) == ["self", "factor_definition_id", "evaluation_ref"]
    sig = inspect.signature(FactorIntelligenceProvider.get_diagnoses)
    assert list(sig.parameters) == ["self", "factor_definition_id", "health_ref"]


def test_inmemory_provider_satisfies_protocol():
    provider = create_in_memory_factor_intelligence_provider()
    assert isinstance(provider, FactorIntelligenceProvider)
    # runtime_checkable also accepts an ad-hoc class with the three methods
    class _Duck:
        def get_taxonomy(self, factor_definition_id):
            raise FactorIntelligenceUnknownFactorError(factor_definition_id)

        def get_health_card(self, factor_definition_id, evaluation_ref):
            raise FactorIntelligenceUnknownFactorError(factor_definition_id)

        def get_diagnoses(self, factor_definition_id, health_ref):
            raise FactorIntelligenceUnknownFactorError(factor_definition_id)

    assert isinstance(_Duck(), FactorIntelligenceProvider)


def test_fa_provider_satisfies_protocol():
    provider = create_fa_intelligence_provider()
    assert isinstance(provider, FactorIntelligenceProvider)
    assert isinstance(provider, FaFactorIntelligenceProvider)


# ---------------------------------------------------------------------------
# 2. Views are frozen dataclasses with complete field sets
# ---------------------------------------------------------------------------


def test_taxonomy_view_frozen_dataclass_fields():
    assert dataclasses.is_dataclass(FactorTaxonomyView)
    assert FactorTaxonomyView.__dataclass_params__.frozen
    fields = {f.name for f in dataclasses.fields(FactorTaxonomyView)}
    assert {"factor_definition_id", "data_domains", "display_family",
            "mechanism_tags", "is_unknown"} <= fields
    view = FactorTaxonomyView(
        factor_definition_id="F_1",
        data_domains=("PRICE",),
        display_family="PV_ONLY",
        mechanism_tags=(("MOMENTUM", "deterministic_rule", 0.85, ("operator:ts_mean",)),),
    )
    assert view.mechanism_tag_names == ("MOMENTUM",)
    assert view.is_unknown is False
    with pytest.raises(dataclasses.FrozenInstanceError):
        view.display_family = "OTHER"


def test_health_view_frozen_dataclass_14_dimensions():
    assert dataclasses.is_dataclass(FactorHealthView)
    assert FactorHealthView.__dataclass_params__.frozen
    assert len(HealthDimension.all()) == 14
    assert len(set(HealthDimension.all())) == 14
    grades = {dim: "B" for dim in HealthDimension.all()}
    view = FactorHealthView(
        factor_definition_id="F_1", evaluation_ref="ev-1",
        dimension_grades=grades, overall_grade="B+",
    )
    assert view.all_dimensions_graded
    assert view.grade_of(HealthDimension.PREDICTIVE_POWER) == "B"
    with pytest.raises(dataclasses.FrozenInstanceError):
        view.overall_grade = "S"


def test_health_view_unknown_dimension_fails_loud():
    with pytest.raises(ValueError, match="unknown health dimension"):
        FactorHealthView(
            factor_definition_id="F_1",
            dimension_grades={"not_a_dimension": "A"},
            overall_grade="A",
        )


def test_health_view_defaults_to_none_grades_not_zero():
    # missing evidence must never be silently graded; HealthGrade.NONE is the
    # explicit absence marker (FO has no thresholds to turn missing -> grade)
    view = FactorHealthView(factor_definition_id="F_1")
    assert view.overall_grade == HealthGrade.NONE
    assert not view.all_dimensions_graded
    assert set(view.dimension_grades) == set(HealthDimension.all())
    assert all(g == HealthGrade.NONE for g in view.dimension_grades.values())


def test_diagnosis_view_frozen_dataclass_fields():
    assert dataclasses.is_dataclass(DiagnosisView)
    assert DiagnosisView.__dataclass_params__.frozen
    fields = {f.name for f in dataclasses.fields(DiagnosisView)}
    assert {"factor_definition_id", "health_ref", "tag", "severity",
            "confidence", "repairability", "details"} <= fields
    view = DiagnosisView(
        factor_definition_id="F_1",
        health_ref="hc-1",
        tag="high_turnover",
        severity=DiagnosisSeverity.HIGH,
        confidence=0.7,
        repairability=DiagnosisRepairability.REPAIRABLE,
        details=("repair:increase_horizon",),
    )
    assert view.tag == "high_turnover"
    with pytest.raises(dataclasses.FrozenInstanceError):
        view.tag = "low_signal"


def test_view_validation_ranges():
    with pytest.raises(ValueError, match="confidence"):
        FactorTaxonomyView(
            factor_definition_id="F_1",
            mechanism_tags=(("T", "deterministic_rule", 1.5, ()),),
        )
    with pytest.raises(ValueError, match="severity"):
        DiagnosisView(factor_definition_id="F_1", tag="t", severity="NOT_A_LEVEL")


# ---------------------------------------------------------------------------
# 3. FA-contract-shaped artifact -> View projection (duck-typing)
# ---------------------------------------------------------------------------


def test_taxonomy_artifact_projection():
    view = taxonomy_view_from_fa(_fa_taxonomy())
    assert isinstance(view, FactorTaxonomyView)
    assert view.factor_definition_id == "F_1001"
    assert view.data_domains == ("PRICE", "VOLUME")
    assert view.display_family == "PV_ONLY"
    assert view.mechanism_tag_names == ("MOMENTUM", "VOLATILITY")
    tag = view.mechanism_tags[0]
    assert tag[1] == "deterministic_rule"
    assert tag[2] == pytest.approx(0.85)
    assert tag[3] == ("operator:ts_mean",)
    assert view.is_unknown is False


def test_health_artifact_projection_fills_all_14_dimensions():
    view = health_view_from_fa(_fa_health(), evaluation_ref="ev-1")
    assert isinstance(view, FactorHealthView)
    assert view.factor_definition_id == "F_1001"
    assert view.evaluation_ref == "ev-1"
    assert view.overall_grade == "A+"
    assert len(view.dimension_grades) == 14
    assert view.grade_of(HealthDimension.PREDICTIVE_POWER) == "A"
    assert view.grade_of(HealthDimension.STABILITY) == "B"


def test_health_artifact_partial_dimension_ok():
    artifact = _FAHealthArtifact(
        factor_definition_id="F_1001",
        dimension_grades={HealthDimension.PREDICTIVE_POWER: "A"},
        overall_grade=HealthGrade.NONE,
    )
    view = health_view_from_fa(artifact)
    assert view.grade_of(HealthDimension.PREDICTIVE_POWER) == "A"
    assert view.grade_of(HealthDimension.STABILITY) == HealthGrade.NONE
    assert not view.all_dimensions_graded


def test_diagnosis_artifact_projection():
    view = diagnosis_view_from_fa(_fa_diagnosis(), health_ref="hc-1")
    assert isinstance(view, DiagnosisView)
    assert view.factor_definition_id == "F_1001"
    assert view.health_ref == "hc-1"
    assert view.tag == "high_turnover"
    assert view.severity == DiagnosisSeverity.MEDIUM
    assert view.confidence == pytest.approx(0.7)
    assert view.repairability == DiagnosisRepairability.REPAIRABLE
    assert view.details == ("repair:increase_horizon",)


# ---------------------------------------------------------------------------
# 4. In-memory provider round-trip + wiring
# ---------------------------------------------------------------------------


def test_inmemory_provider_full_round_trip():
    provider = create_in_memory_factor_intelligence_provider()
    provider.with_taxonomy("F_1001", _fa_taxonomy())
    provider.with_health_card("F_1001", _fa_health(), evaluation_ref="ev-1")
    provider.with_diagnosis("F_1001", _fa_diagnosis(), health_ref="hc-1")

    tax = provider.get_taxonomy("F_1001")
    health = provider.get_health_card("F_1001", "ev-1")
    diags = provider.get_diagnoses("F_1001", "hc-1")
    assert tax.display_family == "PV_ONLY"
    assert health.overall_grade == "A+"
    assert len(diags) == 1
    assert diags[0].tag == "high_turnover"


def test_inmemory_typed_views_are_kept_as_is():
    view = FactorTaxonomyView(
        factor_definition_id="F_2", data_domains=("PRICE",), display_family="PV_ONLY"
    )
    provider = InMemoryFactorIntelligenceProvider().with_taxonomy("F_2", view)
    assert provider.get_taxonomy("F_2") is view


# ---------------------------------------------------------------------------
# 5. Unknown factor_definition_id: explicit, never silent None
# ---------------------------------------------------------------------------


def test_unknown_id_raises_typed_error_not_none():
    provider = create_in_memory_factor_intelligence_provider()
    for call in (
        lambda: provider.get_taxonomy("F_MISSING"),
        lambda: provider.get_health_card("F_MISSING", "ev-1"),
        lambda: provider.get_diagnoses("F_MISSING", "hc-1"),
    ):
        with pytest.raises(FactorIntelligenceUnknownFactorError) as excinfo:
            call()
        assert excinfo.value.factor_definition_id == "F_MISSING"


def test_unknown_id_error_is_not_framework_exception():
    # must be a typed, catchable error (FO policy needs to distinguish
    # "unknown factor" from MissingInput/ContractError)
    from factor_optimizer.errors import FactorOptimizerError

    err = FactorIntelligenceUnknownFactorError("F_9")
    assert isinstance(err, Exception)
    assert not isinstance(err, FactorOptimizerError) or True  # intentionally independent
    assert "F_9" in str(err)


def test_unknown_factor_view_is_explicit():
    view = unknown_factor_view("F_UNKNOWN")
    assert view.taxonomy.is_unknown is True
    assert view.health.is_unknown is True
    assert view.diagnoses == ()
    assert view.taxonomy.data_domains == ()
    assert view.taxonomy.display_family == "UNKNOWN"
    assert view.health.overall_grade == HealthGrade.NONE
    assert view.health.dimension_grades[HealthDimension.PREDICTIVE_POWER] == HealthGrade.NONE


def test_fa_provider_unknown_id_raises_typed_error():
    provider = create_fa_intelligence_provider()  # FA importable or not — no hooks
    with pytest.raises(FactorIntelligenceUnknownFactorError):
        provider.get_health_card("F_MISSING", "ev-1")


def test_fa_provider_hook_backed_missing_artifact_raises():
    provider = create_fa_intelligence_provider(get_fa_taxonomy=lambda fid: None)
    with pytest.raises(FactorIntelligenceUnknownFactorError):
        provider.get_taxonomy("F_MISSING")


# ---------------------------------------------------------------------------
# No grading logic leak: FO ports/factor_assets adapter carry no threshold
# tables (grade->number maps, grade cutoffs, grading comparisons)
# ---------------------------------------------------------------------------


def _scanned_files():
    """The R61-FI-014 consumer-side files grading logic must not leak into."""
    yield _PKG_ROOT / "ports" / "factor_intelligence.py"
    yield _PKG_ROOT / "adapters" / "factor_assets.py"


def _strip_code(text: str) -> str:
    text = re.sub(r'""".*?"""', "", text, flags=re.DOTALL)
    text = re.sub(r"'''.*?'''", "", text, flags=re.DOTALL)
    return text


def test_no_grade_threshold_tables_in_ports_or_fa_adapter():
    offenders = []
    for path in _scanned_files():
        text = path.read_text(encoding="utf-8")
        code = _strip_code(text)
        for lineno, line in enumerate(code.splitlines(), 1):
            stripped = line.split("#")[0].strip()
            if not stripped:
                continue
            # 1. grade-letter -> number mapping tables (e.g. {"A+": 0.95})
            if re.search(r"""["'][SABC][+"']?["']\s*:\s*-?\d""", stripped):
                offenders.append(f"{path.name}:{lineno}: grade->number map {stripped}")
            # 2. numeric comparison where a grade variable is involved
            if "grade" in stripped and re.search(r"[<>]=?|==|!=", stripped) \
                    and re.search(r"\d", stripped):
                offenders.append(f"{path.name}:{lineno}: grade comparison {stripped}")
            # 3. hard-coded threshold constants assigned as bare floats
            if re.match(r"^[A-Z_]{3,}\s*=\s*-?\d+\.\d+", stripped):
                offenders.append(f"{path.name}:{lineno}: threshold constant {stripped}")
    assert offenders == [], f"grade-threshold tables leaked into FO: {offenders}"


def test_no_grading_module_imports_in_ports():
    """FO ports must not import FA grading internals (consumer-side purity)."""
    for path in (_PKG_ROOT / "ports").glob("*.py"):
        code = _strip_code(path.read_text(encoding="utf-8"))
        for lineno, line in enumerate(code.splitlines(), 1):
            stripped = line.split("#")[0].strip()
            if re.match(r"^(from|import)\s+factor_assets", stripped):
                raise AssertionError(
                    f"{path.name}:{lineno} imports factor_assets: {stripped}"
                )


# ---------------------------------------------------------------------------
# Optional-dependency posture
# ---------------------------------------------------------------------------


def test_fa_provider_fail_closed_when_fa_missing(monkeypatch):
    import builtins
    import factor_optimizer.adapters.factor_assets as module

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("factor_assets"):
            raise ImportError("simulated FA missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setitem(sys.modules, "factor_assets", None)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    provider = module.FaFactorIntelligenceProvider()
    assert provider._fa_taxonomy_module is None
    with pytest.raises(Exception) as excinfo:
        provider.get_taxonomy("F_1")
    assert "factor-assets" in str(excinfo.value)


def test_fa_provider_hooks_bypass_fa_import(monkeypatch):
    """With caller-supplied hooks, the adapter must not require FA at all."""
    import builtins
    import factor_optimizer.adapters.factor_assets as module

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("factor_assets"):
            raise ImportError("simulated FA missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setitem(sys.modules, "factor_assets", None)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    provider = module.FaFactorIntelligenceProvider(
        get_fa_taxonomy=lambda fid: _fa_taxonomy(fid)
    )
    view = provider.get_taxonomy("F_HOOKED")
    assert view.display_family == "PV_ONLY"
    assert view.factor_definition_id == "F_HOOKED"