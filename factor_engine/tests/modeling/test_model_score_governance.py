# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pytest

from modeling.dsl_bridge import (
    ArtifactResolver,
    ArtifactStore,
    FrozenScorer,
    ModelArtifactResolutionContext,
    ModelScoreResolutionStatus,
    NoLegalArtifactError,
    _score_block,
    lower_model_score,
    model_score,
)
from modeling.model_catalog import ModelArtifactCatalog, validate_artifact_id
from tests.modeling.test_regime_moe_dsl_bridge import _make_artifact


def test_model_score_lowers_to_typed_ir_without_binding_current_artifact():
    expr = model_score("alpha", "quality", "momentum")
    ir = lower_model_score(expr)

    assert ir.model_name == "alpha"
    assert ir.feature_read.feature_names == ("quality", "momentum")
    assert ir.frozen_score.resolution is ir.resolution
    assert ir.semantic_identity == {
        "op": "model_score",
        "model_name": "alpha",
        "resolution_policy": "latest_legal_asof",
        "missing_artifact_policy": "fail_closed",
        "feature_names": ("quality", "momentum"),
        "output_semantic_kind": "ModelScoreBlock",
    }
    assert "artifact_id" not in ir.semantic_identity


def test_no_legal_artifact_is_explicit_and_fail_closed():
    resolver = ArtifactResolver(store=ArtifactStore())
    ir = lower_model_score(model_score("alpha"))
    result = resolver.resolve_result("alpha", "2020-01-01")
    assert result.status is ModelScoreResolutionStatus.NO_LEGAL_ARTIFACT

    with pytest.raises(NoLegalArtifactError, match="NO_LEGAL_ARTIFACT"):
        _score_block(ir, np.zeros((3, 2)), "2020-01-01", resolver)


def test_research_nan_policy_is_explicit():
    resolver = ArtifactResolver(store=ArtifactStore())
    ir = lower_model_score(model_score("alpha", missing_artifact_policy="nan"))
    block = _score_block(ir, np.zeros((3, 2)), "2020-01-01", resolver)
    assert np.isnan(block.values).all()
    assert block.cache_identity[-1] == "NO_LEGAL_ARTIFACT"


def test_frozen_scorer_has_no_fit_surface():
    artifact = _make_artifact("alpha", "2020-01-01", "2020-01-01")
    scorer = FrozenScorer(artifact)
    assert not hasattr(scorer, "fit")
    X = np.random.default_rng(4).normal(size=(8, 2))
    np.testing.assert_allclose(scorer.score(X), artifact.predict(X))


def test_artifact_id_machine_validator_rejects_path_escape():
    assert validate_artifact_id("alpha:v1.2_3") == "alpha:v1.2_3"
    for invalid in ("../alpha", "a/b", "a b", "alpha\x00x", "模型"):
        with pytest.raises(ValueError, match="artifact_id"):
            validate_artifact_id(invalid)


def test_catalog_survives_restart_and_scopes_resolution(tmp_path):
    store = ArtifactStore(str(tmp_path / "artifacts"))
    catalog_path = tmp_path / "model_catalog.sqlite"
    old = _make_artifact("alpha", "2020-01-01", "2020-01-01", "1.0", seed=1)
    new = _make_artifact("alpha", "2021-01-01", "2021-01-01", "2.0", seed=2)
    ctx = ModelArtifactResolutionContext(
        tenant="tenant-a", project="proj-a", market="ashare", namespace="prod"
    )

    catalog = ModelArtifactCatalog(catalog_path)
    resolver = ArtifactResolver(store=store, catalog=catalog)
    resolver.register(old, context=ctx)
    resolver.register(new, context=ctx)
    catalog.close()

    reopened = ModelArtifactCatalog(catalog_path)
    resolver2 = ArtifactResolver(store=store, catalog=reopened)
    assert resolver2.resolve("alpha", "2020-06-01", context=ctx).artifact_id == old.artifact_id
    assert resolver2.resolve("alpha", "2021-06-01", context=ctx).artifact_id == new.artifact_id
    assert resolver2.resolve(
        "alpha",
        "2021-06-01",
        context=ModelArtifactResolutionContext(
            tenant="other", project="proj-a", market="ashare", namespace="prod"
        ),
    ) is None
    reopened.close()


def test_catalog_revocation_removes_artifact_from_legal_snapshot(tmp_path):
    store = ArtifactStore(str(tmp_path / "artifacts"))
    catalog = ModelArtifactCatalog(tmp_path / "catalog.sqlite")
    artifact = _make_artifact("alpha", "2020-01-01", "2020-01-01")
    resolver = ArtifactResolver(store=store, catalog=catalog)
    resolver.register(artifact)
    assert resolver.resolve("alpha", "2021-01-01") is not None

    catalog.revoke(artifact.artifact_id)
    assert resolver.resolve_result(
        "alpha", "2021-01-01"
    ).status is ModelScoreResolutionStatus.NO_LEGAL_ARTIFACT
    catalog.close()


def test_score_block_cache_identity_binds_asof_artifact_lineage():
    old = _make_artifact("alpha", "2020-01-01", "2020-01-01", "1.0", seed=1)
    new = _make_artifact("alpha", "2021-01-01", "2021-01-01", "2.0", seed=2)
    resolver = ArtifactResolver(store=ArtifactStore())
    resolver.register(old)
    resolver.register(new)
    ir = lower_model_score(model_score("alpha"))
    X = np.random.default_rng(9).normal(size=(5, 2))

    old_block = _score_block(ir, X, "2020-06-01", resolver)
    new_block = _score_block(ir, X, "2021-06-01", resolver)
    assert old_block.artifact_id == old.artifact_id
    assert new_block.artifact_id == new.artifact_id
    assert old_block.cache_identity != new_block.cache_identity
    assert old.artifact_id in old_block.cache_identity
    assert new.artifact_id in new_block.cache_identity
