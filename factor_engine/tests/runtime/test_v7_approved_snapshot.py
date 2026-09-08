from types import SimpleNamespace

import pytest

from factor_engine.runtime import default_engine as module
from factor_engine.runtime.default_execution_policy import DefaultExecutionPolicy
from factor_engine.storage.sources.data_access_source import DataAccessSource
from factor_engine.tests.runtime.test_v6_default_engine import profile


def _source(token):
    source = object.__new__(DataAccessSource)
    source._manifest_token = token
    source._data_snapshot_id = None
    source.dataset = 'ashare_stock_daily_adj'
    source.refresh_snapshot = lambda **kwargs: source._manifest_token
    source.close = lambda: None
    source.bind_resource_broker = lambda broker: None
    return source


def test_worker_factory_rejects_replacement_since_parent_approval(tmp_path, monkeypatch):
    business = profile(tmp_path)
    business['expected_snapshot_token'] = 'manifest:approved:1'
    config = module.ExecutionCoreWorkerConfig(
        module.ApprovedDeploymentProfile.from_mapping(business), DefaultExecutionPolicy())
    source = _source('manifest:replacement:2')
    monkeypatch.setattr('factor_engine.storage.factory.build_data_source', lambda *a, **k: source)
    monkeypatch.setattr(module, '_validate_hfq_source_contract', lambda source: None)
    monkeypatch.setattr('factor_engine.backend.factory.build_backend', lambda *a: object())
    monkeypatch.setattr('factor_engine.runtime.engine.FactorEngine', lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr('factor_engine.runtime.resource_broker.get_v2_resource_broker', lambda p: object())
    with pytest.raises(ValueError, match='approved.*snapshot'):
        module.build_execution_core_from_worker_config(config)


def test_worker_factory_propagates_approved_content_digest(tmp_path, monkeypatch):
    business = profile(tmp_path)
    approved = business['expected_source_content_digests'][business['data_source']['dataset']]
    config = module.ExecutionCoreWorkerConfig(
        module.ApprovedDeploymentProfile.from_mapping(business), DefaultExecutionPolicy())
    source = _source(None)
    monkeypatch.setattr('factor_engine.storage.factory.build_data_source', lambda *a, **k: source)
    monkeypatch.setattr(module, '_validate_hfq_source_contract', lambda source: None)
    monkeypatch.setattr('factor_engine.backend.factory.build_backend', lambda *a: object())
    monkeypatch.setattr('factor_engine.runtime.engine.FactorEngine', lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr('factor_engine.runtime.resource_broker.get_v2_resource_broker', lambda p: object())

    module.build_execution_core_from_worker_config(config)
    assert source._approved_content_digest == approved
    assert source._approved_source_content_digests == {source.dataset: approved}


def test_bound_source_cannot_accept_a_later_consistent_snapshot():
    source = _source('manifest:approved:1')
    source.bind_approved_snapshot_token('manifest:approved:1')
    source._manifest_token = 'manifest:replacement:2'
    with pytest.raises(ValueError, match='approved.*snapshot'):
        source.assert_approved_snapshot()
    with pytest.raises(ValueError, match='rebound'):
        source.bind_approved_snapshot_token('manifest:replacement:2')


@pytest.mark.parametrize('snapshots', [{}, {'secondary': 'v1'}, {'ashare_stock_daily_adj': ''}])
def test_approved_multi_source_set_requires_anchor_and_nonempty_tokens(tmp_path, snapshots):
    business = profile(tmp_path)
    business['expected_source_snapshot_tokens'] = snapshots
    with pytest.raises(module.DeploymentConfigurationError):
        module.ApprovedDeploymentProfile.from_mapping(business)


@pytest.mark.parametrize('digests', [
    None, {}, {'secondary': 'a' * 32}, {'ashare_stock_daily_adj': 'not-a-digest'},
    {'ashare_stock_daily_adj': 'A' * 32},
])
def test_v7_profile_requires_approved_anchor_content_digest(tmp_path, digests):
    business = profile(tmp_path)
    if digests is None:
        business.pop('expected_source_content_digests')
    else:
        business['expected_source_content_digests'] = digests
    with pytest.raises(module.DeploymentConfigurationError, match='content.digest'):
        module.ApprovedDeploymentProfile.from_mapping(business)


@pytest.mark.parametrize('change_during_prepare', [False, True])
def test_preparation_rechecks_approval_and_releases_failed_reservation(monkeypatch, change_during_prepare):
    from factor_engine.planner.logical_plan import PlanNode
    from factor_engine.runtime.physical_source_binding import bind_batch_sources

    source = _source('approved')
    source.production = source.pit_enforce = True
    source.params, source.instrument_filter = {}, ['A']
    source.start_date, source.end_date = '2024-01-01', '2024-01-02'
    source.bind_approved_snapshot_token('approved')
    source._ensure_field_plans = lambda fields: {
        name: SimpleNamespace(physical_fields=('Close',)) for name in fields}
    if not change_during_prepare:
        source._manifest_token = 'replacement'
    calls, released = [], []
    lease = object()

    def prepare(*args, **kwargs):
        calls.append(args)
        source._manifest_token = 'replacement'
        return SimpleNamespace(resource_reservation=lease,
                               resolved_source_snapshot=SimpleNamespace(content_digest='new'))

    monkeypatch.setattr('data_access.get_store', lambda: SimpleNamespace(
        prepare_read=prepare, _pipeline=SimpleNamespace(release_reservation=released.append)))
    scope = SimpleNamespace(universe_id='fixture', scope_key=lambda: 'fixture')
    dag = SimpleNamespace(roots=[SimpleNamespace(factor_name='a', execution_scope=scope,
        root=PlanNode('column', attrs={'name': 'Close'}))], shared_nodes={})
    with pytest.raises(ValueError, match='approved.*snapshot'):
        bind_batch_sources(dag, source=source, execution_context=SimpleNamespace())
    assert len(calls) == int(change_during_prepare)
    assert released == ([lease] if change_during_prepare else [])


def test_prepared_content_identity_rejects_aba_and_releases_reservation(monkeypatch):
    from factor_engine.planner.logical_plan import PlanNode
    from factor_engine.runtime import physical_source_binding as binding_module

    source = _source('manifest:approved:1')
    source.production = source.pit_enforce = True
    source.params, source.instrument_filter = {}, ['A']
    source.start_date, source.end_date = '2024-01-01', '2024-01-02'
    source.bind_approved_snapshot_token('manifest:approved:1')
    source._approved_source_snapshot_tokens = {
        source.dataset: 'manifest:approved:1',
    }
    source._approved_source_content_digests = {
        source.dataset: 'a' * 32,
    }
    source._ensure_field_plans = lambda fields: {
        name: SimpleNamespace(physical_fields=('Close',)) for name in fields}
    released = []
    lease = object()

    def prepare(*args, **kwargs):
        # The mutable manifest identity returns to A, but the prepared read has
        # already frozen B.  Pre/post checks of only the live token miss this ABA.
        source._manifest_token = 'manifest:replacement:2'
        frozen = SimpleNamespace(dataset=source.dataset, content_digest='b' * 32)
        source._manifest_token = 'manifest:approved:1'
        return SimpleNamespace(resource_reservation=lease, resolved_source_snapshot=frozen)

    monkeypatch.setattr('data_access.get_store', lambda: SimpleNamespace(
        prepare_read=prepare, _pipeline=SimpleNamespace(release_reservation=released.append)))
    monkeypatch.setattr(binding_module, 'bind_plan_sources', lambda node, **kwargs: node)
    scope = SimpleNamespace(universe_id='fixture', scope_key=lambda: 'fixture')
    dag = SimpleNamespace(roots=[SimpleNamespace(factor_name='a', execution_scope=scope,
        root=PlanNode('column', attrs={'name': 'Close'}))], shared_nodes={})

    with pytest.raises(ValueError, match='approved.*content'):
        binding_module.bind_batch_sources(
            dag, source=source, execution_context=SimpleNamespace())
    assert released == [lease]


def test_approved_content_set_rejects_missing_secondary_before_prepare(monkeypatch):
    from factor_engine.planner.logical_plan import PlanNode
    from factor_engine.runtime.physical_source_binding import bind_batch_sources

    anchor = _source(None)
    anchor._approved_source_snapshot_tokens = {}
    anchor._approved_source_content_digests = {anchor.dataset: 'a' * 32}
    secondary = _source(None)
    secondary.dataset = 'stock_income'
    secondary.production = secondary.pit_enforce = True
    scope_key = object()
    binding = SimpleNamespace(
        encoded_column='encoded-secondary', source_scope=scope_key, field='Revenue')
    discovered = SimpleNamespace(
        bindings={'encoded-secondary': binding},
        source_ref_columns_seen={'encoded-secondary'},
        source_ref_columns_bound={'encoded-secondary'},
    )
    monkeypatch.setattr(
        'factor_engine.planner.source_binding.discover_column_source_bindings',
        lambda plans: discovered,
    )
    monkeypatch.setattr(
        'factor_engine.planner.batch_data_request.BatchSourceResolver.resolve_source',
        lambda self, source_scope: secondary,
    )
    prepare_calls = []
    monkeypatch.setattr('data_access.get_store', lambda: SimpleNamespace(
        prepare_read=lambda *a, **k: prepare_calls.append((a, k))))
    scope = SimpleNamespace(universe_id='fixture', scope_key=lambda: 'fixture')
    dag = SimpleNamespace(roots=[SimpleNamespace(
        factor_name='a', execution_scope=scope,
        root=PlanNode('column', attrs={'name': 'encoded-secondary'}),
    )], shared_nodes={})

    with pytest.raises(ValueError, match='content digest set omits.*dependent'):
        bind_batch_sources(
            dag, source=anchor,
            execution_context=SimpleNamespace(data_source=anchor, market='ashare'))
    assert prepare_calls == []
