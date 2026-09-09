import pytest
from factor_engine.api.static_analysis import analyze_factor_definition
from data_access.read.semantic_catalog import SemanticField,SemanticFieldCatalog,SemanticFieldTaxonomyProvider
from factor_assets.adapters.production_taxonomy import classify_definition_taxonomy


def inputs():
    dsl = 'close + open + high + low'
    analysis = analyze_factor_definition(dsl)
    ids = tuple(u.canonical_field_id for u in analysis.field_usages)
    assert len(ids) == 4
    roles = dict(zip(ids,('alpha','control','eligibility','weight')))
    fields = {}
    for index,fid in enumerate(ids):
        canonical = 'resolved.'+fid
        fields[canonical] = SemanticField(logical_name=canonical,dataset='fixture',physical_name=f'field{index}',
            aliases=(fid,),data_domains=('PRICE',) if index == 0 else ('FUNDAMENTAL.QUALITY',))
    return dsl,ids,roles,SemanticFieldTaxonomyProvider(SemanticFieldCatalog(fields))


def test_actual_fe_da_fa_path_resolves_aliases_and_preserves_all_usage_roles():
    dsl,ids,roles,provider = inputs()
    result = classify_definition_taxonomy(dsl,taxonomy_provider=provider,field_roles=roles)
    assert result.alpha_source_fields == ('resolved.'+ids[0],)
    assert result.control_fields == ('resolved.'+ids[1],)
    assert result.eligibility_fields == ('resolved.'+ids[2],)
    assert result.weight_fields == ('resolved.'+ids[3],)
    assert 'PRICE' in result.data_domains
    assert any('QUALITY' in domain for domain in result.data_domains)
    # All sources are visible, but controls cannot inject fundamental alpha mechanisms.
    assert not any('QUALITY' in tag.tag.upper() or 'VALUE' in tag.tag.upper() for tag in result.mechanism_tags)


def test_missing_roles_or_catalog_fields_are_not_guessed():
    dsl,ids,roles,provider = inputs()
    with pytest.raises(ValueError,match='exactly bind'):
        classify_definition_taxonomy(dsl,taxonomy_provider=provider,field_roles={})
    with pytest.raises(Exception,match='未登记|未知|not found|not registered'):
        classify_definition_taxonomy(dsl,taxonomy_provider=SemanticFieldTaxonomyProvider(SemanticFieldCatalog({})),field_roles=roles)
    with pytest.raises(ValueError,match='usage role'):
        classify_definition_taxonomy(dsl,taxonomy_provider=provider,field_roles={**roles,ids[0]:'price_level'})


def test_registered_unknown_semantics_remain_unknown():
    dsl='rank(close)'
    fid=analyze_factor_definition(dsl).field_usages[0].canonical_field_id
    provider=SemanticFieldTaxonomyProvider(SemanticFieldCatalog({fid:SemanticField(logical_name=fid,dataset='fixture',physical_name='c')}))
    result=classify_definition_taxonomy(dsl,taxonomy_provider=provider,field_roles={fid:'alpha'})
    assert 'UNKNOWN' in result.data_domains
