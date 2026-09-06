import hashlib
import json
from dataclasses import replace
import pytest
from factor_engine.fields.registry import FieldRegistry
from factor_engine.fields.spec import FieldSpec, TableSpec


def registry():
    return FieldRegistry([FieldSpec('x','prices','x',metadata={'nested': {'items': [1, 2]}})],
                         [TableSpec('prices','prices',metadata={'version': 1})])


def fresh_hash(r):
    return hashlib.sha256(json.dumps(r.export_catalog(),sort_keys=True,separators=(',',':'),ensure_ascii=True,default=str).encode()).hexdigest()


def test_cached_hash_does_not_export_again(monkeypatch):
    r=registry(); expected=fresh_hash(r)
    assert r.catalog_hash()==expected
    def forbidden(): raise AssertionError('warm hash must not export/asdict')
    monkeypatch.setattr(r,'export_catalog',forbidden)
    for _ in range(100): assert r.catalog_hash()==expected


@pytest.mark.parametrize('change', ['field_register','field_replace','table_register','table_replace','nested','table_metadata','direct_replace'])
def test_content_changes_invalidate_hash(change):
    r=registry(); old=r.catalog_hash()
    if change=='field_register': r.register(FieldSpec('y','prices','y'))
    elif change=='field_replace': r.register(replace(r.require('x'),nullable=False),replace=True)
    elif change=='table_register': r.register_table(TableSpec('other','other'))
    elif change=='table_replace': r.register_table(replace(r.resolve_table('prices'),description='changed'),replace=True)
    elif change=='nested': r.require('x').metadata['nested']['items'].append(3)
    elif change=='table_metadata': r.resolve_table('prices').metadata['version']=2
    elif change=='direct_replace': r._fields['prices.x']=replace(r.require('x'),description='changed')
    assert r.catalog_hash()==fresh_hash(r)
    assert r.catalog_hash()!=old


def test_unknown_mutable_metadata_bypasses_cache():
    class Mutable:
        def __init__(self): self.value=1
        def __str__(self): return str(self.value)
    r=registry(); obj=Mutable(); r.require('x').metadata['custom']=obj
    old=r.catalog_hash(); obj.value=2
    assert r.catalog_hash()!=old
    assert r.catalog_hash()==fresh_hash(r)


def test_failed_registration_does_not_change_digest():
    r=registry(); before=r.catalog_hash()
    with pytest.raises(ValueError): r.register(FieldSpec('x','prices','x'))
    assert r.catalog_hash()==before


def test_non_metadata_mutable_attribute_bypasses_cache():
    class Mutable:
        def __init__(self): self.value=1
        def __str__(self): return str(self.value)
    r=registry(); obj=Mutable()
    r.register(replace(r.require('x'),description=obj),replace=True)
    before=r.catalog_hash(); obj.value=2
    assert r.catalog_hash()!=before
    assert r.catalog_hash()==fresh_hash(r)


@pytest.mark.parametrize('before,after', [(0.0,-0.0), (-0.0,0.0), (float('inf'),float('-inf')), (float('nan'),1.0)])
def test_float_metadata_tracks_json_identity(before, after):
    r=registry(); r.require('x').metadata['nested']['float']=before
    old=r.catalog_hash()
    r.require('x').metadata['nested']['float']=after
    assert r.catalog_hash()!=old
    assert r.catalog_hash()==fresh_hash(r)


def test_nan_metadata_can_hit_cache(monkeypatch):
    r=registry(); r.require('x').metadata['nan']=float('nan')
    old=r.catalog_hash()
    monkeypatch.setattr(r,'export_catalog',lambda: pytest.fail('NaN token should be stable'))
    assert r.catalog_hash()==old


def test_warm_cache_does_not_rehash_frozen_specs(monkeypatch):
    r=registry(); expected=r.catalog_hash()
    def forbidden(*args): raise AssertionError('warm fingerprint must not hash every dataclass field')
    monkeypatch.setattr(FieldSpec,'__hash__',forbidden)
    monkeypatch.setattr(TableSpec,'__hash__',forbidden)
    assert r.catalog_hash()==expected


def test_subclass_with_dynamic_serialization_bypasses_cache():
    class DynamicField(FieldSpec):
        revision=1
        def to_dict(self):
            result=super().to_dict(); result['dynamic']=self.revision
            return result
    r=registry(); r.register(DynamicField('y','prices','y'))
    before=r.catalog_hash(); DynamicField.revision=2
    assert r.catalog_hash()!=before
    assert r.catalog_hash()==fresh_hash(r)
