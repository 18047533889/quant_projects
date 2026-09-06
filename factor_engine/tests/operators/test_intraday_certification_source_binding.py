import hashlib
import json
import pytest
from factor_engine.cleaned_operators import r23_cert_intraday as cert


@pytest.fixture
def bound_artifact(tmp_path, monkeypatch):
    root=tmp_path/'factor_engine'
    monkeypatch.setattr(cert,'FE_ROOT',root)
    paths={
        'operator_source_hash':root/'cleaned_operators/microstructure/intraday_agg.py',
        'test_file_hash':root/'tests/backend_parity/test_intraday_minute_parity.py',
        'helper_file_hash':root/'tests/backend_parity/intraday_minute_parity.py',
    }
    data={'schema_version':'intraday_minute_parity.v1',
          'backends':{'intra_amihud':{'polars':True,'duckdb_sql':True,'status':'certified'}}}
    for key,path in paths.items():
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('original '+key)
        data[key]=hashlib.sha256(path.read_bytes()).hexdigest()
    artifact=tmp_path/'evidence/intraday_minute_parity.json'
    artifact.parent.mkdir(); artifact.write_text(json.dumps(data))
    return paths,data,artifact


def test_current_bound_artifact_admitted(bound_artifact):
    assert cert._minute_shape_from_artifact()==frozenset({'intra_amihud'})


@pytest.mark.parametrize('key',['operator_source_hash','test_file_hash','helper_file_hash'])
def test_changed_source_rejects_historical_green(bound_artifact,key):
    paths,_,_=bound_artifact
    paths[key].write_text('changed semantic source')
    assert not cert._minute_shape_from_artifact()


@pytest.mark.parametrize('key',['operator_source_hash','test_file_hash','helper_file_hash','schema_version'])
def test_missing_binding_rejected(bound_artifact,key):
    _,data,artifact=bound_artifact
    del data[key]; artifact.write_text(json.dumps(data))
    assert not cert._minute_shape_from_artifact()


def test_unreadable_source_rejected(bound_artifact):
    paths,_,_=bound_artifact
    paths['operator_source_hash'].unlink()
    assert not cert._minute_shape_from_artifact()
