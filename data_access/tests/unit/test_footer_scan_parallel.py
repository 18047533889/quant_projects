import threading
import time
from types import SimpleNamespace
import pyarrow.parquet as pq
from data_access.registry.schema_validation import _missing_required_columns_per_file


def test_footer_scan_is_bounded_complete_and_deterministic(monkeypatch):
    lock = threading.Lock()
    state = {'active':0, 'peak':0, 'seen':[]}
    monkeypatch.setattr('os.cpu_count', lambda:4)
    def metadata(path):
        with lock:
            state['active']+=1
            state['peak']=max(state['peak'],state['active'])
            state['seen'].append(path)
        try:
            time.sleep(.005)
            if path=='3.parquet':
                raise OSError('manifest owns read error handling')
            cols=['a'] if path=='1.parquet' else ['b'] if path=='2.parquet' else ['a','b']
            return SimpleNamespace(schema=SimpleNamespace(names=cols))
        finally:
            with lock:
                state['active']-=1
    monkeypatch.setattr(pq,'read_metadata',metadata)
    paths=[str(i)+'.parquet' for i in range(25)]
    result=_missing_required_columns_per_file(SimpleNamespace(format='parquet'),
        paths+paths[:2]+['ignore.txt'],{'a':'float','b':'float','partition':'date'},{'partition'})
    assert result==['b','a']
    assert sorted(state['seen'])==sorted(paths)
    assert 1 < state['peak'] <= 4


def test_empty_and_nonparquet_require_no_reads(monkeypatch):
    def forbidden(*a,**k):
        raise AssertionError('unexpected metadata read')
    monkeypatch.setattr(pq,'read_metadata',forbidden)
    assert _missing_required_columns_per_file(SimpleNamespace(format='csv'),['x.parquet'],{'a':'float'},set())==[]
    assert _missing_required_columns_per_file(SimpleNamespace(format='parquet'),[],{'a':'float'},set())==[]
