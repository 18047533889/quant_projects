from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from factor_engine.backend.cleaned_bridge import panel_to_series

CTX=SimpleNamespace(timestamp_col='decision_time',instrument_col='asset')

def reference(panel,template=None):
    result=panel.stack(future_stack=True)
    result.index.names=[CTX.timestamp_col,CTX.instrument_col]
    if template is None: return result
    target=template if isinstance(template,pd.Index) else template.index
    return result if len(result)==len(target) and result.index.equals(target) else result.reindex(target)

def panel():
    return pd.DataFrame([[1.,np.nan,np.inf],[4.,-np.inf,6.],[7.,8.,9.]],
        index=pd.date_range('2024-01-01',periods=3,name='original_time'),
        columns=pd.Index(['c','a','b'],name='original_columns'))

@pytest.mark.parametrize('timezone',[None,'Asia/Hong_Kong'])
@pytest.mark.parametrize('numeric_columns',[False,True])
@pytest.mark.parametrize('target_kind',['none','reversed','subset','series','missing'])
def test_dense_axis_null_name_and_template_parity(monkeypatch,timezone,numeric_columns,target_kind):
    frame=panel().iloc[::-1]
    if timezone: frame.index=frame.index.tz_localize(timezone)
    if numeric_columns: frame.columns=pd.Index([20,3,7],name='codes')
    target=reference(frame).index
    if target_kind=='none': target=None
    elif target_kind=='reversed': target=target[::-1]
    elif target_kind=='subset': target=target[::2]
    elif target_kind=='series': target=pd.Series(0.,index=target,name='template_name')
    elif target_kind=='missing': target=target.append(pd.MultiIndex.from_tuples([(frame.index[0],999 if numeric_columns else 'missing')],names=target.names))
    expected=reference(frame,target)
    def forbidden(*args,**kwargs): raise AssertionError('eligible panel must not stack')
    monkeypatch.setattr(pd.DataFrame,'stack',forbidden)
    result=panel_to_series(frame,CTX,template=target)
    pd.testing.assert_series_equal(result,expected)
    assert not np.shares_memory(result.to_numpy(),frame.to_numpy())
    original=frame.copy(); result.iloc[0]=12345.
    pd.testing.assert_frame_equal(frame,original)

@pytest.mark.parametrize('case',['empty_rows','empty_columns','nan_column','nat_index','categorical_columns','categorical_index','nullable','mixed','duplicate_index','numeric_index','attrs','string_extension'])
def test_ineligible_panels_use_reference_stack(monkeypatch,case):
    frame=panel()
    if case=='empty_rows': frame=frame.iloc[:0]
    elif case=='empty_columns': frame=frame.iloc[:,:0]
    elif case=='nan_column': frame.columns=['a',np.nan,'b']
    elif case=='nat_index': frame.index=pd.DatetimeIndex([pd.NaT,'2024-01-01','2024-01-02'])
    elif case=='categorical_columns': frame.columns=pd.CategoricalIndex(['c','a','b'],categories=['a','b','c','unused'],ordered=True)
    elif case=='categorical_index': frame.index=pd.CategoricalIndex(['t2','t1','t3'])
    elif case=='nullable': frame=frame.astype('Float64')
    elif case=='mixed': frame['a']=['x','y','z']
    elif case=='duplicate_index': frame.index=pd.DatetimeIndex(['2024-01-01','2024-01-01','2024-01-02'])
    elif case=='numeric_index': frame.index=pd.Index([2,0,1])
    elif case=='attrs': frame.attrs={'nested':{'source':'test'}}
    elif case=='string_extension': frame.columns=pd.Index(['c','a','b'],dtype='string')
    if case=='nan_column':
        # pandas future_stack currently rejects NaN column labels as duplicate.
        # Preserve that reference exception rather than inventing new behavior.
        with pytest.raises(ValueError): reference(frame)
        with pytest.raises(ValueError): panel_to_series(frame,CTX)
        return
    expected=reference(frame)
    original=pd.DataFrame.stack; calls=[]
    def counted(self,*args,**kwargs): calls.append(True); return original(self,*args,**kwargs)
    monkeypatch.setattr(pd.DataFrame,'stack',counted)
    result=panel_to_series(frame,CTX)
    pd.testing.assert_series_equal(result,expected)
    assert result.attrs==expected.attrs
    assert calls==[True]

def test_duplicate_columns_keep_reference_error():
    frame=panel(); frame.columns=['a','a','b']
    with pytest.raises(ValueError): reference(frame)
    with pytest.raises(ValueError): panel_to_series(frame,CTX)

def test_prefix_preserves_time_axis():
    frame=panel()
    expected=panel_to_series(frame,CTX)
    prefix=panel_to_series(frame.iloc[:2],CTX)
    pd.testing.assert_series_equal(prefix,expected.iloc[:6])
