from __future__ import annotations
import numpy as np, pandas as pd
from storage.datasource import DataSource

class Source(DataSource):
    def __init__(self,data): self.data=data
    def load_column(self,name): return self.data[name]
    def load_columns(self,names): return {n:self.data[n] for n in names if n in self.data}
    def prefetch_columns(self,names): return self.load_columns(names)

def synthetic_panel(n_days=820,n_assets=20,seed=20260802):
    rng=np.random.default_rng(seed)
    dates=pd.bdate_range('2020-01-02',periods=n_days)
    assets=[f'S{i:03d}' for i in range(n_assets)]
    idx=pd.MultiIndex.from_product([dates,assets],names=['timestamp','instrument'])
    market=rng.normal(0.0002,0.009,(n_days,1)); beta=rng.uniform(0.6,1.4,(1,n_assets)); idio=rng.normal(0,0.012,(n_days,n_assets))
    ret=np.clip(market*beta+idio,-0.18,0.18)
    close=np.empty_like(ret); close[0]=rng.uniform(8,120,n_assets)
    for t in range(1,n_days): close[t]=close[t-1]*(1+ret[t])
    pre=np.vstack([close[0],close[:-1]])
    overnight=np.clip(rng.normal(0,0.004,(n_days,n_assets))+market*.2,-0.06,0.06); open_=pre*(1+overnight); intra=close/open_-1
    spread=np.abs(rng.normal(.012,.006,(n_days,n_assets)))
    high=np.maximum(open_,close)*(1+spread*rng.uniform(.2,1,(n_days,n_assets))); low=np.maximum(np.minimum(open_,close)*(1-spread*rng.uniform(.2,1,(n_days,n_assets))),0.01)
    vwap=(open_+high+low+close)/4
    base_vol=rng.lognormal(13,0.8,(1,n_assets)); volume=base_vol*np.exp(rng.normal(0,.45,(n_days,n_assets)))*(1+6*np.abs(ret)); amount=volume*vwap
    avgvol=pd.DataFrame(volume).rolling(20,min_periods=1).mean().to_numpy(); shares=rng.lognormal(18,0.7,(1,n_assets)); circ=shares*rng.uniform(.45,.95,(1,n_assets)); mcap=close*shares; turn=volume/np.maximum(circ,1)
    # Heterogeneous, time-varying quoted spread; wider for less liquid names and volatile days.
    rel_qspread=np.clip(rng.lognormal(-7.2,.45,(n_days,n_assets))*(1+3*np.abs(ret))*(1+0.15*np.arange(n_assets)[None,:]/n_assets),2e-5,.02)
    bid=close*(1-rel_qspread/2); ask=close*(1+rel_qspread/2)
    bid_size=volume*rng.uniform(.005,.04,(n_days,n_assets)); ask_size=volume*rng.uniform(.005,.04,(n_days,n_assets))
    short_ratio=np.clip(rng.beta(2,8,(n_days,n_assets)),0,1); short_interest=avgvol*rng.uniform(1,15,(1,n_assets)); days_to_cover=short_interest/np.maximum(avgvol,1)
    industry=np.tile(np.arange(n_assets)%6,(n_days,1)).astype(float); exchange=np.tile(np.arange(n_assets)%2,(n_days,1)).astype(float)
    # Positive, stock-specific adjustment-factor paths with staggered corporate actions.
    adj=np.ones((n_days,n_assets)); factor=np.ones((n_days,n_assets))
    for j in range(n_assets):
        dates_j=[120+(j*23)%250, 420+(j*17)%250]
        mults=[1.2+0.1*(j%4), 1.1+0.05*(j%5)]
        for d,m in zip(dates_j,mults):
            if d<n_days: adj[d:,j]*=m
        dates_f=[80+(j*31)%300, 390+(j*19)%300]
        for d,m in zip(dates_f,mults[::-1]):
            if d<n_days: factor[d:,j]*=m
    q=((np.arange(n_days)//63)%4)+1; year=2020+(np.arange(n_days)//252); period=(year*10+q).reshape(-1,1)*np.ones((1,n_assets))
    rev0=rng.lognormal(20,1,(1,n_assets)); trend=rng.uniform(.01,.08,(1,n_assets)); qi=(np.arange(n_days)//63).reshape(-1,1)
    revenue=rev0*(1+trend)**qi*(1+rng.normal(0,.01,(n_days,n_assets))); net_income=revenue*rng.uniform(.03,.25,(1,n_assets))*(1+rng.normal(0,.03,(n_days,n_assets)))
    upper=(high-np.maximum(open_,close))/np.maximum(high-low,1e-9)
    data={'close':close,'pre_close':pre,'open':open_,'high':high,'low':low,'vwap':vwap,'ret':ret,'ret__overnight':overnight,'ret__intra':intra,
          'volume':volume,'amount':amount,'average_volume':avgvol,'turnover_ratio':turn,'circulating_cap':np.broadcast_to(circ,(n_days,n_assets)),'market_cap':mcap,
          'bid_price':bid,'ask_price':ask,'bid_size':bid_size,'ask_size':ask_size,'short_volume_ratio':short_ratio,'short_interest':short_interest,'days_to_cover':days_to_cover,
          'industry':industry,'exchange':exchange,'adj_factor':adj,'factor':factor,'fiscal_quarter':period,'revenue':revenue,'net_income':net_income,
          'high__low__ratio':high/np.maximum(low,1e-9),'upper__shadow__ratio':upper,'vwap__close__dist':vwap/np.maximum(close,1e-9)-1}
    return {k:pd.Series(np.asarray(v).ravel(),index=idx,name=k) for k,v in data.items()}


def long_dataframe(data):
    frames=[]
    for name,s in data.items():
        part=s.rename(name).reset_index()
        frames.append(part)
    out=frames[0]
    for part in frames[1:]:
        out=out.merge(part,on=["timestamp","instrument"],how="outer")
    return out
