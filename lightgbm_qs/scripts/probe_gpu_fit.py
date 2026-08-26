import pandas as pd, numpy as np, lightgbm as lgb
ROOT="/home/sunhaiwei/quant_projects/lightgbm_qs"
df = pd.read_parquet(f"{ROOT}/data/build/features_all.parquet")
df["date"]=pd.to_datetime(df["date"])
FEATURES=[c for c in df.columns if c not in ("date","asset")]
print("rows",len(df),"feats",len(FEATURES), flush=True)
train = df[df["date"] < "2019-01-01"]
print("train rows", len(train), flush=True)
Xt=train[FEATURES].values.astype(np.float32)
print("Xt", Xt.shape, "nan%", np.isnan(Xt).mean(), flush=True)
