# Requested factors — DSL / code

## A. Panel hash `ext_*`（31）

### ext_38cd4f77 → `38cd4f77966c72f5`

- factor_name: `factorminer_20260720220701_7246fa3b`
- status: **ok**
- campaign: `factorminer_ashare_pv_202607202126`
- source: `/home/hsunbj/quant_projects/data/factor_pools/candidate_pool/factorminer_ashare_pv_202607202126/factorminer_20260720220701_7246fa3b/manifest.json`

**DSL/formula:**
```
rank(ts_corr(high - low, volume, 20) * ts_std(ret, 20))
```

### ext_90298bd1 → `90298bd17226d029`

- factor_name: `factorminer_20260720160523_aa583e33`
- status: **ok**
- campaign: `factorminer_ashare_pv_202607201538`
- source: `/home/hsunbj/quant_projects/data/factor_pools/candidate_pool/factorminer_ashare_pv_202607201538/factorminer_20260720160523_aa583e33/manifest.json`

**DSL/formula:**
```
rank(ts_std((close - vwap) / vwap, 10))
```

### ext_a313f83b → `a313f83be74cdbde`

- factor_name: `factorminer_20260721101800_82a030f9`
- status: **ok**
- campaign: `factorminer_ashare_pv_202607210959`
- source: `/home/hsunbj/quant_projects/data/factor_pools/candidate_pool/factorminer_ashare_pv_202607210959/factorminer_20260721101800_82a030f9/manifest.json`

**DSL/formula:**
```
rank(((close - ts_min(low, 10)) / (ts_max(high, 10) - ts_min(low, 10))) * ts_std(volume, 10))
```

### ext_cc3b5fb0 → `cc3b5fb05d6aba9b`

- factor_name: `Intraday_Overnight_Divergence_Signal`
- status: **ok**
- campaign: `all_factors_library.json`
- source: `/home/hsunbj/quant_projects/quantaalpha/data/factorlib/all_factors_library.json`

**DSL/formula:**
```
(EMA($close/$open-1,5)-TS_MEAN($open/$pre_close-1,10))/(TS_STD(EMA($close/$open-1,5)-TS_MEAN($open/$pre_close-1,10),20)+1e-8)
```


**code:**
```python
File: factor.py


import pandas as pd
import numpy as np
import os
from quantaalpha.factors.coder.expr_parser import parse_expression, parse_symbol
from quantaalpha.factors.coder.function_lib import *


def calculate_factor(expr: str, name: str):
    # stock dataframe
    df = pd.read_hdf('./daily_pv.h5', key='data')
    
    expr = parse_symbol(expr, df.columns)
    expr = parse_expression(expr)

    # replace '$var' by 'df['var'] to extract var's actual values
    for col in df.columns:
        expr = expr.replace(col[1:], f"df[\'{col}\']")

    df[name] = eval(expr)
    result = df[name].astype(np.float64)

    if os.path.exists('result.h5'):
        os.remove('result.h5')
    result.to_hdf('result.h5', key='data')

if __name__ == '__main__':
    # Input factor expression. Do NOT use the variable format like "df['$xxx']" in factor expressions. Instead, you should use "$xxx". 
    expr = "(EMA((close/open - 1), 5) - TS_MEAN((open/pre_close - 1), 10)) / (TS_STD(EMA((close/open - 1), 5) - TS_MEAN((open/pre_close - 1), 10), 20) + 1e-8)" # Your output factor expression will be filled in here
    name = "Intraday_Overnight_Divergence_Signal" # Your output factor name will be filled in here
    calculate_factor(expr, name)
```

### ext_2b3d9cff → `2b3d9cff331562ca`

- factor_name: `factorminer_20260721074433_3063df6c`
- status: **ok**
- campaign: `factorminer_ashare_pv_202607210644`
- source: `/home/hsunbj/quant_projects/data/factor_pools/candidate_pool/factorminer_ashare_pv_202607210644/factorminer_20260721074433_3063df6c/manifest.json`

**DSL/formula:**
```
rank((close - ts_min(low, 50)) / (ts_max(high, 50) + ts_std(ret, 50)))
```

### ext_1ef16bdf → `1ef16bdfdecadb4b`

- factor_name: `Overnight_Intraday_Divergence_Scaled`
- status: **ok**
- campaign: `all_factors_library.json`
- source: `/home/hsunbj/quant_projects/quantaalpha/data/factorlib/all_factors_library.json`

**DSL/formula:**
```
(open/pre_close - close/open) / (ABS(open/pre_close - 1) + ABS(close/open - 1) + 1e-8)
```


**code:**
```python
File: factor.py


import pandas as pd
import numpy as np
import os
from quantaalpha.factors.coder.expr_parser import parse_expression, parse_symbol
from quantaalpha.factors.coder.function_lib import *


def calculate_factor(expr: str, name: str):
    # stock dataframe
    df = pd.read_hdf('./daily_pv.h5', key='data')
    
    expr = parse_symbol(expr, df.columns)
    expr = parse_expression(expr)

    # replace '$var' by 'df['var'] to extract var's actual values
    for col in df.columns:
        expr = expr.replace(col[1:], f"df[\'{col}\']")

    df[name] = eval(expr)
    result = df[name].astype(np.float64)

    if os.path.exists('result.h5'):
        os.remove('result.h5')
    result.to_hdf('result.h5', key='data')

if __name__ == '__main__':
    # Input factor expression. Do NOT use the variable format like "df['$xxx']" in factor expressions. Instead, you should use "$xxx". 
    expr = "(open / pre_close - close / open) / (ABS(open / pre_close - 1) + ABS(close / open - 1) + 1e-8)" # Your output factor expression will be filled in here
    name = "Overnight_Intraday_Divergence_Scaled" # Your output factor name will be filled in here
    calculate_factor(expr, name)
```

### ext_73a60264 → `73a60264c7454707`

- factor_name: `MomentumVolume_5d_20d_Rank`
- status: **ok**
- campaign: `all_factors_library.json`
- source: `/home/hsunbj/quant_projects/quantaalpha/data/factorlib/all_factors_library.json`

**DSL/formula:**
```
RANK(TS_PCTCHANGE($close, 5) * (TS_MEAN($volume, 5) / (TS_MEAN($volume, 20) + 1e-8)))
```


**code:**
```python
File: factor.py


import pandas as pd
import numpy as np
import os
from quantaalpha.factors.coder.expr_parser import parse_expression, parse_symbol
from quantaalpha.factors.coder.function_lib import *


def calculate_factor(expr: str, name: str):
    # stock dataframe
    df = pd.read_hdf('./daily_pv.h5', key='data')
    
    expr = parse_symbol(expr, df.columns)
    expr = parse_expression(expr)

    # replace '$var' by 'df['var'] to extract var's actual values
    for col in df.columns:
        expr = expr.replace(col[1:], f"df[\'{col}\']")

    df[name] = eval(expr)
    result = df[name].astype(np.float64)

    if os.path.exists('result.h5'):
        os.remove('result.h5')
    result.to_hdf('result.h5', key='data')

if __name__ == '__main__':
    # Input factor expression. Do NOT use the variable format like "df['$xxx']" in factor expressions. Instead, you should use "$xxx". 
    expr = "RANK(TS_PCTCHANGE($close, 5) * (TS_MEAN($volume, 5) / (TS_MEAN($volume, 20) + 0.000001)))" # Your output factor expression will be filled in here
    name = "MomentumVolume_5d_20d_Rank" # Your output factor name will be filled in here
    calculate_factor(expr, name)
```

### ext_7cd58c86 → `7cd58c864b8296df`

- factor_name: `factorminer_20260701043529_bd3579f7`
- status: **ok**
- campaign: `factorminer_ashare_pv_202606301321`
- source: `/home/hsunbj/quant_projects/data/factor_pools/candidate_pool/factorminer_ashare_pv_202606301321/factorminer_20260701043529_bd3579f7/manifest.json`

**DSL/formula:**
```
ts_corr(close, volume, 20)
```

### ext_287712d7 → `287712d7d7a5bc94`

- factor_name: `factorminer_20260721041016_02024d82`
- status: **ok**
- campaign: `factorminer_ashare_pv_202607210316`
- source: `/home/hsunbj/quant_projects/data/factor_pools/candidate_pool/factorminer_ashare_pv_202607210316/factorminer_20260721041016_02024d82/manifest.json`

**DSL/formula:**
```
rank(EMA(ret, 20) / ts_std(ret, 60))
```

### ext_d6c071a1 → `d6c071a1a8d0fff7`

- factor_name: `factorminer_20260701043532_1768424b`
- status: **ok**
- campaign: `factorminer_ashare_pv_202606301321`
- source: `/home/hsunbj/quant_projects/data/factor_pools/candidate_pool/factorminer_ashare_pv_202606301321/factorminer_20260701043532_1768424b/manifest.json`

**DSL/formula:**
```
-1 * rank(ts_std(abs(close - open), 10) + (close - open))
```

### ext_b52617c5 → `b52617c554fd4f78`

- factor_name: `factorminer_20260721132615_495adc86`
- status: **ok**
- campaign: `factorminer_ashare_pv_202607211146`
- source: `/home/hsunbj/quant_projects/data/factor_pools/candidate_pool/factorminer_ashare_pv_202607211146/factorminer_20260721132615_495adc86/manifest.json`

**DSL/formula:**
```
rank((close - ts_mean(close, 40)) / (ts_max(high, 40) - ts_min(low, 40)))
```

### ext_cc11c428 → `cc11c42818516d6a`

- factor_name: `factorminer_20260721054148_0f3bb96b`
- status: **ok**
- campaign: `factorminer_ashare_pv_202607210447`
- source: `/home/hsunbj/quant_projects/data/factor_pools/candidate_pool/factorminer_ashare_pv_202607210447/factorminer_20260721054148_0f3bb96b/manifest.json`

**DSL/formula:**
```
rank(EMA(ret, 12) / ts_std(ret, 50))
```

### ext_d53e006f → `d53e006f07eb6b5f`

- factor_name: `factorminer_20260721023342_228d435a`
- status: **ok**
- campaign: `factorminer_ashare_pv_202607210151`
- source: `/home/hsunbj/quant_projects/data/factor_pools/candidate_pool/factorminer_ashare_pv_202607210151/factorminer_20260721023342_228d435a/manifest.json`

**DSL/formula:**
```
rank((close - ts_mean(close, 20)) / ts_std(close, 20))
```

### ext_909472af → `909472afa0a8458e`

- factor_name: `factorminer_20260720172802_c58572a9`
- status: **ok**
- campaign: `factorminer_ashare_pv_202607201649`
- source: `/home/hsunbj/quant_projects/data/factor_pools/candidate_pool/factorminer_ashare_pv_202607201649/factorminer_20260720172802_c58572a9/manifest.json`

**DSL/formula:**
```
rank(((close - ts_min(low, 30)) / (ts_max(high, 30) - ts_min(low, 30))) * (volume / ts_mean(volume, 30)))
```

### ext_7f827abe → `7f827abeb7c4040d`

- factor_name: `factorminer_20260720235708_b7793e00`
- status: **ok**
- campaign: `factorminer_ashare_pv_202607202244`
- source: `/home/hsunbj/quant_projects/data/factor_pools/candidate_pool/factorminer_ashare_pv_202607202244/factorminer_20260720235708_b7793e00/manifest.json`

**DSL/formula:**
```
rank((close - WMA(close, 20)) / ts_std(close, 20))
```

### ext_131e5281 → `131e5281fdc7692f`

- factor_name: `factorminer_20260716132922_4e81d647`
- status: **ok**
- campaign: `factorminer_ashare_pv_202607161305`
- source: `/home/hsunbj/quant_projects/data/factor_pools/candidate_pool/factorminer_ashare_pv_202607161305/factorminer_20260716132922_4e81d647/manifest.json`

**DSL/formula:**
```
rank((close - EMA(close, 50)) / ts_std(close, 20))
```

### ext_8d6b4d53 → `8d6b4d5359b20259`

- factor_name: `factorminer_20260721023344_050468d9`
- status: **ok**
- campaign: `factorminer_ashare_pv_202607210151`
- source: `/home/hsunbj/quant_projects/data/factor_pools/candidate_pool/factorminer_ashare_pv_202607210151/factorminer_20260721023344_050468d9/manifest.json`

**DSL/formula:**
```
rank(((close - ts_min(low, 20)) / (ts_max(high, 20) - ts_min(low, 20))) * (volume / ts_mean(volume, 20)))
```

### ext_50b318f3 → `50b318f30ed7f2c2`

- factor_name: `factorminer_20260720172805_94c607cb`
- status: **ok**
- campaign: `factorminer_ashare_pv_202607201649`
- source: `/home/hsunbj/quant_projects/data/factor_pools/candidate_pool/factorminer_ashare_pv_202607201649/factorminer_20260720172805_94c607cb/manifest.json`

**DSL/formula:**
```
rank(EMA(ret, 5) / ts_std(ret, 20))
```

### ext_e1d40349 → `e1d40349635d0f18`

- factor_name: `factorminer_20260720220757_73812eff`
- status: **ok**
- campaign: `factorminer_ashare_pv_202607202126`
- source: `/home/hsunbj/quant_projects/data/factor_pools/candidate_pool/factorminer_ashare_pv_202607202126/factorminer_20260720220757_73812eff/manifest.json`

**DSL/formula:**
```
rank(ts_delta(close, 5) / ts_std(high - low, 20))
```

### ext_bdd0fb42 → `bdd0fb4267edaec2`

- factor_name: `factorminer_20260720235702_f1d9f013`
- status: **ok**
- campaign: `factorminer_ashare_pv_202607202244`
- source: `/home/hsunbj/quant_projects/data/factor_pools/candidate_pool/factorminer_ashare_pv_202607202244/factorminer_20260720235702_f1d9f013/manifest.json`

**DSL/formula:**
```
rank((log(close) - log(ts_min(low, 50))) / (log(ts_max(high, 50)) - log(ts_min(low, 50))))
```

### ext_b512e792 → `b512e79289a878af`

- factor_name: `factorminer_20260721005313_7df8453d`
- status: **ok**
- campaign: `factorminer_ashare_pv_202607210000`
- source: `/home/hsunbj/quant_projects/data/factor_pools/candidate_pool/factorminer_ashare_pv_202607210000/factorminer_20260721005313_7df8453d/manifest.json`

**DSL/formula:**
```
rank(-(ts_delta(close, 5) / ts_std(ret, 60)))
```

### ext_52c6c08d → `52c6c08da79a1c58`

- factor_name: `factorminer_20260716001750_b648a895`
- status: **ok**
- campaign: `factorminer_ashare_pv_202607152337`
- source: `/home/hsunbj/quant_projects/data/factor_pools/candidate_pool/factorminer_ashare_pv_202607152337/factorminer_20260716001750_b648a895/manifest.json`

**DSL/formula:**
```
-rank(ts_delta(close, 5) / (ts_std(ret, 20) + 0.001))
```

### ext_760efb3b → `760efb3bed62cf5b`

- factor_name: `factorminer_20260718030655_d31daa26`
- status: **ok**
- campaign: `factorminer_ashare_pv_202607180234`
- source: `/home/hsunbj/quant_projects/data/factor_pools/candidate_pool/factorminer_ashare_pv_202607180234/factorminer_20260718030655_d31daa26/manifest.json`

**DSL/formula:**
```
rank(ts_delta(close, 5) / ts_std(ret, 20))
```

### ext_62f1f060 → `62f1f0609a60dd87`

- factor_name: `factorminer_20260701043522_c3e31ddf`
- status: **ok**
- campaign: `factorminer_ashare_pv_202606301321`
- source: `/home/hsunbj/quant_projects/data/factor_pools/candidate_pool/factorminer_ashare_pv_202606301321/factorminer_20260701043522_c3e31ddf/manifest.json`

**DSL/formula:**
```
(close + volume) / 2
```

### ext_70b9e66b → `70b9e66b9f92acd6`

- factor_name: `factorminer_20260713172720_43ececd1`
- status: **ok**
- campaign: `factorminer_ashare_pv_202607131609`
- source: `/home/hsunbj/quant_projects/data/factor_pools/candidate_pool/factorminer_ashare_pv_202607131609/factorminer_20260713172720_43ececd1/manifest.json`

**DSL/formula:**
```
rank(-(ts_delta(close, 5) / sqrt(ts_std(volume, 20))))
```

### ext_981562e9 → `981562e9e0e7d712`

- factor_name: `factorminer_20260713155001_7185a0d8`
- status: **ok**
- campaign: `factorminer_ashare_pv_202607131430`
- source: `/home/hsunbj/quant_projects/data/factor_pools/candidate_pool/factorminer_ashare_pv_202607131430/factorminer_20260713155001_7185a0d8/manifest.json`

**DSL/formula:**
```
rank((close - ts_min(low, 40)) / ((ts_max(high, 40) - ts_min(low, 40)) + ts_std(ret, 40)))
```

### ext_1490f2fe → `1490f2fef2419d80`

- factor_name: `factorminer_20260721160819_1b3352d5`
- status: **ok**
- campaign: `factorminer_ashare_pv_202607211415`
- source: `/home/hsunbj/quant_projects/data/factor_pools/candidate_pool/factorminer_ashare_pv_202607211415/factorminer_20260721160819_1b3352d5/manifest.json`

**DSL/formula:**
```
rank((close - ts_min(low, 30)) / (ts_std(high, 30) + ts_std(low, 30)))
```

### ext_773ab168 → `773ab1682d536bb0`

- factor_name: `factorminer_20260720235704_dffe1f97`
- status: **ok**
- campaign: `factorminer_ashare_pv_202607202244`
- source: `/home/hsunbj/quant_projects/data/factor_pools/candidate_pool/factorminer_ashare_pv_202607202244/factorminer_20260720235704_dffe1f97/manifest.json`

**DSL/formula:**
```
rank((close - open) / ts_std(close - open, 10))
```

### ext_b67c0953 → `b67c0953fc581983`

- factor_name: `factorminer_20260721101802_9e6df3cc`
- status: **ok**
- campaign: `factorminer_ashare_pv_202607210959`
- source: `/home/hsunbj/quant_projects/data/factor_pools/candidate_pool/factorminer_ashare_pv_202607210959/factorminer_20260721101802_9e6df3cc/manifest.json`

**DSL/formula:**
```
rank((open - close) / ts_std(ret, 10))
```

### ext_39e7b7a6 → `39e7b7a65fd0b7cb`

- factor_name: `factorminer_20260720224049_750c7a10`
- status: **ok**
- campaign: `factorminer_ashare_pv_202607202211`
- source: `/home/hsunbj/quant_projects/data/factor_pools/candidate_pool/factorminer_ashare_pv_202607202211/factorminer_20260720224049_750c7a10/manifest.json`

**DSL/formula:**
```
rank((-(close - open)) / ts_std(close, 60))
```

### ext_2fffdf67 → `2fffdf676ed5628e`

- factor_name: `factorminer_20260720144455_d9217e48`
- status: **ok**
- campaign: `factorminer_ashare_pv_202607201405`
- source: `/home/hsunbj/quant_projects/data/factor_pools/candidate_pool/factorminer_ashare_pv_202607201405/factorminer_20260720144455_d9217e48/manifest.json`

**DSL/formula:**
```
rank((close - ts_min(low, 20)) / ((ts_max(high, 20) - ts_min(low, 20)) + ts_std(ret, 20)))
```


## B. Named `cand_*`（14）

### cand_vw_ew_spread_responsive → `factor_vw_ew_spread_responsive`

- found_name: `factor_vw_ew_spread_responsive`
- status: **ok**
- source: `/home/hsunbj/quant_projects/workspaces/zhangjiayin/admitted_pool/ashare_cogalpha/factors/627cb56ee6214186/metadata.json`

- lineage id (非 DSL): `g1_mutation_pv_coherence_001_826d5d95`


**code:**
```python
def factor_vw_ew_spread_responsive(df):
    """Volume-weighted vs equal-weighted daily return spread, using a responsive exponential smoothing (alpha=0.1) without volatility scaling."""
    df_copy = df.copy()
    ret = df_copy["close"].pct_change()
    vol = df_copy["volume"]
    vw_ret = (ret * vol).ewm(alpha=0.1, adjust=False).mean() / vol.ewm(alpha=0.1, adjust=False).mean()
    ew_ret = ret.ewm(alpha=0.1, adjust=False).mean()
    spread = vw_ret - ew_ret
    df_copy["factor_vw_ew_spread_responsive"] = spread
    return df_copy["factor_vw_ew_spread_responsive"]
```

### cand_overnight_intraday_divergence_simplified → `factor_overnight_intraday_divergence_simplified`

- found_name: `factor_overnight_intraday_divergence_simplified`
- status: **ok**
- source: `/home/hsunbj/quant_projects/workspaces/zhangjiayin/admitted_pool/ashare_cogalpha/factors/3521b00162459637/metadata.json`

- lineage id (非 DSL): `g2_mutation_cand_oi_001_82a6b93e`


**code:**
```python
def factor_overnight_intraday_divergence_simplified(df):
    df_copy = df.copy()
    overnight_ret, intraday_ret = alpha_tools.decompose_overnight_intraday(df_copy["close"], df_copy["open"])
    ema_overnight = overnight_ret.ewm(span=5, min_periods=5).mean()
    ema_intraday = intraday_ret.ewm(span=5, min_periods=5).mean()
    factor = ema_intraday - ema_overnight
    df_copy["factor_overnight_intraday_divergence_simplified"] = factor.fillna(0)
    return df_copy["factor_overnight_intraday_divergence_simplified"]
```

### cand_fusion_upcap_volregime → `factor_fusion_upcap_volregime`

- found_name: `factor_fusion_upcap_volregime`
- status: **ok**
- source: `/home/hsunbj/quant_projects/workspaces/zhangjiayin/admitted_pool/ashare_cogalpha/factors/9bd4189d923b7bcd/metadata.json`

- lineage id (非 DSL): `alpha_composite_fusion_upcap_volregime_001`


**code:**
```python
def factor_fusion_upcap_volregime(df):
    df_copy = df.copy()
    ret = df_copy["close"] / df_copy["close"].shift(1) - 1
    up_vol = df_copy["volume"] * (ret > 0)
    sum_up_vol = up_vol.rolling(20, min_periods=20).sum()
    sum_vol = df_copy["volume"].rolling(20, min_periods=20).sum()
    up_capture = sum_up_vol / sum_vol.replace(0, np.nan)
    _, _, vol_ratio = alpha_tools.classify_volume_regime(df_copy["volume"], window=20)
    factor = up_capture * (1 + (vol_ratio - 1) * 0.5)
    df_copy["factor_fusion_upcap_volregime"] = factor.fillna(0)
    return df_copy["factor_fusion_upcap_volregime"]
```

### cand_volume_confirmed_intraday_momentum → `factor_volume_confirmed_intraday_momentum`

- found_name: `factor_volume_confirmed_intraday_momentum`
- status: **ok**
- source: `/home/hsunbj/quant_projects/workspaces/zhangjiayin/admitted_pool/ashare_cogalpha/factors/c63f00bca8822e7b/metadata.json`

**DSL/formula:**
```
rollmean10(intraday_return * ((volume - rollmean20(volume)) / rollstd20(volume)))
```


**code:**
```python
def factor_volume_confirmed_intraday_momentum(df):
    """Intraday return weighted by volume anomaly (z-score) to capture volume-confirmed moves."""
    df_copy = df.copy()
    intraday_ret = (df_copy["close"] - df_copy["open"]) / df_copy["open"].replace(0, np.nan)
    vol_mean = df_copy["volume"].rolling(20, min_periods=1).mean()
    vol_std = df_copy["volume"].rolling(20, min_periods=1).std(ddof=1)
    vol_zscore = (df_copy["volume"] - vol_mean) / vol_std.replace(0, np.nan)
    raw = intraday_ret * vol_zscore
    signal = raw.rolling(10, min_periods=1).mean()
    df_copy["factor_volume_confirmed_intraday_momentum"] = signal
    return df_copy["factor_volume_confirmed_intraday_momentum"]
```

### cand_vol_asymmetry_diff_blend → `factor_vol_asymmetry_diff_blend`

- found_name: `factor_vol_asymmetry_diff_blend`
- status: **ok**
- source: `/home/hsunbj/quant_projects/workspaces/zhangjiayin/admitted_pool/ashare_cogalpha/factors/98f0ec034ad05b66/metadata.json`

**DSL/formula:**
```
asym_short = down_std(5) - up_std(5); asym_long = down_std(20) - up_std(20); vol_rank = rank_63(ATR(14)/close); factor = vol_rank * asym_short + (1 - vol_rank) * asym_long
```


**code:**
```python
def factor_vol_asymmetry_diff_blend(df):
    df_copy = df.copy()
    close = df_copy["close"]
    high = df_copy["high"]
    low = df_copy["low"]
    ret = close.pct_change()

    # manual ATR(14) for volatility regime gate
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    vol_ratio = atr / close.replace(0.0, np.nan)
    vol_rank = vol_ratio.rolling(63, min_periods=21).rank(pct=True)

    # short-term (5-day) volatility asymmetry difference
    short_win = 5
    min_periods_short = int(short_win/2)
    is_up = (ret > 0).astype(float)
    is_down = (ret < 0).astype(float)
    sq_ret = ret ** 2
    sum_up_sq_5 = (sq_ret * is_up).rolling(short_win, min_periods=min_periods_short).sum()
    cnt_up_5 = is_up.rolling(short_win, min_periods=min_periods_short).sum()
    sum_down_sq_5 = (sq_ret * is_down).rolling(short_win, min_periods=min_periods_short).sum()
    cnt_down_5 = is_down.rolling(short_win, min_periods=min_periods_short).sum()
    up_std_5 = np.sqrt(sum_up_sq_5 / cnt_up_5.replace(0, np.nan))
    down_std_5 = np.sqrt(sum_down_sq_5 / cnt_down_5.replace(0, np.nan))
    asym_short = down_std_5 - up_std_5

    # long-term (20-day) volatility asymmetry difference
    long_win = 20
    min_periods_long = int(long_win/2)
    sum_up_sq_20 = (sq_ret * is_up).rolling(long_win, min_periods=min_periods_long).sum()
    cnt_up_20 = is_up.rolling(long_win, min_periods=min_periods_long).sum()
    sum_down_sq_20 = (sq_ret * is_down).rolling(long_win, min_periods=min_periods_long).sum()
    cnt_down_20 = is_down.rolling(long_win, min_periods=min_periods_long).sum()
    up_std_20 = np.sqrt(sum_up_sq_20 / cnt_up_20.replace(0, np.nan))
    down_std_20 = np.sqrt(sum_down_sq_20 / cnt_down_20.replace(0, np.nan))
    asym_long = down_std_20 - up_std_20

    # blend using volatility regime percentile
    factor = vol_rank * asym_short + (1.0 - vol_rank) * asym_long
    df_copy["factor_vol_asymmetry_diff_blend"] = factor
    return df_copy["factor_vol_asymmetry_diff_blend"]
```

### cand_liquidity_spread_vol_continuous_median → `factor_liquidity_spread_vol_continuous_median`

- found_name: `factor_liquidity_spread_vol_continuous_median`
- status: **ok**
- source: `/home/hsunbj/quant_projects/workspaces/zhangjiayin/admitted_pool/ashare_cogalpha/factors/c6e6695e50a65f8c/metadata.json`

**DSL/formula:**
```
smoothed_median(intraday_ret - overnight_ret) * clip(vol_ratio, 0.5, 2.0)
```


**code:**
```python
def factor_liquidity_spread_vol_continuous_median(df):
    """Overnight-intraday return spread modulated by continuous volume ratio, smoothed with median."""
    df_copy = df.copy()
    overnight_ret, intraday_ret = alpha_tools.decompose_overnight_intraday(df_copy["close"], df_copy["open"])
    spread = intraday_ret - overnight_ret
    spread_smooth = spread.rolling(5, min_periods=5).median()
    _, _, vol_ratio = alpha_tools.classify_volume_regime(df_copy["volume"], window=20)
    vol_multiplier = np.clip(vol_ratio, 0.5, 2.0)
    factor = spread_smooth * vol_multiplier
    df_copy["factor_liquidity_spread_vol_continuous_median"] = factor
    return df_copy["factor_liquidity_spread_vol_continuous_median"]
```

### cand_gap_reversal_intensity → `factor_gap_reversal_intensity`

- found_name: `factor_gap_reversal_intensity`
- status: **ok**
- source: `/home/hsunbj/quant_projects/workspaces/zhangjiayin/admitted_pool/ashare_cogalpha/factors/b6d320b1a76d319f/metadata.json`

**code:**
```python
def factor_gap_reversal_intensity(df):
    """Intensity of gap reversal: magnified product of overnight drop and intraday recovery indicates fragile market structure."""
    df_copy = df.copy()
    prev_close = df_copy["close"].shift(1)
    overnight_ret = (df_copy["open"] - prev_close) / prev_close.replace(0.0, np.nan)
    intraday_ret = (df_copy["close"] - df_copy["open"]) / df_copy["open"].replace(0.0, np.nan)
    reversal_score = -np.minimum(overnight_ret, 0) * np.maximum(intraday_ret, 0)
    factor = reversal_score.rolling(5, min_periods=5).sum()
    df_copy["factor_gap_reversal_intensity"] = factor
    return df_copy["factor_gap_reversal_intensity"]
```

### cand_herding_pv_sync_sma → `factor_herding_pv_sync_sma`

- found_name: `factor_herding_pv_sync_sma`
- status: **ok**
- source: `/home/hsunbj/quant_projects/workspaces/zhangjiayin/admitted_pool/ashare_cogalpha/factors/055c3f5b009618c5/metadata.json`

**DSL/formula:**
```
sma(sign(return) == sign(volume_change) with 1/-1 encoding, window=20)
```


**code:**
```python
def factor_herding_pv_sync_sma(df):
    df_copy = df.copy()
    ret = df_copy["close"].pct_change()
    vol_chg = df_copy["volume"].pct_change()
    sign_ret = np.sign(ret).fillna(0)
    sign_vol = np.sign(vol_chg).fillna(0)
    both_nonzero = (sign_ret != 0) & (sign_vol != 0)
    same = sign_ret == sign_vol
    sync = np.where(both_nonzero, np.where(same, 1.0, -1.0), 0.0)
    smooth = pd.Series(sync, index=df_copy.index).rolling(window=20, min_periods=20).mean()
    df_copy["factor_herding_pv_sync_sma"] = smooth
    return df_copy["factor_herding_pv_sync_sma"]
```

### cand_gpdev_volregime_ema10 → `factor_gpdev_volregime_ema10`

- found_name: `factor_gpdev_volregime_ema10`
- status: **ok**
- source: `/home/hsunbj/quant_projects/workspaces/zhangjiayin/admitted_pool/ashare_cogalpha/factors/186f02bef23c40b7/metadata.json`

- lineage id (非 DSL): `g2_crossover_then_mutation_g1_mutation_agentcreative_g0_c2_8fd45102_g1_cros_7acef99a`


**code:**
```python
def factor_gpdev_volregime_ema10(df):
    """Geometric mean deviation with volume regime confirmation and longer EMA smoothing."""
    df_copy = df.copy()
    gp_mean = np.sqrt(df_copy["high"] * df_copy["low"])
    deviation = df_copy["close"] / gp_mean.replace(0, np.nan) - 1.0
    _, _, vol_ratio = alpha_tools.classify_volume_regime(df_copy["volume"], window=20)
    vol_strength = vol_ratio - 1.0
    raw = deviation * vol_strength
    df_copy["factor_gpdev_volregime_ema10"] = raw.ewm(span=10, min_periods=1).mean()
    return df_copy["factor_gpdev_volregime_ema10"]
```

### cand_gap_vol_cluster_adaptive_w30 → `factor_gap_vol_cluster_adaptive_w30`

- found_name: `factor_gap_vol_cluster_adaptive_w30`
- status: **ok**
- source: `/home/hsunbj/quant_projects/workspaces/zhangjiayin/admitted_pool/ashare_cogalpha/factors/63291f4803195f44/metadata.json`

- lineage id (非 DSL): `g1_crossover_then_mutation_alpha_creative_0_001_vol_clustering_001_63939995`


**code:**
```python
def factor_gap_vol_cluster_adaptive_w30(df):
    df_copy = df.copy()
    overnight_ret, intraday_ret = alpha_tools.decompose_overnight_intraday(df_copy['close'], df_copy['open'])
    diff = intraday_ret - overnight_ret
    _, _, vol_ratio = alpha_tools.classify_volume_regime(df_copy['volume'], window=20)
    primary = np.tanh(diff * vol_ratio * 10.0)
    ret = df_copy['close'].pct_change()
    sq = ret ** 2
    acorr = sq.rolling(30, min_periods=10).corr(sq.shift(1))
    cluster = -acorr
    weight = 1.0 + 0.5 * np.tanh(cluster * 2.0)
    signal = primary * weight
    df_copy['factor_gap_vol_cluster_adaptive_w30'] = signal
    return df_copy['factor_gap_vol_cluster_adaptive_w30']
```

### cand_vol_persistence_momentum → `factor_vol_persistence_momentum`

- found_name: `factor_vol_persistence_momentum`
- status: **ok**
- source: `/home/hsunbj/quant_projects/workspaces/zhangjiayin/admitted_pool/ashare_cogalpha/factors/3b2917c210a1e270/metadata.json`

**DSL/formula:**
```
ret_5d * (autocorr(sq_ret, 20) - 0.5)
```


**code:**
```python
def factor_vol_persistence_momentum(df):
    df_copy = df.copy()
    ret = df_copy['close'].pct_change()
    sq_ret = ret**2
    vol_autocorr = sq_ret.rolling(20).corr(sq_ret.shift(1))
    ret_5d = df_copy['close'].pct_change(5)
    signal = ret_5d * (vol_autocorr - 0.5)
    signal = signal.replace([np.inf, -np.inf], np.nan).fillna(0)
    signal.name = 'factor_vol_persistence_momentum'
    return signal
```

### cand_reversal_volregime_overnight_csz → `factor_reversal_volregime_overnight_csz`

- found_name: `factor_reversal_volregime_overnight_csz`
- status: **ok**
- source: `/home/hsunbj/quant_projects/workspaces/zhangjiayin/admitted_pool/ashare_cogalpha/factors/2ee55f2c675ca2fc/metadata.json`

- lineage id (非 DSL): `g2_crossover_g1_mutation_volregime_001_08092d2b_g1_crossover_0e5b8117`


**code:**
```python
def factor_reversal_volregime_overnight_csz(df):
    """Mean-reversion in expanding volatility regimes, attenuated by overnight gap magnitude."""
    df_copy = df.copy()
    high = df_copy["high"]
    low = df_copy["low"]
    close = df_copy["close"]
    open_ = df_copy["open"]
    prev_close = close.shift(1)
    tr = np.maximum(high - low, np.maximum(abs(high - prev_close), abs(low - prev_close)))
    short_atr = tr.rolling(5, min_periods=5).mean()
    long_atr = tr.rolling(20, min_periods=20).mean()
    atr_ratio = short_atr / long_atr.replace(0, np.nan)
    ret_3d = close.pct_change(3)
    raw = -ret_3d * atr_ratio
    overnight_ret, _ = alpha_tools.decompose_overnight_intraday(close, open_)
    overnight_std = overnight_ret.rolling(20, min_periods=20).std(ddof=1)
    overnight_z = np.abs(overnight_ret) / overnight_std.replace(0, np.nan)
    overnight_z = overnight_z.fillna(0.0)
    attenuation = 1.0 / (1.0 + overnight_z)
    factor = raw * attenuation
    df_copy["factor_reversal_volregime_overnight_csz"] = factor
    return df_copy["factor_reversal_volregime_overnight_csz"]
```

### cand_vol_expansion_atr_ratio → `factor_vol_expansion_atr_ratio`

- found_name: `factor_vol_expansion_atr_ratio`
- status: **ok**
- source: `/home/hsunbj/quant_projects/workspaces/zhangjiayin/admitted_pool/ashare_cogalpha/factors/5b5fcfa6bcd83184/metadata.json`

- lineage id (非 DSL): `g2_mutation_g1_crossover_agentrangevol_2_agentrangevol_1_1d0_b85a540a`


**code:**
```python
def factor_vol_expansion_atr_ratio(df):
    """Volatility expansion: short-term ATR over long-term ATR, sigmoid and direction."""
    df_copy = df.copy()
    high = df_copy["high"].values
    low = df_copy["low"].values
    close = df_copy["close"].values
    atr5 = pd.Series(talib.ATR(high, low, close, timeperiod=5), index=df_copy.index)
    atr60 = pd.Series(talib.ATR(high, low, close, timeperiod=60), index=df_copy.index)
    ratio = atr5 / atr60.replace(0.0, np.nan)
    sig = 1.0 / (1.0 + np.exp(-5.0 * (ratio - 1.0)))
    direction = np.sign(df_copy["close"] - df_copy["open"])
    df_copy["factor_vol_expansion_atr_ratio"] = direction * sig
    return df_copy["factor_vol_expansion_atr_ratio"]
```

### cand_geometric_deviation_vol_asym_tilt → `factor_geometric_deviation_vol_asym_tilt`

- found_name: `factor_geometric_deviation_vol_asym_tilt`
- status: **ok**
- source: `/home/hsunbj/quant_projects/workspaces/zhangjiayin/admitted_pool/ashare_cogalpha/factors/d33d0a88a5d76578/metadata.json`

- lineage id (非 DSL): `g2_crossover_vol_asym_02_repaired_g0_creative_001_6af16078`


**code:**
```python
def factor_geometric_deviation_vol_asym_tilt(df):
    """Nonlinear deviation from geometric mean of H, L, C, tilted by upside volatility asymmetry."""
    df_copy = df.copy()
    # Primary mechanism: geometric deviation
    log_h = np.log(df_copy["high"].clip(lower=1e-12))
    log_l = np.log(df_copy["low"].clip(lower=1e-12))
    log_c = np.log(df_copy["close"].clip(lower=1e-12))
    dev = (2 * log_c - log_h - log_l) / 3.0
    raw = np.sign(dev) * dev**2
    geo_signal = raw.ewm(span=5, min_periods=5).mean()

    # Lightweight modifier: upside volatility asymmetry tilt
    daily_ret = df_copy["close"].pct_change()
    up_vol = daily_ret.clip(lower=0).rolling(20, min_periods=20).std(ddof=1)
    down_vol = daily_ret.clip(upper=0).rolling(20, min_periods=20).std(ddof=1)
    vol_total = up_vol + down_vol
    asym_diff = (up_vol - down_vol) / vol_total.replace(0, np.nan)
    tilt = np.maximum(asym_diff, 0)  # only amplify when up_vol >= down_vol

    # Apply tilt: amplify the geometric signal during upside-dominant regimes
    factor = geo_signal * (1 + tilt)
    df_copy["factor_geometric_deviation_vol_asym_tilt"] = factor.replace([np.inf, -np.inf], np.nan)
    return df_copy["factor_geometric_deviation_vol_asym_tilt"]
```
