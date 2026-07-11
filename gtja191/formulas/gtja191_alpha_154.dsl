# GTJA-191 Alpha 154
# source: (((VWAP - MIN(VWAP, 16))) < (CORR(VWAP, MEAN(VOLUME,180), 18)))

col('vwap') - ts_min(col('vwap'), 16) < ts_corr(col('vwap'), ts_mean(volume, 180), 18)
