# GTJA-191 Alpha 154
# source: (((VWAP - MIN(VWAP, 16))) < (CORR(VWAP, MEAN(VOLUME,180), 18)))

(((((high + low + close) / 3) - ts_min(((high + low + close) / 3), 16))) < (ts_corr(((high + low + close) / 3), ts_mean(volume,180), 18)))
