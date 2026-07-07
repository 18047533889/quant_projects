# GTJA-191 Alpha 108
# source: ((RANK((HIGH - MIN(HIGH, 2)))^RANK(CORR((VWAP), (MEAN(VOLUME,120)), 6))) * -1)

power(rank(high - ts_min(high, 2)), rank(ts_corr(((high + low + close) / 3), ts_mean(volume, 120), 6))) * -1
