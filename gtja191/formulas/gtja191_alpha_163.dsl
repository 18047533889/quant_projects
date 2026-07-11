# GTJA-191 Alpha 163
# source: RANK(((((-1 * RET) * MEAN(VOLUME,20)) * VWAP) * (HIGH - CLOSE)))

rank(-1 * ret * ts_mean(volume, 20) * col('vwap') * (high - close))
