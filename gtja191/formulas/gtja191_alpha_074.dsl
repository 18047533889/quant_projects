# GTJA-191 Alpha 074
# source: (RANK(CORR(SUM(((LOW * 0.35) + (VWAP * 0.65)), 20), SUM(MEAN(VOLUME,40), 20), 7)) + RANK(CORR(RANK(VWAP), RANK(VOLUME), 6)))

rank(ts_corr(ts_sum(low * 0.35 + col('vwap') * 0.65, 20), ts_sum(ts_mean(volume, 40), 20), 7)) + rank(ts_corr(rank(col('vwap')), rank(volume), 6))
