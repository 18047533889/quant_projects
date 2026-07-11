# GTJA-191 Alpha 045
# source: (RANK(DELTA((((CLOSE * 0.6) + (OPEN *0.4))), 1)) * RANK(CORR(VWAP, MEAN(VOLUME,150), 15)))

rank(ts_delta(close * 0.6 + open * 0.4, 1)) * rank(ts_corr(col('vwap'), ts_mean(volume, 150), 15))
