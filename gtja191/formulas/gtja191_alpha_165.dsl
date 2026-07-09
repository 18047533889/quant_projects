# GTJA-191 Alpha 165
# source: MAX(SUMAC(CLOSE-MEAN(CLOSE,48)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,48)))/STD(CLOSE,48)

(ts_max(ts_sum(close - ts_mean(close, 48), 48), 48) - ts_min(ts_sum(close - ts_mean(close, 48), 48), 48)) / (ts_std(close, 48) + 1e-08)
