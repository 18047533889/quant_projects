# GTJA-191 Alpha 114
# source: ((RANK(DELAY(((HIGH - LOW) / (SUM(CLOSE, 5) / 5)), 2)) * RANK(RANK(VOLUME))) / (((HIGH - LOW) / (SUM(CLOSE, 5) / 5)) / (VWAP - CLOSE)))

rank(ts_delay((high - low) / (ts_sum(close, 5) / 5), 2)) * rank(rank(volume)) / ((high - low) / (ts_sum(close, 5) / 5) / (col('vwap') - close))
