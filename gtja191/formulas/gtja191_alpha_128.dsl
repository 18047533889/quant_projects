# GTJA-191 Alpha 128
# source: 100-(100/(1+SUM(((HIGH+LOW+CLOSE)/3>DELAY((HIGH+LOW+CLOSE)/3,1)?(HIGH+LOW+CLOSE)/3*VOLUME:0),14)/SUM(((HIGH+LOW+CLOSE)/3<DELAY((HIGH+LOW+CLOSE)/3,1)?(HIGH+LOW+CLOSE)/3*VOLUME:0),14)))

100 - 100 / (1 + ts_sum(where(col('vwap') > ts_delay(col('vwap'), 1), col('vwap') * volume, 0), 14) / (ts_sum(where(col('vwap') < ts_delay(col('vwap'), 1), col('vwap') * volume, 0), 14) + 1e-08))
