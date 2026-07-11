# GTJA-191 Alpha 170
# source: ((((RANK((1 / CLOSE)) * VOLUME) / MEAN(VOLUME,20)) * ((HIGH * RANK((HIGH - CLOSE))) / (SUM(HIGH, 5) / 5))) - RANK((VWAP - DELAY(VWAP, 5))))

rank(1 / close) * volume / ts_mean(volume, 20) * (high * rank(high - close) / (ts_sum(high, 5) / 5)) - rank(col('vwap') - ts_delay(col('vwap'), 5))
