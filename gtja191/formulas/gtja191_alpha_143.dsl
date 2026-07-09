# GTJA-191 Alpha 143
# source: CLOSE>DELAY(CLOSE,1)?(CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*SELF:SELF

where(close > ts_delay(close, 1), (close - ts_delay(close, 1)) / (ts_delay(close, 1) + 1e-08), 0)
