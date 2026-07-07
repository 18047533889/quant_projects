# GTJA-191 Alpha 112
# source: (SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)-SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))/(SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)+SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))*100

(ts_sum(if_else(close - delay(close, 1) > 0, close - delay(close, 1), 0), 12) - ts_sum(if_else(close - delay(close, 1) < 0, abs(close - delay(close, 1)), 0), 12)) / (ts_sum(if_else(close - delay(close, 1) > 0, close - delay(close, 1), 0), 12) + ts_sum(if_else(close - delay(close, 1) < 0, abs(close - delay(close, 1)), 0), 12) + 1e-8) * 100
