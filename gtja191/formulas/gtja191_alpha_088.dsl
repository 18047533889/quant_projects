# GTJA-191 Alpha 088
# source: (CLOSE-DELAY(CLOSE,20))/DELAY(CLOSE,20)*100

(close - ts_delay(close, 20)) / ts_delay(close, 20) * 100
