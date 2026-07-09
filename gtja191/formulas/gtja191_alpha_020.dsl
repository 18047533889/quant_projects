# GTJA-191 Alpha 020
# source: (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100

(close - ts_delay(close, 6)) / ts_delay(close, 6) * 100
