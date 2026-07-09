# GTJA-191 Alpha 029
# source: (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*VOLUME

(close - ts_delay(close, 6)) / ts_delay(close, 6) * volume
