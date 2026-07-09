# GTJA-191 Alpha 134
# source: (CLOSE-DELAY(CLOSE,12))/DELAY(CLOSE,12)*VOLUME

(close - ts_delay(close, 12)) / ts_delay(close, 12) * volume
