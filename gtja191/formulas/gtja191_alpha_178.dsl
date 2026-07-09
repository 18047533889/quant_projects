# GTJA-191 Alpha 178
# source: (CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*VOLUME

(close - ts_delay(close, 1)) / ts_delay(close, 1) * volume
