# GTJA-191 Alpha 019
# source: (CLOSE<DELAY(CLOSE,5)?(CLOSE-DELAY(CLOSE,5))/DELAY(CLOSE,5):(CLOSE=DELAY(CLOSE,5)?0:(CLOSE-DELAY(CLOSE,5))/CLOSE))

where(close < ts_delay(close, 5), (close - ts_delay(close, 5)) / (ts_delay(close, 5) + 1e-08), where(close == ts_delay(close, 5), 0, (close - ts_delay(close, 5)) / (close + 1e-08)))
