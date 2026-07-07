# GTJA-191 Alpha 019
# source: (CLOSE<DELAY(CLOSE,5)?(CLOSE-DELAY(CLOSE,5))/DELAY(CLOSE,5):(CLOSE=DELAY(CLOSE,5)?0:(CLOSE-DELAY(CLOSE,5))/CLOSE))

if_else(close < delay(close, 5), (close - delay(close, 5)) / (delay(close, 5) + 1e-8), if_else(close == delay(close, 5), 0, (close - delay(close, 5)) / (close + 1e-8)))
