# GTJA-191 Alpha 143
# source: CLOSE>DELAY(CLOSE,1)?(CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*SELF:SELF

if_else(close > delay(close, 1), (close - delay(close, 1)) / (delay(close, 1) + 1e-8), 0)
