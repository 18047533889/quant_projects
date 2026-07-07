# GTJA-191 Alpha 086
# source: ((0.25 < (((DELAY(CLOSE, 20) - DELAY(CLOSE, 10)) / 10) - ((DELAY(CLOSE, 10) - CLOSE) / 10))) ? (-1 * 1) : (((((DELAY(CLOSE, 20) - DELAY(CLOSE, 10)) / 10) - ((DELAY(CLOSE, 10) - CLOSE) / 10)) < 0) ? 1 : ((-1 * 1) * (CLOSE - DELAY(CLOSE, 1)))))

if_else(0.25 < (delay(close, 20) - delay(close, 10)) / 10 - (delay(close, 10) - close) / 10, -1, if_else((delay(close, 20) - delay(close, 10)) / 10 - (delay(close, 10) - close) / 10 < 0, 1, -1 * (close - delay(close, 1))))
