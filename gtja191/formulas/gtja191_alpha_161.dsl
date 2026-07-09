# GTJA-191 Alpha 161
# source: MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),12)

ts_mean(flex_max(flex_max(high - low, abs(ts_delay(close, 1) - high)), abs(ts_delay(close, 1) - low)), 12)
