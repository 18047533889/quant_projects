# GTJA-191 Alpha 161
# source: MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),12)

ts_mean(max(max((high-low),abs(delay(close,1)-high)),abs(delay(close,1)-low)),12)
