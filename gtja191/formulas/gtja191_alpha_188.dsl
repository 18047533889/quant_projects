# GTJA-191 Alpha 188
# source: ((HIGH-LOW–SMA(HIGH-LOW,11,2))/SMA(HIGH-LOW,11,2))*100

(high - low - ts_ema(high - low, 10)) / ts_ema(high - low, 10) * 100
