# GTJA-191 Alpha 158
# source: ((HIGH-SMA(CLOSE,15,2))-(LOW-SMA(CLOSE,15,2)))/CLOSE

(high - ts_ema(close, 14) - (low - ts_ema(close, 14))) / close
