# GTJA-191 Alpha 111
# source: SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),11,2)-SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),4,2)

ts_ema(volume * (close - low - (high - close)) / (high - low), 10) - ts_ema(volume * (close - low - (high - close)) / (high - low), 3)
