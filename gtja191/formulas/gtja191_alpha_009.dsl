# GTJA-191 Alpha 009
# source: SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,7,2)

ts_ema(((high + low) / 2 - (ts_delay(high, 1) + ts_delay(low, 1)) / 2) * (high - low) / volume, 6)
