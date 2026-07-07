# GTJA-191 Alpha 102
# source: SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1)/SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1)*100

EMA(max(volume-delay(volume,1),0), 11)/EMA(abs(volume-delay(volume,1)), 11)*100
