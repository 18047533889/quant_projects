# GTJA-191 Alpha 063
# source: SMA(MAX(CLOSE-DELAY(CLOSE,1),0),6,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),6,1)*100

EMA(max(close-delay(close,1),0), 11)/EMA(abs(close-delay(close,1)), 11)*100
