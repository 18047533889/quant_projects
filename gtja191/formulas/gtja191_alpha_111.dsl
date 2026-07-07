# GTJA-191 Alpha 111
# source: SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),11,2)-SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),4,2)

EMA(volume*((close-low)-(high-close))/(high-low), 10)-EMA(volume*((close-low)-(high-close))/(high-low), 3)
