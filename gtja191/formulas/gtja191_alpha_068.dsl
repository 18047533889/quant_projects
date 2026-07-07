# GTJA-191 Alpha 068
# source: SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,15,2)

EMA(((high+low)/2-(delay(high,1)+delay(low,1))/2)*(high-low)/volume, 14)
