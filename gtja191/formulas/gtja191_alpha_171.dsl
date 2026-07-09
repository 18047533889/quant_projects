# GTJA-191 Alpha 171
# source: ((-1 * ((LOW - CLOSE) * (OPEN^5))) / ((CLOSE - HIGH) * (CLOSE^5)))

-1 * ((low - close) * power(open, 5)) / ((close - high) * power(close, 5))
