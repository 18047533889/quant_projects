# GTJA-191 Alpha 013
# source: (((HIGH * LOW)^0.5) - VWAP)

power(high * low, 0.5) - col('vwap')
