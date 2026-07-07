ema(if_else((high - low) > median(high - low, 20), close / open - 1, 0) * ts_max(volume, 20) / volume, 20)
