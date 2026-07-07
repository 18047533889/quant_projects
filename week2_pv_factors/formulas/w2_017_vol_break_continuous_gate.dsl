ema((divide(close - low, high - low) - 0.5) * divide(median(high - low, 20), high - low), 5)
