from shared.alphagen.data.expression import Feature, Ref
from shared.alphagen_qlib.stock_data import FeatureType


high = High = HIGH = Feature(FeatureType.HIGH)
low = Low = LOW = Feature(FeatureType.LOW)
volume = Volume = VOLUME = Feature(FeatureType.VOLUME)
open_ = Open = OPEN = Feature(FeatureType.OPEN)
close = Close = CLOSE = Feature(FeatureType.CLOSE)
vwap = Vwap = VWAP = Feature(FeatureType.VWAP)
# 默认标签：vwap→vwap（与 alphaprobe runner fitness 一致）
target = Ref(vwap, -20) / vwap - 1