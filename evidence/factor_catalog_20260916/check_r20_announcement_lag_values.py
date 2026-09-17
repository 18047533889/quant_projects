"""Check corrected positional contract against calendar arithmetic, including missing."""
import numpy as np
import pandas as pd
from factor_engine.cleaned_operators.relation.ops import FinAnnouncementLag
period=pd.DataFrame([["2025-12-31","2026-03-31",None]])
pub=pd.DataFrame([["2026-04-15","2026-04-20","2026-04-20"]])
out=FinAnnouncementLag()._calculate_series(period,pub)
np.testing.assert_allclose(out.to_numpy(),[[105.,20.,np.nan]],equal_nan=True)
print({"calendar_lag_cases":3,"passed":3,"days":[105,20,None],"direction":"publication minus period end"})
