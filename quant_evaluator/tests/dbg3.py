import sys
print("DBG3 quant_evaluator in sys.modules? (import attempt first)", flush=True)
try:
    import quant_evaluator
    print("DBG3 already-imported resolves to:", quant_evaluator.__file__, flush=True)
except Exception as e:
    print("DBG3 import failed:", e, flush=True)
sys.path.insert(0, "/home/shw/quant_projects/quant_evaluator/build/lib")
import quant_evaluator
print("DBG3 post-insert resolves to:", quant_evaluator.__file__, flush=True)
from quant_evaluator.contracts.factor_batch import FactorBatch
print("DBG3 FactorBatch OK", flush=True)
