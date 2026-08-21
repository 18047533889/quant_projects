import sys
print("DBG4 path[0:4]:", sys.path[0:4], flush=True)
sys.path.insert(0, "/home/shw/quant_projects/quant_evaluator/build/lib")
import quant_evaluator
print("DBG4 import resolves to:", quant_evaluator.__file__, flush=True)
from quant_evaluator.contracts.factor_batch import FactorBatch
print("DBG4 FactorBatch OK", flush=True)
