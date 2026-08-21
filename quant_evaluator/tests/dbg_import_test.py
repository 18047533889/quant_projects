import sys
print("MODULE IMPORTED, module:", __name__, "file:", __file__)
print("sys.path[0:3]:", sys.path[0:3])
import quant_evaluator
print("quant_evaluator ->", quant_evaluator.__file__)
sys.path.insert(0, "/home/shw/quant_projects/quant_evaluator/build/lib")
print("after insert path[0]:", sys.path[0])
import quant_evaluator
print("reimport quant_evaluator ->", quant_evaluator.__file__)
try:
    from quant_evaluator.contracts.factor_batch import FactorBatch
    print("FactorBatch OK")
except Exception as e:
    print("FAIL:", e)
