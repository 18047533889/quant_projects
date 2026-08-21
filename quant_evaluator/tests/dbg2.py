import sys
print("DBG2 path[0:4]:", sys.path[0:4], flush=True)
print("DBG2 quant_evaluator in sys.modules?", "quant_evaluator" in sys.modules, flush=True)
sys.path.insert(0, "/home/shw/quant_projects/quant_evaluator/build/lib")
print("DBG2 after insert path[0:4]:", sys.path[0:4], flush=True)
import quant_evaluator
print("DBG2 import resolves to:", quant_evaluator.__file__, flush=True)
