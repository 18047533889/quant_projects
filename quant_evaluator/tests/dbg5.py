import sys
from pathlib import Path
_BUILD_LIB = Path(__file__).resolve().parents[1] / "build" / "lib"
print("DBG5 _BUILD_LIB:", _BUILD_LIB, "exists:", _BUILD_LIB.exists(), flush=True)
print("DBG5 in sys.path?", str(_BUILD_LIB) in sys.path, flush=True)
sys.path.insert(0, str(_BUILD_LIB))
print("DBG5 after insert path[0:4]:", sys.path[0:4], flush=True)
import quant_evaluator
print("DBG5 import resolves to:", quant_evaluator.__file__, flush=True)
try:
    from quant_evaluator.contracts.factor_batch import FactorBatch
    print("DBG5 FactorBatch OK", flush=True)
except Exception as e:
    print("DBG5 FAIL:", e, flush=True)
