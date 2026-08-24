from shared.alphagen.data.expression import *
from shared.alphagen_qlib.stock_data import FeatureType

# GFN Task Hyperparameters
MAX_EXPR_LENGTH = 20

# GFN Model Hyperparameters
HIDDEN_DIM = 128
NUM_ENCODER_LAYERS = 2
NUM_HEADS = 4
DROPOUT = 0.1

# Training Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
NUM_EPOCHS = 100

# Action Space（与 factor_engine 白名单对齐）
from alphaprobe.fe_bridge.allowlist import CONSTANTS, DELTA_TIMES, FEATURES, OPERATORS
