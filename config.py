# Configuration variables that remain unchanged between training runs and those which carry over to production
from pathlib import Path

from model.losses import dot_score, cos_sim

DATA_DIR = Path("../data")

# Validation set size and training seed
VAL_SPLIT_SEED = 623

# Bi-Encoder
BIENC_MODEL_NAME = "bert-base-multilingual-uncased"

TOPIC_NUM_TOKENS = 128
CONTENT_NUM_TOKENS = 128

SCORE_FN = cos_sim
