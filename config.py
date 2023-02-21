# Configuration variables that remain unchanged between training runs and those which carry over to production
from pathlib import Path

from bienc.losses import dot_score, cos_sim


def to_config_dct(cfg_class):
    dct = {k: v for k, v in cfg_class.__dict__.items() if not k.startswith("__")}
    dct["DATA_DIR"] = str(dct["DATA_DIR"])
    dct["SCORE_FN"] = dct["SCORE_FN"].__name__
    dct = {k: v for k, v in dct.items() if v is not None}
    return dct


# class GLB:
#     device = None


class CFG:
    # Config options shared between training and inference
    DATA_DIR = Path("../data")

    # Validation set size and training seed
    VAL_SPLIT_SEED = 623

    # Bi-Encoder
    BIENC_MODEL_NAME = "bert-base-multilingual-uncased"

    TOPIC_NUM_TOKENS = 128
    CONTENT_NUM_TOKENS = 128

    SCORE_FN = cos_sim

    NUM_WORKERS = 24

    NUM_NEIGHBORS = 100

    # Cross-Encoder
    CROSS_MODEL_NAME = "bert-base-multilingual-uncased"
    CROSS_CORR_FNAME = "cross_corr.csv"
    CROSS_NUM_TOKENS = 256

    # Config options only needed during training
    tiny = None
    batch_size = None
    max_lr = None
    weight_decay = None
    margin = None
    num_epochs = None
    use_amp = None
    experiment_name = None
    folds = None
    output_dir = None
