# Configuration variables that remain unchanged between training runs and those which carry over to production
import os
from pathlib import Path


def to_config_dct(cfg_class):
    dct = {k: v for k, v in cfg_class.__dict__.items() if not k.startswith("__")}
    dct["DATA_DIR"] = str(dct["DATA_DIR"])
    dct = {k: v for k, v in dct.items() if v is not None}
    return dct


class CFG:
    # Config options shared between training and inference
    DATA_DIR = Path("../data")
    NUM_WORKERS = min(32, os.cpu_count())

    # Validation split seed and training seed
    VAL_SPLIT_SEED = 623
    TRAINING_SEED = 23227

    # Bi-Encoder
    BIENC_MODEL_NAME = "bert-base-multilingual-uncased"
    TOPIC_NUM_TOKENS = 64
    CONTENT_NUM_TOKENS = 64
    SCORE_FN = "cos_sim"
    # TODO: Validate scale? Scale as optimizable parameter?
    SCORE_SCALE = 20.0

    NUM_NEIGHBORS = 400 # Used for NN model
    MAX_NUM_CANDS = 100 # Used for writing cand df
    FILTER_LANG = True

    # Cross-Encoder
    CROSS_MODEL_NAME = "bert-base-multilingual-uncased"
    CROSS_NUM_TOKENS = 128

    # Config options only needed during training
    tiny = None
    batch_size = None
    max_lr = None
    head_lr = None
    weight_decay = None
    clip = None
    token_dropout = None
    oversample = None
    margin = None
    num_epochs = None
    use_amp = None
    tune_lr = None
    tune_bs = None
    experiment_name = None
    folds = None
    num_folds = None
    output_dir = None
    pseudo_dir = None
    checkpoint = None
    checkpoint_dir = None
    cross_dropout = None
    cross_num_cands = None
    cross_corr_fname = None
    scheduler = "none"
