# import sys
# sys.path.insert(0, "/kaggle/input/kolibri-code")
from config import CFG
from submit import main

CFG.NUM_NEIGHBORS = 50
CFG.NUM_WORKERS = 16
CFG.cross_dropout = 0.0
CLASSIFIER_THRESHOLD = 0.06

# DATA_DIR = "/kaggle/input/learning-equality-curriculum-recommendations"
# BIENC_TOKENIZER_DIR = "/kaggle/input/kolibri-model/tokenizer"
# CROSS_TOKENIZER_DIR = BIENC_TOKENIZER_DIR
# BIENC_DIR = "/kaggle/input/kolibri-model/bienc"
# CROSS_DIR = "/kaggle/input/kolibri-cross/cross"

DATA_DIR = "../dbg-data"
BIENC_TOKENIZER_DIR = "../kaggle/model/tokenizer"
CROSS_TOKENIZER_DIR = BIENC_TOKENIZER_DIR
BIENC_DIR = "../kaggle/model/bienc"
CROSS_DIR = "../kaggle/cross/cross"

submission_df = main(CLASSIFIER_THRESHOLD,
                     DATA_DIR,
                     BIENC_TOKENIZER_DIR, BIENC_DIR,
                     CROSS_TOKENIZER_DIR, CROSS_DIR,
                     filter_lang=True,
                     bienc_batch_size=128, cross_batch_size=64)

submission_df.to_csv("submission.csv", index=False)
