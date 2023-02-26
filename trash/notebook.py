from config import CFG
from submit import main

CFG.NUM_NEIGHBORS = 44
CFG.NUM_WORKERS = 2
CFG.cross_dropout = 0.0

CFG.tiny = True

# DATA_DIR = "/kaggle/input/learning-equality-curriculum-recommendations"
# TOKENIZER_DIR = "/kaggle/input/kolibri-model/tokenizer"
# BIENC_DIR = "/kaggle/input/kolibri-model/bienc"
# CROSS_DIR = "/kaggle/input/kolibri-cross/cross"

DATA_DIR = "../../data"
TOKENIZER_DIR = "../out/distiluse-base-multilingual-cased-v2_KLB-83/tokenizer/"
BIENC_DIR = "../out/distiluse-base-multilingual-cased-v2_KLB-83/bienc/"
CROSS_DIR = "../cout/distiluse-base-multilingual-cased-v2_KLB-95/cross/"

submission_df = main(0.1,
                     DATA_DIR, TOKENIZER_DIR, BIENC_DIR, CROSS_DIR,
                     32, 32)

submission_df.to_csv("submission.csv", index=False)
