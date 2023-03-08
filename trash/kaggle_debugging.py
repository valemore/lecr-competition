import sys
sys.path.insert(0, "/kaggle/input/kolibri-code")
from config import CFG
from submit import main

CFG.NUM_NEIGHBORS = 50
CFG.NUM_WORKERS = 2
CFG.cross_dropout = 0.0


topics_df = pd.read_csv("/kaggle/input/learning-equality-curriculum-recommendations/topics.csv", keep_default_na=False)
content_df = pd.read_csv("/kaggle/input/learning-equality-curriculum-recommendations/content.csv", keep_default_na=False).sample(100)
input_df = pd.read_csv("/kaggle/input/learning-equality-curriculum-recommendations/sample_submission.csv", keep_default_na=False)

content_df.to_csv("/kaggle/working/content.csv", index=False)
topics_df.to_csv("/kaggle/working/topics.csv", index=False)
sub_df.to_csv("/kaggle/working/sample_submission.csv", index=False)

DATA_DIR = "/kaggle/input/learning-equality-curriculum-recommendations"
DATA_DIR = "/kaggle/working/"
BIENC_TOKENIZER_DIR = "/kaggle/input/kolibri-model/tokenizer"
CROSS_TOKENIZER_DIR = BIENC_TOKENIZER_DIR
BIENC_DIR = "/kaggle/input/kolibri-model/bienc"
CROSS_DIR = "/kaggle/input/kolibri-cross/cross"

submission_df = main(0.06,
                     DATA_DIR,
                     BIENC_TOKENIZER_DIR, BIENC_DIR,
                     CROSS_TOKENIZER_DIR, CROSS_DIR,
                     filter_lang=True,
                     bienc_batch_size=128, cross_batch_size=64)

submission_df.to_csv("submission.csv", index=False)


# import pandas as pd
# from submit import *

# from pathlib import Path
# DATA_DIR = Path(DATA_DIR)

# topics_df = pd.read_csv(DATA_DIR / "topics.csv", keep_default_na=False)
# content_df = pd.read_csv(DATA_DIR / "content.csv", keep_default_na=False).sample(100)
# input_df = pd.read_csv(DATA_DIR / "sample_submission.csv", keep_default_na=False)

# content_df.to_csv("content.csv", index=False)
# topics_df.to_csv("content.csv", index=False)
# sub_df.to_csv("content.csv", index=False)

# data_dir = Path(".")
# bienc_tokenizer_dir = BIENC_TOKENIZER_DIR
# cross_tokenizer_dir = CROSS_TOKENIZER_DIR
# bienc_dir = BIENC_DIR
# cross_dir = CROSS_DIR
# bienc_batch_size, cross_batch_size = 128, 64
# filter_lang = True
# device = torch.device("cuda")
# init_bienc_tokenizer(bienc_tokenizer_dir)
# init_cross_tokenizer(cross_tokenizer_dir)

# input_df = input_df.merge(topics_df, left_on="topic_id", right_on="id", how="left")

# topic_ids = sorted(list(set(input_df["topic_id"])))
# content_ids, c2i = get_content_ids_c2i(content_df)
# topic2text = get_topic2text(topics_df)
# content2text = get_content2text(content_df)

# t2lang, c2lang = get_t2lang_c2lang(input_df, content_df)

# indices = bienc_main(topic_ids, content_ids,
#                      topic2text, content2text, c2i,
#                      filter_lang, t2lang, c2lang,
#                      bienc_dir, bienc_batch_size, device)

# indices.shape

# cand_df = get_cand_df(topic_ids, indices, c2i)

# classifier_thresh = 0.06

# df = cross_main(classifier_thresh, cand_df, topic2text, content2text, cross_dir, cross_batch_size, device)

# df = df.loc[df["pred"] == 1, ["topic_id", "content_id"]].reset_index(drop=True)

# df

# df = df.groupby("topic_id").agg(lambda x: " ".join(x)).reset_index()

# df

# df = df.rename(columns={"content_id": "content_ids"})

# df

