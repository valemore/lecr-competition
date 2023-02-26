from collections import defaultdict
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pathlib import Path
import pandas as pd


pd.options.display.max_columns = None
pd.options.display.max_colwidth = None

DATA_DIR = Path("../../data")

corr_df = pd.read_csv(DATA_DIR / "correlations.csv")
topics_df = pd.read_csv(DATA_DIR / "topics.csv")
content_df = pd.read_csv(DATA_DIR / "content.csv")

corr_df["num_contents"] = [len(x.split()) for x in corr_df["content_ids"]]

all_parent_ids = set(topics_df["parent"])
def is_leaf(topic_id):
    return topic_id in all_parent_ids
corr_df["is_leaf"] = [is_leaf(x) for x in corr_df["topic_id"]]

topics_df.loc[topics_df["title"].isna(), "title"] = ""
t2title = {}
for topic_id, title in zip(topics_df["id"], topics_df["title"]):
    t2title[topic_id] = title
def get_topic_title_len(topic_id):
    return len(t2title[topic_id])

corr_df["title_lens"] = [get_topic_title_len(x) for x in corr_df["topic_id"]]

import numpy as np
np.corrcoef(corr_df["num_contents"], "title_lens")
np.corrcoef(corr_df["num_contents"], corr_df["title_lens"])
np.corrcoef(corr_df["num_contents"], corr_df["title_lens"], corr_df["is_leaf"])
corr_df["is_leaf"]
np.corrcoef(corr_df["num_contents"], corr_df["title_lens"], corr_df["is_leaf"].astype(int))
np.corrcoef(corr_df["num_contents"], corr_df["is_leaf"])
np.corrcoef(corr_df["num_contents"])

topics_df["category"].value_counts(normalize=True)
all_source_topic_ids = set(topics_df.loc[topics_df["category"] == "source", "id"])
corr_df["from_source"] = [x in all_source_topic_ids for x in corr_df["topic_id"]]
corr_df["from_source"].value_counts()
corr_df["from_source"].value_counts(normalize=True)

np.corrcoef(corr_df["num_contents"], corr_df["from_source"])