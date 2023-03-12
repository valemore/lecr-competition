import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import fscore_from_prec_rec
from utils import get_dfs, merge_cols

PSEUDO_DIR = Path("../pseudo/sunnight_0312-013744")
NUM_FOLDS = 3
THRESH = 0.05

df = pd.read_csv(PSEUDO_DIR / "all_folds.csv")

df["pred"] = df["prob"] >= THRESH


topics_df, content_df, corr_df = get_dfs("../data", "bienc")
df = merge_cols(df, topics_df, ["language"], left_on="topic_id", right_on="id")

topic2title = {}
topic2description = {}
topic2parent = {}
for topic_id, title, description, parent_id in zip(topics_df["id"], topics_df["title"], topics_df["description"], topics_df["parent"]):
    topic2title[topic_id] = title.strip()
    topic2description[topic_id] = description.strip()
    topic2parent[topic_id] = parent_id.strip()

content2title = {}
content2description = {}
content2text = {}
for content_id, title, description, text in zip(content_df["id"], content_df["title"], content_df["description"], content_df["text"]):
    content2title[content_id] = title.strip()
    content2description[content_id] = description.strip()
    content2text[content_id] = text.strip()


def build_topic_rep(topic_id, depth=2):
    text = topic2title[topic_id]
    if depth == 0:
        return text
    parent = topic2parent[topic_id]
    if parent:
        text += " <<< " + build_topic_rep(parent, depth-1)
    return text

def build_content_rep(content_id):
    text = content2title[content_id]
    text += " *** " + content2description[content_id]
    return text


foo = df.loc[df["pred"] != df["label"], :].sample(1000).reset_index(drop=True)

foo["t"] = [build_topic_rep(x, 3) for x in foo["topic_id"]]
foo["t_full"] = [build_topic_rep(x, math.inf) for x in foo["topic_id"]]
foo["c"] = [build_content_rep(x) for x in foo["content_id"]]
foo["c_full"] = [build_content_rep(x) for x in foo["content_id"]]

bar = df.loc[df["pred"] == df["label"], :].sample(1000).reset_index(drop=True)

bar["t"] = [build_topic_rep(x, 3) for x in bar["topic_id"]]
bar["t_full"] = [build_topic_rep(x, math.inf) for x in bar["topic_id"]]
bar["c"] = [build_content_rep(x) for x in bar["content_id"]]
bar["c_full"] = [build_content_rep(x) for x in bar["content_id"]]

dup_top = set(df.loc[df.content_id == "c_27af4aa0dc3d", "topic_id"].tolist())
for x in dup_top:
    print(build_topic_rep(x))

dup_top = set(df.loc[df.topic_id == "t_4712540f67be", "topic_id"].tolist())
for x in dup_top:
    print(build_topic_rep(x))

