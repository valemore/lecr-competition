from collections import defaultdict
import warnings

from config import CFG
from cross.dset import CrossDataset
from data.content import get_content2text
from data.topics import get_topic2text
from explore.topic_tree import TopicTree
from metrics import single_fscore, fscore_from_prec_rec
from utils import get_dfs, merge_cols, get_topic_id_gold

warnings.simplefilter(action='ignore', category=FutureWarning)
from pathlib import Path
import pandas as pd


pd.options.display.max_columns = None
pd.options.display.max_colwidth = None


topics_df, content_df, corr_df = get_dfs("../data", "bienc")

tree = TopicTree(topics_df)


corr_df["num_gold"] = [len(x.split()) for x in corr_df["content_ids"]]

corr_df["level"] = [tree[t].level for t in corr_df["topic_id"]]
corr_df["height"] = [tree[t].height() for t in corr_df["topic_id"]]
corr_df["is_leaf"] = [tree[t].is_leaf() for t in corr_df["topic_id"]]
corr_df["num_siblings"] = [tree[t].num_siblings() for t in corr_df["topic_id"]]

corr_df.loc[:, ["num_gold", "level", "height", "is_leaf", "num_siblings"]].corr()


CFG.cross_corr_fname = "../cross/roberta-large-cos10_0307-003338/all_folds.csv"
_, _, cross_df = get_dfs("../data", "cross")



cross_df = merge_cols(cross_df, corr_df, ["num_gold"], on="topic_id")

all_f2 = []
for cat_gold, cat_cands in zip(cross_df.content_ids, cross_df.cands):
    gold = set(cat_gold.split())
    num_gold = len(gold)
    pred = set(cat_cands.split()[:num_gold])
    all_f2.append(single_fscore(gold, pred))

import numpy as np
np.mean(all_f2)


df = pd.read_csv("../pseudo/satnight_0311-022906/all_folds.csv")
df = df.sort_values(["topic_id", "prob"], ascending=[True, False]).reset_index(drop=True)
df["rank"] = df.groupby("topic_id").cumcount()
df = merge_cols(df, corr_df, ["num_gold"], on="topic_id")
df["tp"] = (df["rank"] <= (df["num_gold"])) & df["label"]

grouped = df.loc[:, ["topic_id", "tp"]].groupby("topic_id").sum().reset_index()
grouped = merge_cols(grouped, corr_df, ["num_gold"], on="topic_id")

grouped["p"] = grouped["tp"] / grouped["num_gold"]
grouped["r"] = grouped["tp"] / grouped["num_gold"]
grouped["f2"] = [fscore_from_prec_rec(p, r) for p, r in zip(grouped["p"], grouped["r"])]
grouped["f2"].mean()


df = pd.read_csv("../pseudo/satnight_0311-022906/all_folds.csv")
df = df.sort_values(["topic_id", "prob"], ascending=[True, False]).reset_index(drop=True)
df["rank"] = df.groupby("topic_id").cumcount()
df = merge_cols(df, corr_df, ["num_gold"], on="topic_id")
df["tp"] = (df["rank"] <= (df["num_gold"] + 1)) & df["label"]

grouped = df.loc[:, ["topic_id", "tp"]].groupby("topic_id").sum().reset_index()
grouped = merge_cols(grouped, corr_df, ["num_gold"], on="topic_id")

grouped["p"] = grouped["tp"] / (grouped["num_gold"] + 1)
grouped["r"] = grouped["tp"] / grouped["num_gold"]
grouped["f2"] = [fscore_from_prec_rec(p, r) for p, r in zip(grouped["p"], grouped["r"])]
grouped["f2"].mean()


# -----
noise = np.random.randn(corr_df.shape[0]) * 2.0
corr_df["noisy_num_gold"] = (corr_df["num_gold"] + noise).clip(lower=1.0)

df = pd.read_csv("../pseudo/satnight_0311-022906/all_folds.csv")
df = df.sort_values(["topic_id", "prob"], ascending=[True, False]).reset_index(drop=True)
df["rank"] = df.groupby("topic_id").cumcount()
df = merge_cols(df, corr_df, ["num_gold", "noisy_num_gold"], on="topic_id")
df["tp"] = (df["rank"] <= (df["noisy_num_gold"])) & df["label"]

grouped = df.loc[:, ["topic_id", "tp"]].groupby("topic_id").sum().reset_index()
grouped = merge_cols(grouped, corr_df, ["num_gold", "noisy_num_gold"], on="topic_id")

grouped["p"] = grouped["tp"] / grouped["noisy_num_gold"]
grouped["r"] = grouped["tp"] / grouped["num_gold"]
grouped["f2"] = [fscore_from_prec_rec(p, r) for p, r in zip(grouped["p"], grouped["r"])]
grouped["f2"].mean()


df.groupby("topic_id")

noise = np.random.randn(corr_df.shape[0]) * 2.0

corr_df["noisy_num_gold"] = (corr_df["num_gold"] + noise).clip(lower=1.0)

df = pd.read_csv("../pseudo/satnight_0311-022906/all_folds.csv")

df = df.sort_values(["topic_id", "prob"], ascending=[True, False]).reset_index(drop=True)

df["rank"] = df.groupby("topic_id").cumcount()

df = merge_cols(df, corr_df, ["num_gold"], on="topic_id")

df["tp"] = (df["rank"] <= df["num_gold"]) & df["label"]

grouped = df.loc[:, ["topic_id", "tp"]].groupby("topic_id").sum().reset_index()
grouped = merge_cols(grouped, corr_df, ["noisy_num_gold", "num_gold"], on="topic_id")

grouped["p"] = grouped["num_gold"] / grouped["tp"]
grouped["r"] = grouped["tp"] / grouped["num_gold"]
grouped["f2"] = [fscore_from_prec_rec(p, r) for p, r in zip(grouped["p"], grouped["r"])]
grouped["f2"].mean()



foo = df.iloc[:1000, :]


CUTOFF = 8

df = pd.read_csv("../pseudo/satnight_0311-022906/all_folds.csv")
df = df.sort_values(["topic_id", "prob"], ascending=[True, False]).reset_index(drop=True)
df["rank"] = df.groupby("topic_id").cumcount()
df = merge_cols(df, corr_df, ["num_gold"], on="topic_id")
df["tp"] = (df["rank"] <= CUTOFF) & df["label"]

grouped = df.loc[:, ["topic_id", "tp"]].groupby("topic_id").sum().reset_index()
grouped = merge_cols(grouped, corr_df, ["num_gold"], on="topic_id")

grouped["p"] = grouped["tp"] / CUTOFF
grouped["r"] = grouped["tp"] / grouped["num_gold"]
grouped["f2"] = [fscore_from_prec_rec(p, r) for p, r in zip(grouped["p"], grouped["r"])]
grouped["f2"].mean()