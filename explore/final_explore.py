from collections import defaultdict
import warnings

from config import CFG
from cross.dset import CrossDataset
from data.content import get_content2text
from data.topics import get_topic2text
from explore.topic_tree import TopicTree
from utils import get_dfs, merge_cols

warnings.simplefilter(action='ignore', category=FutureWarning)
from pathlib import Path
import pandas as pd


pd.options.display.max_columns = None
pd.options.display.max_colwidth = None


CFG.cross_corr_fname = "../cross/roberta-large-cos10_0307-003338/all_folds.csv"
topics_df, content_df, cross_df = get_dfs("../data", "cross")

tree = TopicTree(topics_df)


topic2text = get_topic2text(topics_df)
content2text = get_content2text(content_df)
cross_dset = CrossDataset(cross_df["topic_id"], cross_df["content_ids"], cross_df["cands"],
                          topic2text, content2text, CFG.CROSS_NUM_TOKENS, is_val=False)

df = pd.DataFrame({"topic_id": cross_dset.topic_ids, "content_id": cross_dset.content_ids, "label": cross_dset.labels})
df = merge_cols(df, topics_df, ["language", "category", "channel"], left_on="topic_id", right_on="id")
df = merge_cols(df, content_df, ["kind"], left_on="content_id", right_on="id")

df.groupby("language").label.value_counts(normalize=True)
# Add language info to cross?

df.loc[df["category"] != "source"].groupby("language").label.value_counts(normalize=True)

df.groupby("category").label.value_counts(normalize=True)
# category      label
# aligned       0        0.966388
#               1        0.033612
# source        0        0.925703
#               1        0.074297
# supplemental  0        0.945445
#               1        0.054555
df.groupby("kind").label.value_counts(normalize=True)

df.groupby("channel").label.value_counts(normalize=True)

df["is_leaf"] = [tree[x].is_leaf() for x in df["topic_id"]]
df["height"] = [tree[x].height() for x in df["topic_id"]]
df["num_siblings"] = [tree[x].num_siblings() for x in df["topic_id"]]
df["num_ancestors_with_content"] = [tree[x].num_ancestors_with_content() for x in df["topic_id"]]
df["level"] = [tree[x].level for x in df["topic_id"]]
df["num_children"] = [len(tree[x].children) for x in df["topic_id"]]

df.loc[:, ["label", "is_leaf", "height", "num_siblings", "num_ancestors_with_content", "level", "num_children"]].corr()
df.loc[df["category"] != "source", ["label", "is_leaf", "height", "num_siblings", "num_ancestors_with_content", "level", "num_children"]].corr()

tree.root_nodes

def count_children(node):
    c = len(node.children)
    for child in node.children:
        c += count_children(child)
    return c

tree_sizes = [count_children(x) for x in tree.root_nodes]



