import warnings

from data.content import get_content2text
from data.topics import get_topic2text
from utils import get_dfs, merge_cols

warnings.simplefilter(action='ignore', category=FutureWarning)
from pathlib import Path
import pandas as pd


pd.options.display.max_columns = None
pd.options.display.max_colwidth = None


topics_df, content_df, corr_df = get_dfs("../data", "bienc")

topic2title = {}
topic2description = {}
topic2parent = {}
for topic_id, title, description, parent_id in zip(topics_df["id"], topics_df["title"], topics_df["description"], topics_df["parent"]):
    topic2title[topic_id] = title.strip()
    topic2description[topic_id] = description.strip()
    topic2parent[topic_id] = parent_id.strip()


def build_topic_repr(topic_id, depth=2):
    text = topic2title[topic_id]
    if depth == 0:
        return text
    parent = topic2parent[topic_id]
    if parent:
        text += " <<< " + build_topic_repr(parent, depth-1)
    return text

REPR_DEPTH = 0

corr_df["rep"] = [build_topic_repr(x, REPR_DEPTH) for x in corr_df["topic_id"]]


corr_df["group"] = corr_df.groupby(["language", "rep"]).ngroup()
group_num_topics  = corr_df.loc[:, ["group", "topic_id"]].groupby("group").nunique()
corr_df["num_topics"] = [group_num_topics.loc[g].item() for g in corr_df["group"]]

dup = corr_df.loc[corr_df.num_topics > 1, :].reset_index(drop=True)



corr_df.loc[corr_df.group == 34092, :]





corr_df["num_topics"] = corr_df.groupby(["language", "rep"])["topic_id"].nunique()


grouped = corr_df.loc[:, ["topic_id", "language", "rep"]].groupby(["language", "rep"]).count().reset_index()

grouped.loc[grouped["topic_id"] > 1, ""]
