from collections import defaultdict
import warnings

from data.content import get_content2text
from data.topics import get_topic2text

warnings.simplefilter(action='ignore', category=FutureWarning)
from pathlib import Path
import pandas as pd


pd.options.display.max_columns = None
pd.options.display.max_colwidth = None

DATA_DIR = Path("../data")

corr_df = pd.read_csv(DATA_DIR / "correlations.csv")
topics_df = pd.read_csv(DATA_DIR / "topics.csv", keep_default_na=False)
content_df = pd.read_csv(DATA_DIR / "content.csv", keep_default_na=False)


t2parent = {}
t2title = {}
t2description = {}
t2level = {}
for topic_id, parent_id, title, description, level in zip(topics_df["id"], topics_df["parent"], topics_df["title"], topics_df["description"], topics_df["level"]):
    t2parent[topic_id] = parent_id
    t2title[topic_id] = title
    t2description[topic_id] = description
    t2level[topic_id] = level

t2node = {}

class TopicNode:
    def __init__(self, topic_id):
        self.topic_id = topic_id
        self.parent = None
        self.children = []
        self.title = t2title[topic_id]
        self.description = t2description[topic_id]
        self.level = t2level[topic_id]

    def __repr__(self):
        parent_info = f"with parent {self.parent.topic_id}" if self.parent is not None else "at root"
        return f"<Topic {self.topic_id} {parent_info} and {len(self.children)} children."

def add_node(topic_id):
    if topic_id not in t2node:
        node = TopicNode(topic_id)
        parent_id = t2parent[topic_id]
        if parent_id != "":
            if parent_id not in t2node:
                add_node(parent_id)
            parent_node = t2node[parent_id]
            node.parent = parent_node
            parent_node.children.append(node)
        t2node[topic_id] = node


for topic_id in topics_df["id"]:
    add_node(topic_id)


def find_root(node):
    while(node.parent):
        node = node.parent
    return node

def in_same_tree(topic1, topic2):
    node1, node2 = t2node[topic1], t2node[topic2]
    return node1 == node2




node = t2node["t_ee80f47eb3be"]


corr_df = pd.read_csv("../data/correlations.csv")
topic2text = get_topic2text(topics_df)
content2text = get_content2text(content_df)

corr_df["content_ids"].duplicated().sum() / corr_df.shape[0]
# 0.23112310418258367
corr_df["content_ids"].duplicated(keep=False).sum() / corr_df.shape[0]
# 0.37553846904107807
corr_df["dup"] = corr_df["content_ids"].duplicated(keep=False)

dup_df = corr_df.loc[corr_df["dup"], :].sort_values("content_ids")
dup_df["t_text"] = [topic2text[x] for x in dup_df["topic_id"]]

dup_df["gidx"] = dup_df.groupby("")