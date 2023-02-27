from collections import defaultdict
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pathlib import Path
import pandas as pd


pd.options.display.max_columns = None
pd.options.display.max_colwidth = None

DATA_DIR = Path("../data")

corr_df = pd.read_csv(DATA_DIR / "correlations.csv")
topics_df = pd.read_csv(DATA_DIR / "topics.csv", keep_default_na=False)
content_df = pd.read_csv(DATA_DIR / "content.csv")


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


node = t2node["t_ee80f47eb3be"]
