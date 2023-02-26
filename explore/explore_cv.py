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
for topic_id, parent_id in zip(topics_df["id"], topics_df["parent"]):
    t2parent[topic_id] = parent_id

t2node = {}

class TopicNode:
    def __init__(self, topic_id):
        self.topic_id = topic_id
        self.parent = None
        self.children = []
        self.num_descendants = 0

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

root_topics = [topic_id for topic_id, node in t2node.items() if node.parent is None]

def count_descendants(node):
    if not node.children:
        return 0
    acc = 0
    for child in node.children:
        acc += 1
        acc += count_descendants(child)
    return acc

root2num_descendants = {}
for root in root_topics:
    root_node = t2node[root]
    root2num_descendants[root] = count_descendants(root_node)

t2category = {}
for topic_id, category in zip(topics_df["id"], topics_df["category"]):
    t2category[topic_id] = category
pd.Series([t2category[root] for root in root_topics]).value_counts(normalize=True)


def all_children_in_cat(node, category):
    if t2category[node.topic_id] != category:
        return False
    return all((all_children_in_cat(child, category) for child in node.children))

for root in root_topics:
    root_node = t2node[root]
    root_category = t2category[root_node.topic_id]
    if not all_children_in_cat(root_node, root_category):
        print("hol up")
        break
