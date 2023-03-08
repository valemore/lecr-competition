import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from data.topics import get_topic2text
from data.content import get_content2text
from utils import get_topic_id_gold

pd.options.display.max_columns = None
pd.options.display.max_colwidth = None

corr_df = pd.read_csv("../data/correlations.csv")
topics_df = pd.read_csv("../data/topics.csv", keep_default_na=False)
content_df = pd.read_csv("../data/content.csv", keep_default_na=False)
topic2text = get_topic2text(topics_df)
content2text = get_content2text(content_df)

corr_df["content_ids"].duplicated().sum() / corr_df.shape[0]
# 0.23112310418258367
corr_df["content_ids"].duplicated(keep=False).sum() / corr_df.shape[0]
# 0.37553846904107807
corr_df["dup"] = corr_df["content_ids"].duplicated(keep=False)

dup_df = corr_df.loc[corr_df["dup"], :].sort_values("content_ids")
dup_df["t_text"] = [topic2text[x] for x in dup_df["topic_id"]]

dup_df = dup_df.merge(topics_df.loc[:, ["id", "channel", "category", "title"]], left_on="topic_id", right_on="id", how="left")
dup_df["category"].value_counts(normalize=True)
# source          0.606831
# supplemental    0.223920
# aligned         0.169249

dup_df["channel"].value_counts(normalize=True).iloc[:10]

channel2topic = {}
for channel, topic_id, parent_id in zip(topics_df["channel"], topics_df["id"], topics_df["parent"]):
    if not parent_id:
        channel2topic[channel] = topic_id

channel2rep = {k: topic2text[v] for k, v in channel2topic.items()}
[channel2rep[x] for x in dup_df["channel"].value_counts(normalize=True).iloc[:20].index]

foo = dup_df.groupby("content_ids")["title"].nunique()
(foo == 1).value_counts(normalize=True)

nondup = corr_df.loc[~corr_df["topic_id"].isin(set(dup_df["topic_id"])), :]
nondup = nondup.merge(topics_df.loc[:, ["id", "channel", "category", "title"]], left_on="topic_id", right_on="id", how="left")
nondup["title"].duplicated(keep=False).sum() / nondup.shape[0]
# 0.33760249902381884


from Levenshtein import distance
dup_df



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


root_topics = [topic_id for topic_id, node in t2node.items() if node.parent is None]
[x for x in root_topics if t2title[x].startswith("Khan Academy (English")]
india_root = t2node["t_8046010a623f"]
us_root = t2node["t_b22641d12ddc"]

