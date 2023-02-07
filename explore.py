from collections import defaultdict
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pathlib import Path
import pandas as pd

DATA_DIR = Path("../data")

corr_df = pd.read_csv(DATA_DIR / "correlations.csv")
topics_df = pd.read_csv(DATA_DIR / "topics.csv")
content_df = pd.read_csv(DATA_DIR / "content.csv")

foo = pd.Series([len(x.split()) for x in corr_df["content_ids"].items]) > 50

corr_df.loc[foo, :].sample(1)

pd.options.display.max_columns = None
pd.options.display.max_colwidth = None
topics_df.loc[topics_df["id"] == "t_8623f5b58d4c", :]

topics_df.loc[topics_df["id"] == "t_92cf4e58f786", :]


len(set(topics_df["channel"]))



content2topics = defaultdict(set)

for topic_id, content_ids in zip(corr_df["topic_id"], corr_df["content_ids"]):
    for content_id in content_ids.split():
        content2topics[content_id].add(topic_id)

pd.Series(len(x) for x in content2topics.values()).describe()


topic2channel = {}
for topic_id, channel in zip(topics_df["id"], topics_df["channel"]):
    topic2channel[topic_id] = channel

content2channels = defaultdict(set)

for topic_id, content_ids in zip(corr_df["topic_id"], corr_df["content_ids"]):
    for content_id in content_ids.split():
        content2channels[content_id].add(topic2channel[topic_id])


(pd.Series(len(x) for x in content2channels.values()) > 10).value_counts()


# Can content be associated with multiple topics/channels?
# Split by channel or by topic? Also somehow by content?


channel2topics = defaultdict(set)
for topic_id, channel_id in topic2channel.items():
    channel2topics[channel_id].add(topic_id)

[len(x) for x in channel2topics.values()]