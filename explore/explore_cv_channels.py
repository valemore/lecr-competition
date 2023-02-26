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

topic2channel = {}
for topic_id, channel in zip(topics_df["id"], topics_df["channel"]):
    topic2channel[topic_id] = channel

corr_df["num_examples"] = [len(x.split()) for x in corr_df["content_ids"]]
corr_df["channel"] = [topic2channel[x] for x in corr_df["topic_id"]]
foo = corr_df.groupby("channel").sum("num_examples")


topics_df = pd.read_csv(DATA_DIR / "topics.csv")
channel2category = {}
for channel in set(corr_df["channel"]):
    channel2category[channel] = topics_df.loc[(topics_df["channel"] == channel) & (topics_df["parent"].isna()), "category"].item()

