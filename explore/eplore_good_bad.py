from collections import defaultdict
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pathlib import Path
import pandas as pd


pd.options.display.max_columns = None
pd.options.display.max_colwidth = None
# pd.options.display.max_rows = None

DATA_DIR = Path("../data")

corr_df = pd.read_csv(DATA_DIR / "correlations.csv")
topics_df = pd.read_csv(DATA_DIR / "topics.csv")
content_df = pd.read_csv(DATA_DIR / "content.csv")

good_df = pd.DataFrame()
for fold_idx in range(5):
    train_df = pd.read_csv(DATA_DIR / f"../good/train_fold{fold_idx}.csv")
    val_df = pd.read_csv(DATA_DIR / f"../good/val_fold{fold_idx}.csv")
    train_topics = set(train_df["topic_id"])
    val_topics = set(val_df["topic_id"])
    assert not train_topics & val_topics
    assert len(train_topics) + len(val_topics) == corr_df.shape[0]
    val_df["fold"] = fold_idx
    good_df = pd.concat([good_df, val_df])


bad_df = pd.DataFrame()
for fold_idx in range(5):
    train_df = pd.read_csv(DATA_DIR / f"../bad/train_fold{fold_idx}.csv")
    val_df = pd.read_csv(DATA_DIR / f"../bad/val_fold{fold_idx}.csv")
    train_topics = set(train_df["topic_id"])
    val_topics = set(val_df["topic_id"])
    assert not train_topics & val_topics
    assert len(train_topics) + len(val_topics) == corr_df.shape[0]
    val_df["fold"] = fold_idx
    bad_df = pd.concat([bad_df, val_df])

del train_df, val_df

del corr_df
good_df.drop(columns=["content_ids"], inplace=True)
bad_df.drop(columns=["content_ids"], inplace=True)

channel_titles = topics_df.loc[topics_df["parent"].isna(), ["channel", "title"]]
good_df = good_df.merge(channel_titles, on="channel", how="left")
bad_df = bad_df.merge(channel_titles, on="channel", how="left")


bad_df.groupby("fold")["title"].unique()[4]


good_df = good_df.merge(topics_df.loc[:, ["id", "language"]], left_on="topic_id", right_on="id", how="left")
bad_df = bad_df.merge(topics_df.loc[:, ["id", "language"]], left_on="topic_id", right_on="id", how="left")

good_df.groupby("fold")["language"].value_counts(normalize=True)
bad_df.groupby("fold")["language"].value_counts(normalize=True)



good_df['content_ids'] = good_df['content_ids'].str.split()
good_df = (good_df.explode("content_ids")
           .rename(columns={"content_ids": "content_id"})
           .reset_index(drop = True))

good_df.groupby("fold").count()


bad_df['content_ids'] = bad_df['content_ids'].str.split()
bad_df = (bad_df.explode("content_ids")
           .rename(columns={"content_ids": "content_id"})
           .reset_index(drop = True))

bad_df.groupby("fold").count()
