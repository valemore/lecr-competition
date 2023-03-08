import pandas as pd

num_cands = 20

cand_df = pd.read_csv("../cross/ssl-seed_0227-173614/all_folds.csv")
cand_df["cands"] = [x.split()[:num_cands] for x in cand_df["cands"]]
cand_df = cand_df.explode("cands")
cand_df = cand_df.rename(columns={"cands": "cand_id"})
cand_df["label"] = [cand in set(x.split()) for cand, x in zip(cand_df["cand_id"], cand_df["content_ids"])]
cand_df["label"] = cand_df["label"].astype(int)
cand_df.sample()

cand_df = cand_df.loc[:, ["topic_id", "cand_id", "label"]]
cand_df = cand_df.loc[cand_df["cand_id"] != "dummy", :].reset_index(drop=True)
cand_df.groupby("cand_id")["topic_id"].count().describe()
cand_df.groupby("cand_id")["topic_id"].count().quantile([0.8, 0.9, 0.95, 0.99])



from data.topics import get_topic2text
from data.content import get_content2text
topics_df = pd.read_csv("../data/topics.csv", keep_default_na=False)
content_df = pd.read_csv("../data/content.csv", keep_default_na=False)
topic2text = get_topic2text(topics_df)
content2text = get_content2text(content_df)


(cand_df.groupby("cand_id")["label"].sum() == 0).value_counts()
(cand_df.groupby("cand_id")["label"].sum() == 0).describe()
cand_df.groupby("cand_id")["label"].sum().describe()
cand_df.groupby("cand_id")["label"].count().describe()
cand_df.groupby("cand_id")["label"].count().quantile([0.8, 0.9, 0.95, 0.99])
cand_df.groupby("cand_id")["label"].sum().quantile([0.8, 0.9, 0.95, 0.99])
cand_df = cand_df.merge(topics_df.loc[:, ["id", "channel"]], left_on="topic_id", right_on="id", how="left")
cand_df.groupby("cand_id")["channel"].nunique().describe()
cand_df.groupby("cand_id")["channel"].nunique().quantile([0.8, 0.9, 0.95, 0.99])
cand_df.groupby(["cand_id", "channel"]).first().shape


# by topics
cand_df = pd.read_csv("../cross/ssl-seed_0227-173614/all_folds.csv")
cand_df["cands"] = [x.split()[:num_cands] for x in cand_df["cands"]]
cand_df = cand_df.explode("cands")
cand_df = cand_df.rename(columns={"cands": "cand_id"})
cand_df["label"] = [cand in set(x.split()) for cand, x in zip(cand_df["cand_id"], cand_df["content_ids"])]
cand_df["label"] = cand_df["label"].astype(int)
cand_df = cand_df.sort_values(["topic_id", "label"], ascending=[True, False])

cand_df = cand_df.loc[:, ["topic_id", "cand_id", "language", "label"]]
can_df = cand_df.iloc[:10000, :]