import pandas as pd
from pathlib import Path

cross_fname = Path("../cross/base-sll_0226-181408/fold-0.csv")

topics_df = pd.read_csv("../data/topics.csv")
content_df = pd.read_csv("../data/content.csv")
df = pd.read_csv(cross_fname)

df = df.merge(topics_df.loc[:, ["id", "language"]], left_on="topic_id", right_on="id")

content2lang = {}
for content_id, lang in zip(content_df["id"], content_df["language"]):
    content2lang[content_id] = lang

all_topic_ids = []
all_cands = []
all_cand_ranks = []
all_cand_langs = []
all_labels = []

for topic_id, cat_gold, cat_cands in zip(df["topic_id"], df["content_ids"], df["cands"]):
    gold = set(cat_gold.split())
    cands = cat_cands.split()
    for rank, cand in enumerate(cands):
        all_topic_ids.append(topic_id)
        all_cands.append(cand)
        all_cand_ranks.append(rank)
        all_cand_langs.append(content2lang[cand])
        if cand in gold:
            all_labels.append(1)
        else:
            all_labels.append(0)

foo = pd.DataFrame({"topic_id": all_topic_ids, "content_id": all_cands, "rank": all_cand_ranks, "cand_lang": all_cand_langs, "labels": all_labels})

foo = foo.merge(topics_df.loc[:, ["id", "language"]], left_on="topic_id", right_on="id", how="left")

bar = df.loc[df["topic_id"] == "t_007e3770673d", :]