import pandas as pd

from data.topics import get_topic2text
from data.content import get_content2text
from utils import get_topic_id_gold

pd.options.display.max_columns = None
pd.options.display.max_colwidth = None

num_cands = 20



cand_df = pd.read_csv("../cross/ssl-seed_0227-173614/all_folds.csv")
cand_df["cands"] = [x.split()[:num_cands] for x in cand_df["cands"]]
cand_df = cand_df.explode("cands")
cand_df = cand_df.rename(columns={"cands": "cand_id"})
cand_df["rank"] = cand_df.groupby("topic_id").cumcount()

cand_df["label"] = [cand in set(x.split()) for cand, x in zip(cand_df["cand_id"], cand_df["content_ids"])]
cand_df["label"] = cand_df["label"].astype(int)
cand_df = cand_df.loc[cand_df["cand_id"] != "dummy", :].reset_index(drop=True)
cand_df = cand_df.loc[:, ["topic_id", "cand_id", "label", "rank"]]
cand_df.sample()

cand_df = cand_df.sort_values(["topic_id", "label"], ascending=[True, False])

topics_df = pd.read_csv("../data/topics.csv", keep_default_na=False)
content_df = pd.read_csv("../data/content.csv", keep_default_na=False)
topic2text = get_topic2text(topics_df)
content2text = get_content2text(content_df)

cand_df["t_repr"] = [topic2text[x] for x in cand_df["topic_id"]]
cand_df["c_repr"] = [content2text[x] for x in cand_df["cand_id"]]

import random
idx = random.randint(0, cand_df.shape[0])
foo = cand_df.iloc[idx:(idx+500), :]
# idx = 1089152


idx = random.randint(0, cand_df.shape[0])
foo = cand_df.iloc[idx:(idx+500), :]