import numpy as np
import pandas as pd

from bienc.dset import BiencDataset
import bienc.tokenizer as tokenizer
from ceevee import get_topics_in_corr
from config import CFG
from data.content import get_content2text
from data.topics import get_topic2text
from utils import get_t2lang_c2lang

content_df = pd.read_csv(CFG.DATA_DIR / "content.csv")
corr_df = pd.read_csv(CFG.DATA_DIR / "correlations.csv")
topics_df = pd.read_csv(CFG.DATA_DIR / "topics.csv")
corr_df = corr_df.merge(topics_df.loc[:, ["id", "language"]], left_on="topic_id", right_on="id", how="left")

t2lang, c2lang = get_t2lang_c2lang(corr_df, content_df)
topics_in_corr = get_topics_in_corr(corr_df)

c2i = {content_id: content_idx for content_idx, content_id in enumerate(sorted(set(content_df["id"])))}
topic2text = get_topic2text(topics_df)
content2text = get_content2text(content_df)

t2i = {topic: idx for idx, topic in enumerate(sorted(list(set(corr_df["topic_id"]))))}

dset = BiencDataset(corr_df["topic_id"], corr_df["content_ids"],
                    corr_df["language"],
                    topic2text, content2text, -1, -1, t2i, c2i)


topic_seq_lens = []
content_seq_lens = []

tokenizer.init_tokenizer()

for idx in range(len(dset)):
    topic_input_ids, _, content_input_ids, *_ = dset[idx]
    topic_seq_lens.append(len(topic_input_ids))
    content_seq_lens.append(len(content_input_ids))


quantile_probs = [np.round(x, 2) for x in np.arange(0.75, 1.01, 0.01)]

pd.Series(topic_seq_lens).describe()
# count    279919.000000
# mean         56.054312
# std          53.798255
# min           3.000000
# 25%          27.000000
# 50%          40.000000
# 75%          70.000000
# max        1236.000000
pd.Series(topic_seq_lens).quantile(quantile_probs)
# 0.75      70.0
# 0.76      72.0
# 0.77      73.0
# 0.78      75.0
# 0.79      77.0
# 0.80      79.0
# 0.81      81.0
# 0.82      84.0
# 0.83      86.0
# 0.84      88.0
# 0.85      91.0
# 0.86      94.0
# 0.87      97.0
# 0.88     100.0
# 0.89     104.0
# 0.90     108.0
# 0.91     112.0
# 0.92     116.0
# 0.93     120.0
# 0.94     125.0
# 0.95     131.0
# 0.96     141.0
# 0.97     159.0
# 0.98     183.0
# 0.99     259.0
# 1.00    1236.0

# WITHOUT CONTENT_TEXT
pd.Series(content_seq_lens).describe()
# count    279919.000000
# mean         33.572555
# std          42.573578
# min           2.000000
# 25%          13.000000
# 50%          27.000000
# 75%          42.000000
# max        2206.000000
pd.Series(content_seq_lens).quantile(quantile_probs)
# 0.75      42.0
# 0.76      43.0
# 0.77      44.0
# 0.78      44.0
# 0.79      45.0
# 0.80      46.0
# 0.81      47.0
# 0.82      48.0
# 0.83      49.0
# 0.84      51.0
# 0.85      52.0
# 0.86      53.0
# 0.87      55.0
# 0.88      57.0
# 0.89      59.0
# 0.90      61.0
# 0.91      63.0
# 0.92      66.0
# 0.93      70.0
# 0.94      74.0
# 0.95      80.0
# 0.96      88.0
# 0.97      98.0
# 0.98     113.0
# 0.99     147.0
# 1.00    2206.0

# WITH CONTENT_TEXT
pd.Series(content_seq_lens).describe()
# count    279919.000000
# mean       1186.319796
# std        3606.407975
# min           3.000000
# 25%          24.000000
# 50%          54.000000
# 75%        1069.000000
# max       79823.000000
pd.Series(content_seq_lens).quantile(quantile_probs)
# 0.75     1069.00
# 0.76     1125.00
# 0.77     1184.00
# 0.78     1247.00
# 0.79     1315.00
# 0.80     1384.00
# 0.81     1452.00
# 0.82     1529.00
# 0.83     1617.00
# 0.84     1706.00
# 0.85     1809.00
# 0.86     1916.00
# 0.87     2026.00
# 0.88     2144.00
# 0.89     2277.00
# 0.90     2440.00
# 0.91     2649.00
# 0.92     2903.00
# 0.93     3224.00
# 0.94     3718.00
# 0.95     4397.00
# 0.96     5443.00
# 0.97     7093.00
# 0.98    11015.48
# 0.99    22464.00
# 1.00    79823.00

# def tokenize(text: str, num_tokens: int) -> Dict[str, List[int]]:
#     """
#     Get input ids and attention mask.
#     :param text: text to tokenize
#     :param num_tokens: truncate and pad to this many tokens
#     :return: dict with input ids and attention mask
#     """
#     enc = tokenizer(text,
#                     add_special_tokens=True,
#                     return_overflowing_tokens=False,
#                     return_offsets_mapping=False)
#
#     return {"input_ids": enc.input_ids, "attention_mask": enc.attention_mask}

# foo = content_df.sample()
# bar = content_df.loc[~content_df["text"].isna(), :]
# foo = bar.sample(10)
# content_df["title"].str.len()
# content_df["title"].str.len().describe()
# content_df["description"].str.len().describe()
# content_df["text"].str.len().describe()
# bar
# foo
# [x[:128] for x in foo["text"]]
# foobar = [x[:128] for x in foo["text"]]
# foobar[0]
# foobar[3]
# foobar = [x[:196] for x in foo["text"]]
# foobar[1]
# foobar[2]
# foo2 = bar.sample(10)