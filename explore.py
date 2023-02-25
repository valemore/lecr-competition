from collections import defaultdict
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pathlib import Path
import pandas as pd


pd.options.display.max_columns = None
pd.options.display.max_colwidth = None

DATA_DIR = Path("../data")

corr_df = pd.read_csv(DATA_DIR / "correlations.csv")
topics_df = pd.read_csv(DATA_DIR / "topics.csv")
content_df = pd.read_csv(DATA_DIR / "content.csv")

# Check out topics with many contents
# foo = pd.Series([len(x.split()) for x in corr_df["content_ids"].items]) > 50
# corr_df.loc[foo, :].sample(1)

# topics_df.loc[topics_df["id"] == "t_8623f5b58d4c", :]
#
# topics_df.loc[topics_df["id"] == "t_92cf4e58f786", :]


# Number of channels
len(set(topics_df["channel"])) # 171

# Topic 2 contents
pd.Series([len(x.split()) for x in corr_df["content_ids"]]).describe()
# count    61517.000000
# mean         4.550271
# std          6.700255
# min          1.000000
# 25%          2.000000
# 50%          3.000000
# 75%          5.000000
# max        293.000000

(pd.Series([len(x.split()) for x in corr_df["content_ids"]]) > 30).value_counts()
# False    61068
# True       449

# Content 2 topics
content2topics = defaultdict(set)
for topic_id, content_ids in zip(corr_df["topic_id"], corr_df["content_ids"]):
    for content_id in content_ids.split():
        content2topics[content_id].add(topic_id)
pd.Series(len(x) for x in content2topics.values()).describe()
# count    154047.000000
# mean          1.817101
# std           2.003279
# min           1.000000
# 25%           1.000000
# 50%           1.000000
# 75%           2.000000
# max         241.000000

(pd.Series(len(x) for x in content2topics.values()) > 20).value_counts()
# False    153912
# True        135

(pd.Series(len(x) for x in content2topics.values()) > 9).value_counts()
# False    152326
# True       1721

# Topic 2 channel
topic2channel = {}
for topic_id, channel in zip(topics_df["id"], topics_df["channel"]):
    topic2channel[topic_id] = channel


corr_df["num_contents"] = [len(x.split()) for x in corr_df["content_ids"]]
corr_df["channel"] = [topic2channel[x] for x in corr_df["topic_id"]]
foo = corr_df.groupby("channel").sum("num_contents")
channel2num_examples =
# TODO



# Content 2 channels
content2channels = defaultdict(set)
for topic_id, content_ids in zip(corr_df["topic_id"], corr_df["content_ids"]):
    for content_id in content_ids.split():
        content2channels[content_id].add(topic2channel[topic_id])
(pd.Series(len(x) for x in content2channels.values()) > 10).value_counts()
# False    153945
# True        102
(pd.Series(len(x) for x in content2channels.values()) > 4).value_counts()
# False    152068
# True       1979


# Can content be associated with multiple topics/channels? Yes!
# Split by channel or by topic? Also somehow by content?


channel2topics = defaultdict(set)
for topic_id, channel_id in topic2channel.items():
    channel2topics[channel_id].add(topic_id)

pd.Series([len(x) for x in channel2topics.values()]).describe()
# ?

# Language
content2lang = {}
for content_id, language in zip(content_df["id"], content_df["language"]):
    content2lang[content_id] = language

topic2lang = {}
for topic_id, language in zip(topics_df["id"], topics_df["language"]):
    topic2lang[topic_id] = language

# has content flag
hascontent_dct = {}
for topic_id, hascontent in zip(topics_df["id"], topics_df["has_content"]):
    hascontent_dct[topic_id] = hascontent

# Examine whether language flags and has_content flags are always correct
non_matching_lang = set()
incorrect_has_content_flag = set()
for content_id, topic_ids in content2topics.items():
    for topic_id in topic_ids:
        if content2lang[content_id] != topic2lang[topic_id]:
            non_matching_lang.add((content_id, topic_id))
        if not hascontent_dct[topic_id]:
            incorrect_has_content_flag.add(topic_id)

len(set([x[1] for x in non_matching_lang])) / len(corr_df)
# 0.010452395272851406 -  1% of topics have content items that don't align in their language

# Language distribution
topics_df.loc[topics_df["id"].isin(corr_df["topic_id"]), "language"].value_counts()
# en     28053
# es     11769
# pt      3425
# ar      3173
# fr      3034
# bg      2420
# sw      2082
# gu      1809
# bn      1731
# hi      1373
# it       722
# zh       672
# mr       239
# fil      224
# as       126
# my       110
# km       104
# kn        88
# te        66
# ur        54
# or        51
# ta        44
# pnb       40
# swa       33
# pl        28
# tr        26
# ru        21

content_df["language"].value_counts()
# en     65939
# es     30844
# fr     10682
# pt     10435
# ar      7418
# bg      6050
# hi      4042
# zh      3849
# gu      3677
# bn      2513
# sw      1447
# it      1300
# mr       999
# as       641
# fil      516
# km       505
# kn       501
# swa      495
# or       326
# pl       319
# te       285
# ur       245
# tr       225
# ta       216
# my       206
# ru       188
# pnb      184
