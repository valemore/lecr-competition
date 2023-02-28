from collections import defaultdict
import warnings
from tqdm import tqdm

import bienc.tokenizer as tokenizer

warnings.simplefilter(action='ignore', category=FutureWarning)
from pathlib import Path
import pandas as pd


pd.options.display.max_columns = None
pd.options.display.max_colwidth = None

DATA_DIR = Path("../data")

topics_df = pd.read_csv(DATA_DIR / "topics.csv", keep_default_na=False)
content_df = pd.read_csv(DATA_DIR / "content.csv", keep_default_na=False)


tokenizer.init_tokenizer()

def get_num_toks(txt):
    return len(tokenizer.tokenize(txt, None)["input_ids"])

topics_df["toks_title"] = [get_num_toks(x) for x in tqdm(topics_df["title"])]
topics_df["toks_desc"] = [get_num_toks(x) for x in tqdm(topics_df["description"])]

content_df["toks_title"] = [get_num_toks(x) for x in tqdm(content_df["title"])]
content_df["toks_desc"] = [get_num_toks(x) for x in tqdm(content_df["description"])]
content_df["toks_text"] = [get_num_toks(x) for x in tqdm(content_df["text"])]

topics_df["toks_title"].dropna().describe()
# count    76972.000000
# mean         8.877800
# std          5.000023
# min          2.000000
# 25%          5.000000
# 50%          8.000000
# 75%         11.000000
# max         99.000000
topics_df["toks_desc"].dropna().describe()
# count    76972.000000
# mean        22.892883
# std         43.815539
# min          2.000000
# 25%          2.000000
# 50%          2.000000
# 75%         31.000000
# max       1220.000000

content_df["toks_title"].dropna().describe()
# count    154047.000000
# mean         11.207222
# std           5.618898
# min           2.000000
# 25%           7.000000
# 50%          10.000000
# 75%          14.000000
# max          95.000000
content_df["toks_desc"].dropna().describe()
# count    154047.000000
# mean         22.008835
# std          49.444248
# min           2.000000
# 25%           2.000000
# 50%          13.000000
# 75%          28.000000
# max        2203.000000

content_df["toks_text"].dropna().describe()
# count    154047.000000
# mean       1446.374658
# std        4159.554560
# min           2.000000
# 25%           2.000000
# 50%           2.000000
# 75%        1145.000000
# max       79806.000000



topics_df["toks_title"].dropna().quantile([0.9, 0.95, 0.99])
# 0.90    15.0
# 0.95    18.0
# 0.99    26.0

topics_df["toks_desc"].dropna().quantile([0.9, 0.95, 0.99])
# 0.90     64.0
# 0.95     88.0
# 0.99    179.0

content_df["toks_title"].dropna().quantile([0.9, 0.95, 0.99])
# 0.90    18.0
# 0.95    22.0
# 0.99    31.0

content_df["toks_desc"].dropna().quantile([0.9, 0.95, 0.99])
# 0.90     46.0
# 0.95     69.0
# 0.99    156.0


content_df["toks_text"].dropna().quantile([0.9, 0.95, 0.99])
# 0.90     3271.00
# 0.95     6272.70
# 0.99    23529.08


(topics_df["title"].dropna().str.len() / topics_df["toks_title"].dropna()).mean()
# 2.9921655875871123
(topics_df["description"].dropna().str.len() / topics_df["toks_desc"].dropna()).mean()
# 1.6336483090465417


(content_df["title"].dropna().str.len() / content_df["toks_title"].dropna()).mean()
# 3.0743342727679366
(content_df["description"].dropna().str.len() / content_df["toks_desc"].dropna()).mean()
# 1.9359440045441612
(content_df["text"].dropna().str.len() / content_df["toks_text"].dropna()).mean()
# 4.474544362566645