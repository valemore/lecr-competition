from collections import defaultdict
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pathlib import Path
import pandas as pd


pd.options.display.max_columns = None
pd.options.display.max_colwidth = None

DATA_DIR = Path("../data")

topics_df = pd.read_csv(DATA_DIR / "topics.csv", keep_default_na=False)
content_df = pd.read_csv(DATA_DIR / "content.csv", keep_default_na=False)
topics_df["len_title"] = topics_df["title"].str.len()
topics_df["len_desc"] = topics_df["description"].str.len()
topics_df["len_title"].describe()
topics_df["len_title"].quantile(0.9)
topics_df["len_title"].quantile(0.95)
topics_df["description"].quantile(0.95)
topics_df["len_description"].quantile(0.95)
topics_df["len_desc"].quantile(0.95)
content_df["len_title"] = content_df["title"].str.len()
content_df["len_desc"] = content_df["description"].str.len()
content_df["len_text"] = content_df["text"].str.len()
content_df["len_title"].quantile(0.95)
content_df["len_desc"].quantile(0.95)
content_df["len_text"].quantile(0.95)
96 * 3
6 * 128
6 * 64