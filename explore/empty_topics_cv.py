import os
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import fscore_from_prec_rec

PSEUDO_DIR = Path("../pseudo/satnight_0311-022906")
NUM_FOLDS = 3
THRESH = 0.049

df = pd.read_csv(PSEUDO_DIR / "all_folds.csv")

df["pred"] = df["prob"] >= THRESH

topic_preds = df.loc[:, ["topic_id", "pred"]].groupby("topic_id").sum()
null_topics = topic_preds.loc[topic_preds["pred"] == 0, :].index