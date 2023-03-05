from pathlib import Path
import pandas as pd

corr_df = pd.read_csv("../data/correlations.csv", keep_default_na=False)
topics_df = pd.read_csv("../data/topics.csv", keep_default_na=False)
content_df = pd.read_csv("../data/content.csv", keep_default_na=False)

Path("../dbg-data").mkdir(exist_ok=True, parents=True)
test_df = corr_df.sample(100).sort_values("topic_id").to_csv("../dbg-data/sample_submission.csv", index=False)
topics_df.to_csv("../dbg-data/topics.csv", index=False)
content_df.to_csv("../dbg-data/content.csv", index=False)
