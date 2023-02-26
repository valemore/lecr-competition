import pandas as pd
from pathlib import Path


cross_output_dir = Path("../../cross")
experiment_id = "baseFEEDchannelCV_0225-170656"

cross_df = pd.DataFrame()
for fold_idx in range(5):
    cross_df = pd.concat([cross_df, pd.read_csv(cross_output_dir / f"{experiment_id}" / f"fold-{fold_idx}.csv", keep_default_na=False)]).reset_index(drop=True)
cross_df = cross_df.sort_values("topic_id").reset_index(drop=True)
cross_df.to_csv(cross_output_dir / f"{experiment_id}" / "all_folds.csv", index=False)
