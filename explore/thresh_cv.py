import os
import neptune.new as neptune
import numpy as np
import pandas as pd

from cross.metrics import CROSS_EVAL_THRESHS
from metrics import fscore_from_prec_rec


os.environ["NEPTUNE_API_TOKEN"] = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiYjA4NzA3YWQtZjQxNi00MTY3LWE1NzktZDg0MmY2ZWQ3MzdkIn0="


runids = ["KLB-1064", "KLB-1053", "KLB-1050"]

scores = np.empty((len(runids), len(CROSS_EVAL_THRESHS)), dtype=float)
for i, runid in enumerate(runids):
    run = neptune.init_run(project="vmorelli/kolibri", with_id=runid)
    for j, thresh in enumerate(CROSS_EVAL_THRESHS):
        scores[i, j] = run[f"cross/f2@{str(thresh)}"].fetch_last()
    run.stop()

mean_scores = np.mean(scores, axis=0)
best_thresh_idx = np.argmax(mean_scores)

best_thresh = CROSS_EVAL_THRESHS[best_thresh_idx]
# 0.049

scores[:, best_thresh_idx]
# array([0.62275756, 0.61361609, 0.62744043])

mean_scores[best_thresh_idx]
# 0.621271360613253

best_threshs = np.argmax(scores, 1)
CROSS_EVAL_THRESHS[best_threshs]
# array([0.051, 0.083, 0.024])
np.mean(CROSS_EVAL_THRESHS[best_threshs])
# 0.05266666666666667