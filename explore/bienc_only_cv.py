import os
import neptune.new as neptune
import numpy as np
import pandas as pd

from metrics import fscore_from_prec_rec


os.environ["NEPTUNE_API_TOKEN"] = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiYjA4NzA3YWQtZjQxNi00MTY3LWE1NzktZDg0MmY2ZWQ3MzdkIn0="


dct = {
    "cos-15": ["KLB-692"],
    "cos-10": ["KLB-691"],
    "onlybi": ["KLB-885", "KLB-886", "KLB-887", "KLB-896", "KLB-901"],
    "roberta-large-cos10": ["KLB-772", "KLB-773", "KLB-774", "KLB-775", "KLB-794"]
}

rows = []
for name, runids in dct.items():
    for runid in runids:
        run = neptune.init_run(project="vmorelli/kolibri", with_id=runid)
        precision = np.array([run[f"cands/precision@{i}"].fetch_last() for i in range(1, 51)])
        recall = np.array([run[f"cands/recall@{i}"].fetch_last() for i in range(1, 51)])
        f2 = np.array([fscore_from_prec_rec(p, r) for p, r in zip(precision, recall)])
        best_i = np.argmax(f2)
        best_f2 = f2[best_i]

        rows.append({"name": name, "runid": runid, "f2": best_f2, "num_cands": (best_i + 1)})
        run.stop()


df = pd.DataFrame.from_records(rows)

df.loc[:, ["name", "f2", "num_cands"]].groupby("name").mean()

#                            f2  num_cands
# name
# cos-10               0.616976        6.0
# cos-15               0.623210        6.0
# onlybi               0.617060        6.0
# roberta-large-cos10  0.605716        6.8
