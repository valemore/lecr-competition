# Metrics for the Bi-encoder
from typing import Dict, List, Set, Tuple

import numpy as np
from neptune.new import Run

from typehints import MetricDict


BIENC_EVAL_THRESHS = [round(x, 2) for x in np.arange(0.2, 0.62, 0.02)]
BIENC_STANDALONE_THRESHS = [round(x, 2) for x in np.arange(0.1, 0.52, 0.02)]


def get_bienc_metrics(distances, indices,
                      topic_ids: List[str], e2i: Dict[str, int], t2gold: Dict[str, Set[str]]) -> Tuple[MetricDict, MetricDict, float]:
    i2e = {entity_idx: entity_id for entity_id, entity_idx in e2i.items()}
    tp = np.empty_like(indices, dtype=int) # mask indicating whether prediction is a true positive
    num_gold = np.empty(len(topic_ids), dtype=int) # how many content ids are in gold?
    for i, (dists, idxs, topic_id) in enumerate(zip(distances, indices, topic_ids)):
        gold = t2gold[topic_id]
        tp[i, :] = np.array([int(i2e[idx] in gold) for idx in idxs], dtype=int)
        num_gold[i] = len(gold)


    precision_dct = {thresh: 0.0 for thresh in BIENC_EVAL_THRESHS}
    recall_dct = {thresh: 0.0 for thresh in BIENC_EVAL_THRESHS}

    avg_prec = np.zeros(len(topic_ids), dtype=float) # accumulating average precision for all topic ids
    prev_rec = np.zeros(len(topic_ids), dtype=float) # previous recall
    for thresh in BIENC_EVAL_THRESHS:
        mask = distances <= thresh
        thresh_tp = np.copy(tp)
        thresh_tp[~mask] = 0
        num_tp = np.sum(thresh_tp, axis=1)
        num_preds = np.sum(mask, axis=1)
        prec = num_tp / num_preds if num_preds > 0 else 0.0
        rec = num_tp / num_gold if num_gold > 0 else 0.0
        prec[np.isnan(prec)] = -1 # because 0 * nan is not 0
        assert np.sum(prec * (rec - prev_rec) < 0) == 0
        avg_prec += prec * (rec - prev_rec)
        prev_rec = rec
        precision_dct[thresh] = np.mean(prec)
        recall_dct[thresh] = np.mean(rec)
    avg_precision = np.mean(avg_prec)
    return precision_dct, recall_dct, avg_precision.item()


def log_precision_dct(dct: Dict[int, float], label: str, global_step: int, run: Run):
    for k, v in dct.items():
        run[f"{label}@{k}"].log(v, step=global_step)
