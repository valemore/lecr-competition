# Metrics for the Bi-encoder
import math
from typing import Dict, List, Set, Tuple

import numpy as np
from neptune.new import Run

from config import CFG
from typehints import MetricDict
from utils import safe_div_np

BIENC_EVAL_THRESHS = [round(x, 2) for x in np.arange(0.2, 0.62, 0.02)] + [math.inf]
BIENC_STANDALONE_THRESHS = [round(x, 2) for x in np.arange(0.1, 0.52, 0.02)]


def get_bienc_thresh_metrics(distances, indices,
                             topic_ids: List[str], e2i: Dict[str, int], t2gold: Dict[str, Set[str]]) -> Tuple[MetricDict, MetricDict, MetricDict, MetricDict, float]:
    i2e = {entity_idx: entity_id for entity_id, entity_idx in e2i.items()}
    tp = np.empty_like(indices, dtype=int) # mask indicating whether prediction is a true positive
    num_gold = np.empty(len(topic_ids), dtype=int) # how many content ids are in gold?
    for i, (idxs, topic_id) in enumerate(zip(indices, topic_ids)):
        gold = t2gold[topic_id]
        tp[i, :] = np.array([int(i2e[idx] in gold) for idx in idxs], dtype=int)
        num_gold[i] = len(gold)
    total_gold = np.sum(num_gold)


    precision_dct = {thresh: 0.0 for thresh in BIENC_EVAL_THRESHS}
    recall_dct = {thresh: 0.0 for thresh in BIENC_EVAL_THRESHS}
    micro_prec_dct = {thresh: 0.0 for thresh in BIENC_EVAL_THRESHS}
    pcr_dct = {thresh: 0.0 for thresh in BIENC_EVAL_THRESHS}

    avg_prec = np.zeros(len(topic_ids), dtype=float) # accumulating average precision for all topic ids
    prev_rec = np.zeros(len(topic_ids), dtype=float) # previous recall
    for thresh in BIENC_EVAL_THRESHS:
        mask = distances <= thresh
        thresh_tp = np.copy(tp)
        thresh_tp[~mask] = 0
        num_tp = np.sum(thresh_tp, axis=1)
        num_preds = np.sum(mask, axis=1)
        prec = safe_div_np(num_tp, num_preds)
        rec = safe_div_np(num_tp, num_gold)
        prec[np.isnan(prec)] = 0.0 # because 0 * nan is not 0
        assert np.sum(prec * (rec - prev_rec) < 0) == 0
        avg_prec += prec * (rec - prev_rec)
        prev_rec = rec
        precision_dct[thresh] = np.mean(prec)
        recall_dct[thresh] = np.mean(rec)
        micro_prec_dct[thresh] = np.sum(thresh_tp, axis=None) / np.sum(mask, axis=None)
        pcr_dct[thresh] = total_gold / (np.sum((1 - thresh_tp) * mask, axis=None) + total_gold)
    avg_precision = np.mean(avg_prec)
    return precision_dct, recall_dct, micro_prec_dct, pcr_dct, avg_precision.item()


def log_dct(dct: Dict[int, float], label: str, global_step: int, run: Run):
    for k, v in dct.items():
        run[f"{label}@{k}"].log(v, step=global_step)


def get_bienc_cands_metrics(indices, topic_ids: List[str], e2i: Dict[str, int], t2gold: Dict[str, Set[str]], num_cands: int) -> Tuple[MetricDict, MetricDict, MetricDict, MetricDict, float]:
    i2e = {entity_idx: entity_id for entity_id, entity_idx in e2i.items()}
    tp = np.empty_like(indices, dtype=int) # mask indicating whether prediction is a true positive
    num_gold = np.empty(len(topic_ids), dtype=int) # how many content ids are in gold?
    for i, (idxs, topic_id) in enumerate(zip(indices, topic_ids)):
        gold = t2gold[topic_id]
        tp[i, :] = np.array([int(i2e[idx] in gold) for idx in idxs], dtype=int)
        num_gold[i] = len(gold)

    precision_dct = {num_cands: 0.0 for num_cands in range(1, num_cands + 1)}
    recall_dct = {num_cands: 0.0 for num_cands in range(1, num_cands + 1)}
    micro_prec_dct = {num_cands: 0.0 for num_cands in range(1, num_cands + 1)}
    pcr_dct = {num_cands: 0.0 for num_cands in range(1, num_cands + 1)}
    avg_prec = np.zeros(len(topic_ids), dtype=float) # accumulating average precision for all topic ids

    acc_tp = np.zeros(len(topic_ids), dtype=float) # accumulating true positives for all topic ids
    prev_rec = np.zeros(len(topic_ids), dtype=float) # previous recall
    for j, num_cands in enumerate(range(1, num_cands + 1)):
        acc_tp += tp[:, j]
        prec = acc_tp / num_cands
        rec = acc_tp / num_gold
        avg_prec += prec * (rec - prev_rec)
        prev_rec = rec
        precision_dct[num_cands] = np.mean(prec)
        recall_dct[num_cands] = np.mean(rec)
        micro_prec_dct[num_cands] = np.sum(acc_tp) / (len(topic_ids) * num_cands)
        pcr_dct[num_cands] = np.sum(num_gold) / ( len(topic_ids) * num_cands - np.sum(acc_tp) + np.sum(num_gold))
    avg_precision = np.mean(avg_prec).item()
    return precision_dct, recall_dct, micro_prec_dct, pcr_dct, avg_precision


def get_min_max_ranks(indices, data_ids: List[str], data2gold: Dict[str, Set[str]], e2i: Dict[str, int]):
    """
    Get ranks of gold labels in INDICES gathered from predictions and Nearest Neighbor search.
    Returns both the the minimum and maximum rank of gold entities among predicted indices.
    :param indices: numpy array of shape (num_data_ids, num_neighbors) containing predicted entity indices
    :param data_ids: data ids in the same order as indices
    :param data2gold: dict mapping data id to set of topic ids
    :param e2i: dict mapping entity id to topic index
    :return: numpy array of shape (num_data_ids,) with lowest rank of gold label, -1 if not found
    """
    min_ranks = np.full(indices.shape[0], -1, dtype=float)
    max_ranks = np.full(indices.shape[0], -1, dtype=float)
    i = 0
    for idxs, data_id in zip(indices, data_ids):
        gold = data2gold[data_id]
        gold_idxs = np.array([e2i[g] for g in gold])
        found = np.argwhere(idxs.reshape(-1, 1) == gold_idxs.reshape(1, -1))[:, 0]
        if len(found) > 0:
            min_ranks[i] = min(found)
            max_ranks[i] = max(found)
        else:
            min_ranks[i] = np.inf
            max_ranks[i] = np.inf
        i += 1
    return min_ranks, max_ranks


def get_mean_inverse_rank(ranks) -> float:
    """Compute mean inverse rank for RANKS of shape (num_examples,)."""
    mir = 1.0 / (ranks + 1)
    return np.mean(mir).item()


def get_log_mir_metrics(indices,
                        data_ids: List[str], e2i: Dict[str, int], t2gold: Dict[str, Set[str]],
                        global_step: int, run: Run) -> None:
    """Compare with gold, compute and log rank metrics."""
    min_ranks, max_ranks = get_min_max_ranks(indices, data_ids, t2gold, e2i)
    min_mir = get_mean_inverse_rank(min_ranks)
    max_mir = get_mean_inverse_rank(max_ranks)

    print(f"Evaluation inference mode mean inverse min rank: {min_mir:.5}")
    print(f"Evaluation inference mode mean inverse max rank: {max_mir:.5}")

    run["val/min_mir"].log(min_mir, step=global_step)
    run["val/max_mir"].log(max_mir, step=global_step)
