# Metrics for the Bi-encoder
from typing import Dict, List, Set, Tuple

import numpy as np
from neptune.new import Run

from config import CFG
from typehints import MetricDict
from utils import safe_div_np

BIENC_EVAL_THRESHS = [round(x, 2) for x in np.arange(0.2, 0.62, 0.02)]
BIENC_STANDALONE_THRESHS = [round(x, 2) for x in np.arange(0.1, 0.52, 0.02)]


def get_i2c_tp_num_gold(indices, topic_ids: List[str], c2i: Dict[str, int], t2gold: Dict[str, Set[str]]):
    i2c = {content_idx: entity_id for entity_id, content_idx in c2i.items()}
    tp = np.empty_like(indices, dtype=int) # mask indicating whether prediction is a true positive
    num_gold = np.empty(len(topic_ids), dtype=int) # how many content ids are in gold?
    for i, (idxs, topic_id) in enumerate(zip(indices, topic_ids)):
        gold = t2gold[topic_id]
        tp[i, :] = np.array([int(i2c[idx] in gold) for idx in idxs], dtype=int)
        num_gold[i] = len(gold)
    return i2c, tp, num_gold


def get_bienc_thresh_metrics(distances, indices,
                             topic_ids: List[str], e2i: Dict[str, int], t2gold: Dict[str, Set[str]]) -> Tuple[MetricDict, MetricDict, MetricDict, MetricDict]:
    i2e, tp, num_gold = get_i2c_tp_num_gold(indices, topic_ids, e2i, t2gold)
    total_gold = np.sum(num_gold)

    precision_dct = {thresh: 0.0 for thresh in BIENC_EVAL_THRESHS}
    recall_dct = {thresh: 0.0 for thresh in BIENC_EVAL_THRESHS}
    micro_prec_dct = {thresh: 0.0 for thresh in BIENC_EVAL_THRESHS}
    pcr_dct = {thresh: 0.0 for thresh in BIENC_EVAL_THRESHS}

    for thresh in BIENC_EVAL_THRESHS:
        mask = distances <= thresh
        thresh_tp = np.copy(tp)
        thresh_tp[~mask] = 0
        num_tp = np.sum(thresh_tp, axis=1)
        num_preds = np.sum(mask, axis=1)
        prec = safe_div_np(num_tp, num_preds)
        rec = num_tp / num_gold
        precision_dct[thresh] = np.mean(prec)
        recall_dct[thresh] = np.mean(rec)

        num_tp_all = np.sum(thresh_tp)      # total true positive over all topic ids
        num_preds_all = np.sum(mask)        # total predictions over all topic ids
        num_fp_all = np.sum((1 - thresh_tp) * mask, axis=None)
        # Micro precision: Not per sample (i.e. per topic id), but looking at every topic-content pair individually
        micro_prec_dct[thresh] = num_tp_all / num_preds_all
        # Positive class ratio: What we expect the positive class ratio to be in gen_cross_data
        pcr_dct[thresh] = total_gold / (num_fp_all + total_gold)
    return precision_dct, recall_dct, micro_prec_dct, pcr_dct


def get_avg_precision_threshs(distances, indices, topic_ids: List[str], e2i: Dict[str, int], t2gold: Dict[str, Set[str]]) -> float:
    i2e, tp, num_gold = get_i2c_tp_num_gold(indices, topic_ids, e2i, t2gold)

    mesh = [round(x, 2) for x in np.arange(-1.0, 1.0 + 0.01, 0.01)]
    mesh.reverse()
    precs = np.empty((len(topic_ids), len(mesh)), dtype=float)
    recs = np.empty((len(topic_ids), len(mesh)), dtype=float)
    for i, thresh in enumerate(mesh):
        mask = distances > thresh
        tp[mask] = 0
        num_tp = np.sum(tp, axis=1)
        num_preds = np.sum(~mask, axis=1)
        precs[:, len(mesh) - i - 1] = safe_div_np(num_tp, num_preds)
        recs[:, len(mesh) - i - 1] = num_tp / num_gold
    avg_precision = precs * np.diff(np.concatenate([np.zeros((len(topic_ids), 1)), recs], axis=1), axis=1)
    avg_precision = np.mean(np.sum(avg_precision, axis=1)).item()
    return avg_precision


def log_dct(dct: Dict[int, float], label: str, global_step: int, run: Run):
    for k, v in dct.items():
        run[f"{label}@{k}"].log(v, step=global_step)


def get_bienc_cands_metrics(indices, topic_ids: List[str], c2i: Dict[str, int], t2gold: Dict[str, Set[str]], num_cands: int) -> Tuple[MetricDict, MetricDict, MetricDict, MetricDict]:
    i2e, tp, num_gold = get_i2c_tp_num_gold(indices, topic_ids, c2i, t2gold)

    mesh = list(range(1, num_cands + 1))
    precision_dct = {num_cands: 0.0 for num_cands in mesh}
    recall_dct = {num_cands: 0.0 for num_cands in mesh}
    micro_prec_dct = {num_cands: 0.0 for num_cands in mesh}
    pcr_dct = {num_cands: 0.0 for num_cands in mesh}

    acc_tp = np.zeros(len(topic_ids), dtype=float) # accumulating true positives for all topic ids
    for num_cands in mesh:
        acc_tp += tp[:, num_cands - 1]
        prec = acc_tp / num_cands
        rec = acc_tp / num_gold
        precision_dct[num_cands] = np.mean(prec)
        recall_dct[num_cands] = np.mean(rec)

        num_preds = len(topic_ids) * num_cands      # total predictions over all topic_ids
        num_fp = num_preds - np.sum(acc_tp)         # total false positive over all topic_ids
        num_gold_all =  np.sum(num_gold)            # total gold contents over all topic_ids
        # Micro precision: Not per sample (i.e. per topic id), but looking at every topic-content pair individually
        micro_prec_dct[num_cands] = np.sum(acc_tp) / num_preds
        # Positive class ratio: What we expect the positive class ratio to be in gen_cross_data
        pcr_dct[num_cands] = num_gold_all / ( num_fp + num_gold_all)
    return precision_dct, recall_dct, micro_prec_dct, pcr_dct


def get_average_precision_cands(indices, topic_ids: List[str], c2i: Dict[str, int], t2gold: Dict[str, Set[str]]) -> float:
    i2e, tp, num_gold = get_i2c_tp_num_gold(indices, topic_ids, c2i, t2gold)

    acc_tp = np.zeros(len(topic_ids), dtype=float) # accumulating true positives for all topic ids

    mesh = list(range(1, CFG.NUM_NEIGHBORS + 1))
    precs = np.empty((len(topic_ids), len(mesh)), dtype=float)
    recs = np.empty((len(topic_ids), len(mesh)), dtype=float)
    for num_cands in mesh:
        acc_tp += tp[:, num_cands - 1]
        precs[:, num_cands - 1] = acc_tp / num_cands
        recs[:, num_cands - 1] = acc_tp / num_gold
    avg_precision = precs * np.diff(np.concatenate([np.zeros((len(topic_ids), 1)), recs], axis=1), axis=1)
    avg_precision = np.mean(np.sum(avg_precision, axis=1)).item()
    return avg_precision


def get_min_max_ranks(indices, topic_ids: List[str], t2gold: Dict[str, Set[str]], c2i: Dict[str, int]):
    """
    Get ranks of gold labels in INDICES gathered from predictions and Nearest Neighbor search.
    Returns both the the minimum and maximum rank of gold entities among predicted indices.
    :param indices: numpy array of shape (num_data_ids, num_neighbors) containing predicted entity indices
    :param topic_ids: topic ids in the same order as indices
    :param t2gold: dict mapping topic id to set of topic ids
    :param c2i: dict mapping content id to topic index
    :return: numpy array of shape (num_data_ids,) with lowest rank of gold label, -1 if not found
    """
    min_ranks = np.full(indices.shape[0], -1, dtype=float)
    max_ranks = np.full(indices.shape[0], -1, dtype=float)
    i = 0
    for idxs, data_id in zip(indices, topic_ids):
        gold = t2gold[data_id]
        gold_idxs = np.array([c2i[g] for g in gold])
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
                        topic_ids: List[str], c2i: Dict[str, int], t2gold: Dict[str, Set[str]],
                        global_step: int, run: Run) -> None:
    """Compare with gold, compute and log rank metrics."""
    min_ranks, max_ranks = get_min_max_ranks(indices, topic_ids, t2gold, c2i)
    min_mir = get_mean_inverse_rank(min_ranks)
    max_mir = get_mean_inverse_rank(max_ranks)

    print(f"Evaluation inference mode mean inverse min rank: {min_mir:.5}")
    print(f"Evaluation inference mode mean inverse max rank: {max_mir:.5}")

    run["cands/min_mir"].log(min_mir, step=global_step)
    run["cands/max_mir"].log(max_mir, step=global_step)
