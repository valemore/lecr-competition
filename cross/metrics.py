from typing import Tuple

import numpy as np
import pandas as pd

from metrics import np_fscore, single_fscore, fscore_from_prec_rec
from utils import safe_div_np, safe_div

CROSS_EVAL_THRESHS = np.array([round(x, 3) for x in np.arange(0.001, 1.0 + 0.001, 0.001)])


def get_num_tp_num_fp(probs, topic_ids, concatenated_content_ids, concatenated_cand_ids):
    num_tp = np.zeros((len(topic_ids), len(CROSS_EVAL_THRESHS)), dtype=int) # Careful when used as float
    num_fp = np.zeros((len(topic_ids), len(CROSS_EVAL_THRESHS)), dtype=int) # Careful when used as float
    topic_idx = 0
    prob_idx = 0
    for topic_id, topic_content_ids, topic_cand_ids in zip(topic_ids, concatenated_content_ids, concatenated_cand_ids):
        gold_ids = set(topic_content_ids.split())
        cand_ids = set(topic_cand_ids.split())
        positive_ids = sorted(list(cand_ids & gold_ids))
        negative_ids = sorted(list(cand_ids - gold_ids))
        for _ in positive_ids:
            num_tp[topic_idx, :] += probs[prob_idx] >= CROSS_EVAL_THRESHS
            prob_idx += 1
        for _ in negative_ids:
            num_fp[topic_idx, :] += probs[prob_idx] >= CROSS_EVAL_THRESHS
            prob_idx += 1
        topic_idx += 1
    return num_tp, num_fp


def get_cross_f2(probs, corr_df):
    num_tp, num_fp = get_num_tp_num_fp(probs, corr_df["topic_id"], corr_df["content_ids"], corr_df["cands"])
    num_tp, num_fp = num_tp.astype(float), num_fp.astype(float)
    num_gold = np.array([len(x.split()) for x in corr_df["content_ids"]], dtype=float)

    precs = safe_div_np(num_tp, (num_tp + num_fp))
    recs = num_tp / num_gold.reshape(-1, 1)

    fscores = np_fscore(precs, recs, 2.0)
    fscores = np.mean(fscores, axis=0)

    return fscores


def log_fscores(fscores, step, run):
    for thresh, fscore in zip(CROSS_EVAL_THRESHS, fscores):
        run[f"cross/f2@{thresh}"].log(fscore, step=step)

    best_idx = np.argmax(fscores)
    best_thresh = CROSS_EVAL_THRESHS[best_idx]
    best_f2 = fscores[best_idx]

    print(f"Best F2 @ {best_thresh:.5}: {best_f2:.5}")
    run["cross/best_thresh"].log(best_thresh, step=step)
    run["cross/best_f2"].log(best_f2, step=step)


def get_positive_class_ratio(corr_df):
    acc_num_examples = 0
    acc_num_positives = 0
    for cat_gold_ids, cat_cand_ids in zip(corr_df["content_ids"], corr_df["cands"]):
        gold_ids = set(cat_gold_ids.split())
        cand_ids = set(cat_cand_ids.split())
        negative_ids = cand_ids - gold_ids
        acc_num_examples += len(gold_ids) + len(negative_ids)
        acc_num_positives += len(gold_ids)
    return acc_num_positives / acc_num_examples


def sanity_check_bienc_only(corr_df: pd.DataFrame) -> Tuple[float, float]:
    recs = np.empty(len(corr_df), dtype=float)
    scores = np.empty(len(corr_df), dtype=float)
    for i, (cat_content_ids, cat_cand_ids) in enumerate(zip(corr_df["content_ids"], corr_df["cands"])):
        gold = set(cat_content_ids.split())
        pred = set(cat_cand_ids.split())
        recs[i] = len(pred & gold) / len(gold)
        scores[i] = single_fscore(gold, pred, 2.0)
    return np.mean(recs).item(), np.mean(scores).item()


def get_sanity_micro(all_probs, val_dset):
    sanity_pred = (all_probs >= 0.5).astype(float)
    sanity_labels = np.array(val_dset.labels, dtype=float)
    sanity_tp = np.sum(sanity_pred * sanity_labels).item()
    sanity_fp = np.sum(sanity_pred * (1 - sanity_labels)).item()
    sanity_fn = np.sum((1 - sanity_pred) * sanity_labels).item()
    sanity_prec = safe_div(sanity_tp, sanity_tp + sanity_fp)
    sanity_rec = safe_div(sanity_tp, sanity_tp + sanity_fn)
    sanity_f2 = fscore_from_prec_rec(sanity_prec, sanity_rec)
    return sanity_prec, sanity_rec, sanity_f2
