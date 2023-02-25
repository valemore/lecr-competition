import numpy as np

from config import CFG
from metrics import np_fscore
from utils import safe_div_np

CROSS_EVAL_THRESHS = np.array([round(x, 2) for x in np.arange(-0.2, 0.2 + 0.01, 0.01)])


def get_num_tp_num_fp(probs, topic_ids, concatenated_content_ids, concatenated_cand_ids, num_cands):
    num_tp = np.zeros((len(topic_ids), len(CROSS_EVAL_THRESHS)), dtype=int) # Careful when used as float
    num_fp = np.zeros((len(topic_ids), len(CROSS_EVAL_THRESHS)), dtype=int) # Careful when used as float
    topic_idx = 0
    prob_idx = 0
    for topic_id, topic_content_ids, topic_cand_ids in zip(topic_ids, concatenated_content_ids, concatenated_cand_ids):
        gold_ids = set(topic_content_ids.split())
        cand_ids = set(topic_cand_ids.split()[:num_cands])
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
    num_tp, num_fp = get_num_tp_num_fp(probs, corr_df["topic_id"], corr_df["content_ids"], corr_df["cands"], CFG.cross_num_cands)
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
