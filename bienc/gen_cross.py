from typing import List, Dict

import pandas as pd

from config import CFG


def gen_cross_df(indices, topic_ids: List[str], c2i: Dict[str, int]) -> pd.DataFrame:
    """
    Generates data frame that is training input for the crossencoder.
    :param indices: numpy array of shape (num_topic_ids, num_cands)
    :param topic_ids: topic ids for which we are generrating candidates
    :param c2i: dct mapping content ids to idxs
    :return: df with columns 'topic_id' and 'cands' containing the concatenated candidate ids
    """
    i2e = {idx: content_id for content_id, idx in c2i.items()}
    all_cands = []
    for pred_idxs in indices:
        pred_idxs = pred_idxs[:CFG.MAX_NUM_CANDS]
        cands = [i2e[pred_idx] for pred_idx in pred_idxs if pred_idx != -1]
        all_cands.append(" ".join(cands))
    gen_df = pd.DataFrame({"topic_id": topic_ids, "cands": all_cands})
    return gen_df
