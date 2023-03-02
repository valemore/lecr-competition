from typing import List, Dict

import pandas as pd


def get_cand_df(topic_ids: List[str], indices, c2i: Dict[str, int], max_num_cands: int = None) -> pd.DataFrame:
    """
    Converts distances and indices obtained from NN search to dataframe containing candidate contents for all topics.
    :param topic_ids: all topic ids in order
    :param indices: indices - output from NN model (-1 indicates a non-matching language for that content)
    :param c2i: dct mapping topic id to topic idx
    :param max_num_cands: maximum number of candidates to include in the dataframe
    :return: dataframe with two columns: topic ids and concatenated candidate ids
    """
    i2c = {content_idx: content_id for content_id, content_idx in c2i.items()}
    all_topic_cand_ids = []
    for topic_id, pred_idxs in zip(topic_ids, indices):
        if max_num_cands:
            pred_idxs = pred_idxs[:max_num_cands]
        cands = [i2c[pred_idx] for pred_idx in pred_idxs if pred_idx != -1]
        if not cands:
            print(f"No matching language candidates for topic id {topic_id}!")
        all_topic_cand_ids.append(" ".join(cands))
    return pd.DataFrame({"topic_id": topic_ids, "cands": all_topic_cand_ids})
