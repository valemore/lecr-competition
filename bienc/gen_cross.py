from config import CFG


def gen_cross_df(distances, indices, corr_df, e2i):
    """
    Generates data frame that is training input for the crossencoder.
    :param distances: numpy array of shape (num_topic_ids, num_cands) - needed for thresh approach candidate generation
    :param indices: numpy array of shape (num_topic_ids, num_cands)
    :param corr_df: dataframe containing correlations.csv
    :param e2i: dct mapping entities to idxs
    :return: corr_df with an additional column 'cands' containing the concatenated candidate ids
    """
    i2e = {idx: content_id for content_id, idx in e2i.items()}
    all_cands = []
    for pred_idxs in indices:
        pred_idxs = pred_idxs[:CFG.MAX_NUM_CANDS]
        cands = [i2e[pred_idx] for pred_idx in pred_idxs]
        all_cands.append(" ".join(cands))
    gen_df = corr_df.copy()
    gen_df["cands"] = all_cands
    return gen_df
